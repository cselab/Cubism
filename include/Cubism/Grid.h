#pragma once

#include <algorithm>
#include <omp.h>
#include <unordered_map>

#include "BlockInfo.h"
#include "FluxCorrection.h"

namespace cubism // AMR_CUBISM
{

struct BlockGroup
{
   /*
    * When dumping the Grid, blocks are grouped into larger rectangular regions
    * of uniform resolution. These regions (BlockGroups) have blocks with the 
    * same level and with various Space-filling-curve coordinates Z.
    * They have NXX x NYY x NZZ grid points, grid spacing h, an origin and a
    * minimum and maximum index (indices of bottom left and top right blocks).
    */
   int i_min[3];
   int i_max[3];
   int level;
   std::vector<long long> Z;
   size_t ID;
   double origin[3];
   double h;
   int NXX;
   int NYY;
   int NZZ;
};

struct TreePosition
{
   /*
    * Single integer used to recognize if a Block exists in the Grid and
    * by which MPI rank it is owned. 
    */
   int position{-3};
   bool CheckCoarser() const { return position == -2; }
   bool CheckFiner() const { return position == -1; }
   bool Exists() const { return position >= 0; }
   int rank() const { return position; }
   void setrank(const int r) { position = r; }
   void setCheckCoarser() { position = -2; }
   void setCheckFiner() { position = -1; }
};

template <typename Block, template <typename X> class allocator = std::allocator>
class Grid
{
   /*
    * This class holds the blocks and their meta-data (BlockInfos).
    * The BlockInfoAll and Octree objects hold information for the blocks owned by this rank
    * and its neighboring ranks (when running with MPI).
    */
 public:
   typedef Block BlockType;
   using ElementType = typename Block::ElementType; //Blocks hold ElementTypes
   typedef typename Block::RealType Real; // Block MUST provide `RealType`.

   #ifdef CUBISM_USE_MAP
   std::unordered_map<long long, BlockInfo *> BlockInfoAll;
   std::unordered_map<long long, TreePosition> Octree;
   #else
   std::vector<std::vector<BlockInfo *>> BlockInfoAll;
   std::vector<std::vector<TreePosition>> Octree;
   #endif

   std::vector<BlockInfo> m_vInfo;   // meta-data for blocks that belong to this rank
   std::vector<Block *> m_blocks;    // pointers to blocks that belong to this rank
   const int NX;                     // Total # of blocks for level 0 in X-direction
   const int NY;                     // Total # of blocks for level 0 in Y-direction
   const int NZ;                     // Total # of blocks for level 0 in Z-direction
   const double maxextent;           // Maximum domain extent
   const int levelMax;               // Maximum refinement level allowed
   const int levelStart;             // Initial refinement level
   const bool xperiodic;             // grid periodicity in x-direction
   const bool yperiodic;             // grid periodicity in y-direction
   const bool zperiodic;             // grid periodicity in z-direction
   std::vector<BlockGroup> MyGroups; // used for dumping data
   std::vector<long long> level_base;// auxiliary array used when searching is std::unordered_map
   bool UpdateFluxCorrection{true};  // FluxCorrection updates only when grid is refined/compressed
   bool UpdateGroups{true};          // (inactive) BlockGroups updated only when this is true
   bool FiniteDifferences{true};     // used by BlockLab, to determine what kind of coarse-fine 
                                     // interface interpolation to make.
                                     // true means that biased stencils will be used to 
                                     // get an O(h^3) approximation
   FluxCorrection<Grid,Block> CorrectorGrid; // used for AMR flux-corrections at coarse-fine 
                                             // interfaces

   TreePosition &Tree(int m, long long n)
   {
      /*
       * Return the position in the Octree of a Block at level m and SFC coordinate n.
       */
      #ifdef CUBISM_USE_MAP
      long long aux = level_base[m] + n;
      auto retval   = Octree.find(aux);
      if (retval == Octree.end())
      {
         #pragma omp critical
         {
            auto retval1 = Octree.find(aux);
            if (retval1 == Octree.end())
            {
               TreePosition dum;
               Octree[aux] = dum;
            }
         }
         return Tree(m, n);
      }
      else
      {
         return retval->second;
      }
      #else
      return Octree[m][n];
      #endif
   }
   TreePosition &Tree(BlockInfo &info) { return Tree(info.level, info.Z); }
   TreePosition &Tree(const BlockInfo &info) { return Tree(info.level, info.Z); }

   void _alloc() // called in class constructor
   {
      const int m        = levelStart;
      const int TwoPower = 1 << m;
      for (long long n = 0; n < NX * NY * NZ * pow(TwoPower, DIMENSION); n++)
      {
         Tree(m, n).setrank(0);
         _alloc(m, n);
      }
      if (m - 1 >= 0)
      {
         for (long long n = 0; n < NX * NY * NZ * pow((1 << (m - 1)), DIMENSION); n++) Tree(m - 1, n).setCheckFiner();
      }
      if (m + 1 < levelMax)
      {
         for (long long n = 0; n < NX * NY * NZ * pow((1 << (m + 1)), DIMENSION); n++) Tree(m + 1, n).setCheckCoarser();
      }
      FillPos();
   }

   void _alloc(int m, long long n) // called whenever the grid is refined
   {
      allocator<Block> alloc;
      getBlockInfoAll(m, n).ptrBlock    = alloc.allocate(1);
      m_blocks.push_back((Block *)getBlockInfoAll(m, n).ptrBlock);
      m_vInfo.push_back(getBlockInfoAll(m, n));
      Tree(m, n).setrank(rank());
   }

   void _deallocAll() // called in class destructor
   {
      allocator<Block> alloc;
      for (size_t i = 0; i < m_vInfo.size(); i++)
      {
         const int m       = m_vInfo[i].level;
         const long long n = m_vInfo[i].Z;
         alloc.deallocate((Block *)getBlockInfoAll(m, n).ptrBlock, 1);
      }
      #ifdef CUBISM_USE_MAP
      std::vector<long long> aux;
      for (auto & m: BlockInfoAll)
         aux.push_back(m.first);
      for (size_t i = 0 ; i < aux.size() ; i++)
      {
         auto retval = BlockInfoAll.find(aux[i]);
         if (retval != BlockInfoAll.end())
         {
            delete retval->second;
         }
      }
      #else
      for (int m = 0; m < levelMax; m++)
      {
         const long long nmax = getMaxBlocks()[0] * getMaxBlocks()[1] * getMaxBlocks()[2] * pow(1 << m, DIMENSION);
         for (long long n = 0; n < nmax; n++)
         {
            if (Tree(m, n).position != -3)
            {
               delete BlockInfoAll[m][n];
            }
         }
      }
      #endif
      m_blocks.clear();
      m_vInfo.clear();
      BlockInfoAll.clear();
      Octree.clear();
   }

   void _dealloc(int m, long long n) // called whenever the grid is compressed
   {
      allocator<Block> alloc;
      alloc.deallocate((Block *)getBlockInfoAll(m, n).ptrBlock, 1);
      for (size_t j = 0; j < m_vInfo.size(); j++)
      {
         if (m_vInfo[j].level == m && m_vInfo[j].Z == n)
         {
            m_vInfo.erase(m_vInfo.begin() + j);
            m_blocks.erase(m_blocks.begin() + j);
            return;
         }
      }
   }

   void FindBlockInfo(int m, long long n, int m_new, long long n_new)
   {
      /*
       * Used when Block at level m_new with SFC coordinate n_new is added to the Grid
       * as a result of compression of Block (m,n). Sets the state of the newly added 
       * Block. It also replaces BlockInfo(m,n) and Block(m,n) with 
       * BlockInfo(m_new,n_new) and Block(m_new,n_new).
       */
      for (size_t j = 0; j < m_vInfo.size(); j++)
      {
         if (m == m_vInfo[j].level && n == m_vInfo[j].Z)
         {
            getBlockInfoAll(m_new, n_new).state = Leave;
            m_vInfo[j]                          = getBlockInfoAll(m_new, n_new);
            m_blocks[j]                         = (Block *)getBlockInfoAll(m_new, n_new).ptrBlock;
            return;
         }
      }
   }

   virtual void FillPos(bool CopyInfos = true)
   {
      /*
       * The data in BlockInfoAll is always correct (states, blockIDs etc.), but this 
       * is not the case for m_vInfo and m_blocks, whose content might be outdated
       * after grid refinement/compression or exchange of blocks between different 
       * ranks. This function updates their content.
       */
      for (size_t i = 0; i < m_vInfo.size(); i++) m_vInfo[i].blockID = i;
      std::sort(m_vInfo.begin(), m_vInfo.end());
      std::vector<size_t> permutation(m_vInfo.size());
      for (size_t i = 0; i < m_vInfo.size(); i++) permutation[i] = m_vInfo[i].blockID;
      apply_permutation_in_place<Block *>(this->m_blocks, permutation);
      for (size_t i = 0; i < m_vInfo.size(); i++) m_vInfo[i].blockID = i;

      if (CopyInfos)
         for (size_t j = 0; j < m_vInfo.size(); j++)
         {
            int m       = m_vInfo[j].level;
            long long n = m_vInfo[j].Z;
            m_vInfo[j]  = getBlockInfoAll(m, n);

            assert(getBlockInfoAll(m, n).state == m_vInfo[j].state);
            assert(Tree(m, n).Exists());

            m_blocks[j] = (Block *)getBlockInfoAll(m, n).ptrBlock;
         }
      else
         for (size_t j = 0; j < m_vInfo.size(); j++)
         {
            int m            = m_vInfo[j].level;
            long long n      = m_vInfo[j].Z;
            m_vInfo[j].state = getBlockInfoAll(m, n).state;
            assert(Tree(m, n).Exists());
            m_blocks[j] = (Block *)getBlockInfoAll(m, n).ptrBlock;
         }
      for (size_t j = 0; j < m_vInfo.size(); j++)
      {
         int m                         = m_vInfo[j].level;
         long long n                   = m_vInfo[j].Z;
         m_vInfo[j].blockID            = j;
         getBlockInfoAll(m, n).blockID = j;
      }
   }

   Grid(const unsigned int _NX, 
        const unsigned int _NY = 1,
        const unsigned int _NZ = 1,
        const double _maxextent = 1,
        const unsigned int _levelStart = 0,
        const unsigned int _levelMax = 1,
        const bool AllocateBlocks = true,
        const bool a_xperiodic = true,
        const bool a_yperiodic = true,
        const bool a_zperiodic = true)
       : NX(_NX), NY(_NY), NZ(_NZ), maxextent(_maxextent), levelMax(_levelMax), levelStart(_levelStart),
         xperiodic(a_xperiodic), yperiodic(a_yperiodic), zperiodic(a_zperiodic)
   {
      BlockInfo dummy;
      #if DIMENSION == 3
      int nx = dummy.blocks_per_dim(0, NX, NY, NZ);
      int ny = dummy.blocks_per_dim(1, NX, NY, NZ);
      int nz = dummy.blocks_per_dim(2, NX, NY, NZ);
      #else
      int nx = dummy.blocks_per_dim(0, NX, NY);
      int ny = dummy.blocks_per_dim(1, NX, NY);
      int nz = 1;
      #endif
      int lvlMax = dummy.levelMax(levelMax);

      #ifdef CUBISM_USE_MAP
      for (int m = 0; m < lvlMax; m++)
      {
         const int TwoPower   = 1 << m;
         const long long Ntot = nx * ny * nz * pow(TwoPower, DIMENSION);
         if (m == 0) level_base.push_back(Ntot);
         if (m > 0) level_base.push_back(level_base[m - 1] + Ntot);
      }
      #else
      BlockInfoAll.resize(lvlMax);
      Octree.resize(lvlMax);
      for (int m = 0; m < lvlMax; m++)
      {
         const int TwoPower   = 1 << m;
         const long long Ntot = nx * ny * nz * pow(TwoPower, DIMENSION);
         if (m == 0) level_base.push_back(Ntot);
         if (m > 0) level_base.push_back(level_base[m - 1] + Ntot);
         BlockInfoAll[m].resize(Ntot, nullptr);
         Octree[m].resize(Ntot);
      }
      #endif
      if (AllocateBlocks) _alloc();
   }

   virtual ~Grid() { _deallocAll(); }

   virtual Block *avail(int m, long long n) { return (Block *)getBlockInfoAll(m, n).ptrBlock; }

   virtual int rank() const { return 0; }


   virtual void initialize_blocks(const std::vector<long long> & blocksZ, const std::vector<short int> & blockslevel)
   {
      //Given two vectors with the SFC coordinate (Z) and the level of each block, this function
      //will erase the current structure of the grid and create a new one, with the given blocks.
      //This is used when reading data from file (possibly to restart) or when initializing the
      //simulation.
      _deallocAll();
      for (size_t i = 0 ; i < blocksZ.size() ; i++)
      {
         const int level   = blockslevel[i];
         const long long Z = blocksZ[i];

         _alloc(level, Z);
         Tree(level, Z).setrank(rank());

         #if DIMENSION == 3
            int p[3];
            BlockInfo::inverse(Z, level, p[0], p[1], p[2]);
            if (level < levelMax - 1)
               for (int k1 = 0; k1 < 2; k1++)
               for (int j1 = 0; j1 < 2; j1++)
               for (int i1 = 0; i1 < 2; i1++)
               {
                  const long long nc = getZforward(level + 1, 2 * p[0] + i1, 2 * p[1] + j1, 2 * p[2] + k1);
                  Tree(level + 1, nc).setCheckCoarser();
               }
            if (level > 0)
            {
               const long long nf = getZforward(level - 1, p[0] / 2, p[1] / 2, p[2] / 2);
               Tree(level - 1, nf).setCheckFiner();
            }
         #else
            int p[2];
            BlockInfo::inverse(Z, level, p[0], p[1]);
            if (level < levelMax - 1)
               for (int j1 = 0; j1 < 2; j1++)
               for (int i1 = 0; i1 < 2; i1++)
               {
                  const long long nc = getZforward(level + 1, 2 * p[0] + i1, 2 * p[1] + j1);
                  Tree(level + 1, nc).setCheckCoarser();
               }
            if (level > 0)
            {
               const long long nf = getZforward(level - 1, p[0] / 2, p[1] / 2);
               Tree(level - 1, nf).setCheckFiner();
            }
         #endif
      }
      FillPos(true);
      UpdateFluxCorrection = true;
      UpdateGroups = true;
   }


   #if DIMENSION == 3
   long long getZforward(const int level, const int i, const int j, const int k) const
   {
      const int TwoPower = 1 << level;
      const int ix       = (i + TwoPower * NX) % (NX * TwoPower);
      const int iy       = (j + TwoPower * NY) % (NY * TwoPower);
      const int iz       = (k + TwoPower * NZ) % (NZ * TwoPower);
      return BlockInfo::forward(level, ix, iy, iz);
   }
   Block &operator()(int ix, int iy, int iz, int m)
   {
      const long long n = getZforward(m, ix, iy, iz);
      return *(Block *)getBlockInfoAll(m, n).ptrBlock;
   }
   Block *avail1(int ix, int iy, int iz, int m)
   {
      const long long n = getZforward(m, ix, iy, iz);
      return avail(m, n);
   }
   #else // DIMENSION = 2
   long long getZforward(const int level, const int i, const int j) const
   {
      const int TwoPower = 1 << level;
      const int ix       = (i + TwoPower * NX) % (NX * TwoPower);
      const int iy       = (j + TwoPower * NY) % (NY * TwoPower);
      return BlockInfo::forward(level, ix, iy);
   }
   Block &operator()(int ix, int iy, int m)
   {
      const long long n = getZforward(m, ix, iy);
      return *(Block *)getBlockInfoAll(m, n).ptrBlock;
   }
   Block *avail1(int ix, int iy, int m)
   {
      const long long n = getZforward(m, ix, iy);
      return avail(m, n);
   }
   #endif

   std::array<int, 3> getMaxBlocks() const { return {NX, NY, NZ}; }
   std::array<int, 3> getMaxMostRefinedBlocks() const
   {
     return {
       NX << (levelMax - 1),
       NY << (levelMax - 1),
       DIMENSION == 3 ? (NZ << (levelMax - 1)) : 1,
     };
   }
   std::array<int, 3> getMaxMostRefinedCells() const
   {
     const auto b = getMaxMostRefinedBlocks();
     return {b[0] * Block::sizeX, b[1] * Block::sizeY, b[2] * Block::sizeZ};
   }

   inline int getlevelMax() const { return levelMax; }

   BlockInfo &getBlockInfoAll(int m, long long n)
   {
      /*
       * Access BlockInfo at level m with Space-Filling-Curve coordinate n.
       * If the BlockInfo has not been allocated (not found in the std::unordered_map), 
       * allocate it as well.
       */
      #ifdef CUBISM_USE_MAP
      long long aux = level_base[m] + n;
      auto retval   = BlockInfoAll.find(aux);
      if (retval != BlockInfoAll.end())
      {
         return *retval->second;
      }
      else
      {
         #pragma omp critical
         {
            auto retval1 = BlockInfoAll.find(aux);
            if (retval1 == BlockInfoAll.end())
            {
               BlockInfo *dumm = new BlockInfo();
               int TwoPower    = 1 << m;
               double h0 = (maxextent / std::max(NX * Block::sizeX, std::max(NY * Block::sizeY, NZ * Block::sizeZ)));
               double h  = h0 / TwoPower;
               double origin[3];
               int i, j, k;
               #if DIMENSION == 3
               BlockInfo::inverse(n, m, i, j, k);
               #else
               BlockInfo::inverse(n, m, i, j);
               k = 0;
               #endif
               origin[0] = i * Block::sizeX * h;
               origin[1] = j * Block::sizeY * h;
               origin[2] = k * Block::sizeZ * h;
               dumm->setup(m, h, origin, n);
               BlockInfoAll[aux] = dumm;
            }
         }
         return getBlockInfoAll(m, n);
      }
      #else
      if (BlockInfoAll[m][n] != nullptr)
      {
         return *BlockInfoAll[m][n];
      }
      else
      {
         #pragma omp critical
         {
            if (BlockInfoAll[m][n] == nullptr)
            {
               BlockInfo *dummy = new BlockInfo();
               int TwoPower     = 1 << m;
               double h0 = (maxextent / std::max(NX * Block::sizeX, std::max(NY * Block::sizeY, NZ * Block::sizeZ)));
               double h  = h0 / TwoPower;
               double origin[3];
               int i, j, k;
               #if DIMENSION == 3
               BlockInfo::inverse(n, m, i, j, k);
               #else
               BlockInfo::inverse(n, m, i, j);
               k = 0;
               #endif
               origin[0] = i * Block::sizeX * h;
               origin[1] = j * Block::sizeY * h;
               origin[2] = k * Block::sizeZ * h;
               dummy->setup(m, h, origin, n);
               BlockInfoAll[m][n] = dummy;
            }
         }
         return *BlockInfoAll[m][n];
      }
      #endif
   }

   inline std::vector<Block *> &GetBlocks() { return m_blocks; }
   inline const std::vector<Block *> &GetBlocks() const { return m_blocks; }

   std::vector<BlockInfo> &getBlocksInfo() { return m_vInfo; }
   const std::vector<BlockInfo> &getBlocksInfo() const { return m_vInfo; }

   template <typename T>
   void apply_permutation_in_place(std::vector<T> &vec, std::vector<std::size_t> &p)
   {
      std::vector<bool> done(vec.size(), false);
      for (std::size_t i = 0; i < vec.size(); ++i)
      {
         if (done[i])
         {
            continue;
         }
         done[i]            = true;
         std::size_t prev_j = i;
         std::size_t j      = p[i];
         while (i != j)
         {
            std::swap(vec[prev_j], vec[j]);
            done[j] = true;
            prev_j  = j;
            j       = p[j];
         }
      }
   }

   virtual int get_world_size() const { return 1; }

   virtual void UpdateBoundary(bool clean=false){} //does nothing for a single node (no MPI)

   void UpdateMyGroups()
   {
      /*
       * This function is used before dumping the Grid. It groups adjacent blocks of the same
       * resolution (and owned by the same MPI rank) to BlockGroups that will be dumped as
       * a collection of rectangular uniform grids.
       */

      // if (!UpdateGroups) return; //TODO : does not work for CUP2D
      if (rank() == 0) std::cout << "Updating groups..." << std::endl;

      const unsigned int nX = BlockType::sizeX;
      const unsigned int nY = BlockType::sizeY;
      const size_t Ngrids   = getBlocksInfo().size();
      const auto &MyInfos   = getBlocksInfo();
      UpdateGroups          = false;
      MyGroups.clear();
      std::vector<bool> added(MyInfos.size(), false);
      FillPos();

      #if DIMENSION == 3
      const unsigned int nZ = BlockType::sizeZ;
      for (unsigned int m = 0; m < Ngrids; m++)
      {
         const BlockInfo &I = MyInfos[m];

         if (added[I.blockID]) continue;
         added[I.blockID] = true;
         BlockGroup newGroup;

         newGroup.level = I.level;
         newGroup.h     = I.h;
         newGroup.Z.push_back(I.Z);

         const int base[3] = {I.index[0], I.index[1], I.index[2]};
         int i_off[6] = {};
         bool ready_[6] = {};

         int d    = 0;
         auto blk = getMaxBlocks();
         do
         {
            if (ready_[d] == false)
            {
               bool valid = true;
               i_off[d]++;
               const int i0 = (d < 3) ? (base[d] - i_off[d]) : (base[d - 3] + i_off[d]);
               const int d0 = (d < 3) ? (d) % 3 : (d - 3) % 3;
               const int d1 = (d < 3) ? (d + 1) % 3 : (d - 3 + 1) % 3;
               const int d2 = (d < 3) ? (d + 2) % 3 : (d - 3 + 2) % 3;

               for (int i2 = base[d2] - i_off[d2]; i2 <= base[d2] + i_off[d2 + 3]; i2++)
                  for (int i1 = base[d1] - i_off[d1]; i1 <= base[d1] + i_off[d1 + 3]; i1++)
                  {
                     if (valid == false) break;

                     if (i0 < 0 || i1 < 0 || i2 < 0 || i0 >= blk[d0] * (1 << I.level) ||
                         i1 >= blk[d1] * (1 << I.level) || i2 >= blk[d2] * (1 << I.level))
                     {
                        valid = false;
                        break;
                     }
                     long long n;
                     if (d == 0 || d == 3) n = getZforward(I.level, i0, i1, i2);
                     else if (d == 1 || d == 4)
                        n = getZforward(I.level, i2, i0, i1);
                     else /*if (d==2||d==5)*/
                        n = getZforward(I.level, i1, i2, i0);

                     if (Tree(I.level, n).rank() != rank())
                     {
                        valid = false;
                        break;
                     }
                     if (added[getBlockInfoAll(I.level, n).blockID] == true)
                     {
                        valid = false;
                     }
                  }

               if (valid == false)
               {
                  i_off[d]--;
                  ready_[d] = true;
               }
               else
               {
                  for (int i2 = base[d2] - i_off[d2]; i2 <= base[d2] + i_off[d2 + 3]; i2++)
                     for (int i1 = base[d1] - i_off[d1]; i1 <= base[d1] + i_off[d1 + 3]; i1++)
                     {
                        long long n;
                        if (d == 0 || d == 3) n = getZforward(I.level, i0, i1, i2);
                        else if (d == 1 || d == 4)
                           n = getZforward(I.level, i2, i0, i1);
                        else /*if (d==2||d==5)*/
                           n = getZforward(I.level, i1, i2, i0);
                        newGroup.Z.push_back(n);
                        added[getBlockInfoAll(I.level, n).blockID] = true;
                     }
               }
            }
            d = (d + 1) % 6;
         } while (ready_[0] == false || ready_[1] == false || ready_[2] == false || ready_[3] == false ||
                  ready_[4] == false || ready_[5] == false);

         const int ix_min = base[0] - i_off[0];
         const int iy_min = base[1] - i_off[1];
         const int iz_min = base[2] - i_off[2];
         const int ix_max = base[0] + i_off[3];
         const int iy_max = base[1] + i_off[4];
         const int iz_max = base[2] + i_off[5];

         long long n_base = getZforward(I.level, ix_min, iy_min, iz_min);

         newGroup.i_min[0] = ix_min;
         newGroup.i_min[1] = iy_min;
         newGroup.i_min[2] = iz_min;

         newGroup.i_max[0] = ix_max;
         newGroup.i_max[1] = iy_max;
         newGroup.i_max[2] = iz_max;

         newGroup.origin[0] = getBlockInfoAll(I.level, n_base).origin[0];
         newGroup.origin[1] = getBlockInfoAll(I.level, n_base).origin[1];
         newGroup.origin[2] = getBlockInfoAll(I.level, n_base).origin[2];

         newGroup.NXX = (newGroup.i_max[0] - newGroup.i_min[0] + 1) * nX + 1;
         newGroup.NYY = (newGroup.i_max[1] - newGroup.i_min[1] + 1) * nY + 1;
         newGroup.NZZ = (newGroup.i_max[2] - newGroup.i_min[2] + 1) * nZ + 1;

         MyGroups.push_back(newGroup);
      }
      #else
      for (unsigned int m = 0; m < Ngrids; m++)
      {
         const BlockInfo &I = MyInfos[m];

         if (added[I.blockID]) continue;
         added[I.blockID] = true;
         BlockGroup newGroup;

         newGroup.level = I.level;
         newGroup.h     = I.h;
         newGroup.Z.push_back(I.Z);

         const int base[3] = {I.index[0], I.index[1], 0};  // I.index[2]
         int i_off[4] = {};
         bool ready_[4] = {};

         int d    = 0;
         auto blk = getMaxBlocks();
         do
         {
            if (ready_[d] == false)
            {
               bool valid = true;
               i_off[d]++;
               const int i0 = (d < 2) ? (base[d] - i_off[d]) : (base[d - 2] + i_off[d]);
               const int d0 = (d < 2) ? (d) % 2 : (d - 2) % 2;
               const int d1 = (d < 2) ? (d + 1) % 2 : (d - 2 + 1) % 2;

               for (int i1 = base[d1] - i_off[d1]; i1 <= base[d1] + i_off[d1 + 2]; i1++)
               {
                  if (valid == false) break;

                  if (i0 < 0 || i1 < 0 || i0 >= blk[d0] * (1 << I.level) || i1 >= blk[d1] * (1 << I.level))
                  {
                     valid = false;
                     break;
                  }
                  long long n = (d == 0 || d == 2) ? getZforward(I.level, i0, i1) : getZforward(I.level, i1, i0);

                  if (Tree(I.level, n).rank() != rank())
                  {
                     valid = false;
                     break;
                  }
                  if (added[getBlockInfoAll(I.level, n).blockID] == true)
                  {
                     valid = false;
                  }
               }

               if (valid == false)
               {
                  i_off[d]--;
                  ready_[d] = true;
               }
               else
               {
                  for (int i1 = base[d1] - i_off[d1]; i1 <= base[d1] + i_off[d1 + 2]; i1++)
                  {
                     long long n = (d == 0 || d == 2) ? getZforward(I.level, i0, i1) : getZforward(I.level, i1, i0);
                     newGroup.Z.push_back(n);
                     added[getBlockInfoAll(I.level, n).blockID] = true;
                  }
               }
            }
            d = (d + 1) % 4;
         } while (ready_[0] == false || ready_[1] == false || ready_[2] == false || ready_[3] == false);

         const int ix_min = base[0] - i_off[0];
         const int iy_min = base[1] - i_off[1];
         const int iz_min = 0; // base[2] - i_off[2];
         const int ix_max = base[0] + i_off[2];
         const int iy_max = base[1] + i_off[3];
         const int iz_max = 0; // base[2] + i_off[5];

         long long n_base = getZforward(I.level, ix_min, iy_min);

         newGroup.i_min[0] = ix_min;
         newGroup.i_min[1] = iy_min;
         newGroup.i_min[2] = iz_min;

         newGroup.i_max[0] = ix_max;
         newGroup.i_max[1] = iy_max;
         newGroup.i_max[2] = iz_max;

         newGroup.origin[0] = getBlockInfoAll(I.level, n_base).origin[0];
         newGroup.origin[1] = getBlockInfoAll(I.level, n_base).origin[1];
         newGroup.origin[2] = getBlockInfoAll(I.level, n_base).origin[2];

         newGroup.NXX = (newGroup.i_max[0] - newGroup.i_min[0] + 1) * nX + 1;
         newGroup.NYY = (newGroup.i_max[1] - newGroup.i_min[1] + 1) * nY + 1;
         newGroup.NZZ = 2; //(newGroup.i_max[2] - newGroup.i_min[2] + 1)*nZ + 1;

         MyGroups.push_back(newGroup);
      }
      #endif
   }
};

} // namespace cubism
