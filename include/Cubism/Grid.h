#pragma once

#include <algorithm>
#include <omp.h>
#include <unordered_map>

#ifdef CUBISM_USE_NUMA
#include <numa.h>
#endif

#include "BlockInfo.h"

namespace cubism // AMR_CUBISM
{

struct BlockGroup
{
   int i_min[3];
   int i_max[3];
   int level;
   std::vector<int> Z;
   size_t ID;
   double origin[3];
   double h;
   int NXX;
   int NYY;
   int NZZ;
};

struct TreePosition
{
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
 protected:
#ifdef CUBISM_USE_MAP
   std::unordered_map<size_t, BlockInfo *> BlockInfoAll;
   std::unordered_map<size_t, TreePosition> Octree;
#else
   std::vector<std::vector<BlockInfo *>> BlockInfoAll;
   std::vector<std::vector<TreePosition>> Octree;
#endif
   std::vector<BlockInfo> m_vInfo; // meta-data for blocks that belong to this rank
   std::vector<Block *> m_blocks;  // pointers to blocks that belong to this rank

   const int NX;           // Total # of blocks for level 0 in X-direction
   const int NY;           // Total # of blocks for level 0 in Y-direction
   const int NZ;           // Total # of blocks for level 0 in Z-direction
   const double maxextent; // Maximum domain extent
   const int levelMax;     // Maximum refinement level allowed
   const int levelStart;   // Initial refinement level

   std::vector<size_t> level_base;

 public:
   typedef Block BlockType;
   typedef typename Block::RealType Real; // Block MUST provide `RealType`.

   const bool xperiodic;             // grid periodicity in x-direction
   const bool yperiodic;             // grid periodicity in y-direction
   const bool zperiodic;             // grid periodicity in z-direction
   std::vector<BlockGroup> MyGroups; // used for dumping data
   bool UpdateGroups{true};

   TreePosition &Tree(int m, int n)
   {
#ifdef CUBISM_USE_MAP
      size_t aux  = level_base[m] + n;
      auto retval = Octree.find(aux);
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

   bool UpdateFluxCorrection{true};

   void _alloc() // called in class constructor
   {
      const int m        = levelStart;
      const int TwoPower = 1 << m;
      for (int n = 0; n < NX * NY * NZ * pow(TwoPower, DIMENSION); n++)
      {
         Tree(m, n).setrank(0);
         _alloc(m, n);
      }
      if (m - 1 >= 0)
      {
         for (int n = 0; n < NX * NY * NZ * pow((1 << (m - 1)), DIMENSION); n++) Tree(m - 1, n).setCheckFiner();
      }
      if (m + 1 < levelMax)
      {
         for (int n = 0; n < NX * NY * NZ * pow((1 << (m + 1)), DIMENSION); n++) Tree(m + 1, n).setCheckCoarser();
      }
      FillPos();
   }

   void _alloc(int m, int n) // called whenever the grid is refined
   {
      allocator<Block> alloc;
      getBlockInfoAll(m, n).ptrBlock    = alloc.allocate(1);
      getBlockInfoAll(m, n).changed     = true;
      getBlockInfoAll(m, n).h_gridpoint = getBlockInfoAll(m, n).h;
      m_blocks.push_back((Block *)getBlockInfoAll(m, n).ptrBlock);
      m_vInfo.push_back(getBlockInfoAll(m, n));
      Tree(m, n).setrank(rank());
   }

   void _deallocAll() // called in class destructor
   {
      allocator<Block> alloc;
      for (size_t i = 0; i < m_vInfo.size(); i++)
      {
         const int m = m_vInfo[i].level;
         const int n = m_vInfo[i].Z;
         alloc.deallocate((Block *)getBlockInfoAll(m, n).ptrBlock, 1);
      }
      for (int m = 0; m < levelMax; m++)
      {
         const size_t nmax = getMaxBlocks()[0] * getMaxBlocks()[1] * getMaxBlocks()[2] * pow(1 << m, 3);
         for (size_t n = 0; n < nmax; n++)
         {
            if (Tree(m, n).position != -3)
            {
#ifndef CUBISM_USE_MAP
               delete BlockInfoAll[m][n];
#else
               size_t aux  = level_base[m] + n;
               auto retval = BlockInfoAll.find(aux);
               if (retval != BlockInfoAll.end())
               {
                  delete retval->second;
               }
#endif
            }
         }
      }
      m_blocks.clear();
      m_vInfo.clear();
      BlockInfoAll.clear();
   }

   void _dealloc(int m, int n) // called whenever the grid is compressed
   {
      allocator<Block> alloc;
      alloc.deallocate((Block *)getBlockInfoAll(m, n).ptrBlock, 1);
      Tree(m, n).setCheckCoarser();
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

   void FindBlockInfo(int m, int n, int m_new, int n_new)
   {
      for (size_t j = 0; j < m_vInfo.size(); j++)
      {
         if (m == m_vInfo[j].level && n == m_vInfo[j].Z)
         {
            getBlockInfoAll(m_new, n_new).state   = Leave;
            getBlockInfoAll(m_new, n_new).changed = true;
            m_vInfo[j]                            = getBlockInfoAll(m_new, n_new);
            m_blocks[j]                           = (Block *)getBlockInfoAll(m_new, n_new).ptrBlock;
            return;
         }
      }
   }

   virtual void FillPos(bool CopyInfos = true)
   {
      if (CopyInfos)
         for (size_t j = 0; j < m_vInfo.size(); j++)
         {
            int m      = m_vInfo[j].level;
            int n      = m_vInfo[j].Z;
            m_vInfo[j] = getBlockInfoAll(m, n);

            assert(getBlockInfoAll(m, n).state == m_vInfo[j].state);
            assert(Tree(m, n).Exists());

            m_blocks[j] = (Block *)getBlockInfoAll(m, n).ptrBlock;
         }
      else
         for (size_t j = 0; j < m_vInfo.size(); j++)
         {
            int m            = m_vInfo[j].level;
            int n            = m_vInfo[j].Z;
            m_vInfo[j].state = getBlockInfoAll(m, n).state;
            assert(Tree(m, n).Exists());
            m_blocks[j] = (Block *)getBlockInfoAll(m, n).ptrBlock;
         }
      for (size_t j = 0; j < m_vInfo.size(); j++)
      {
         int m                         = m_vInfo[j].level;
         int n                         = m_vInfo[j].Z;
         m_vInfo[j].blockID            = j;
         getBlockInfoAll(m, n).blockID = j;
      }
   }

   Grid(const unsigned int _NX, const unsigned int _NY = 1, const unsigned int _NZ = 1, const double _maxextent = 1, const unsigned int _levelStart = 0, const unsigned int _levelMax = 1, const bool AllocateBlocks = true, const bool a_xperiodic = true, const bool a_yperiodic = true, const bool a_zperiodic = true) : NX(_NX), NY(_NY), NZ(_NZ), maxextent(_maxextent), levelMax(_levelMax), levelStart(_levelStart), xperiodic(a_xperiodic), yperiodic(a_yperiodic), zperiodic(a_zperiodic)
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
         int TwoPower            = 1 << m;
         const unsigned int Ntot = nx * ny * nz * pow(TwoPower, DIMENSION);
         if (m == 0) level_base.push_back(Ntot);
         if (m > 0) level_base.push_back(level_base[m - 1] + Ntot);
      }
#else
      BlockInfoAll.resize(lvlMax);
      Octree.resize(lvlMax);
      for (int m = 0; m < lvlMax; m++)
      {
         int TwoPower            = 1 << m;
         const unsigned int Ntot = nx * ny * nz * pow(TwoPower, DIMENSION);
         if (m == 0) level_base.push_back(Ntot);
         if (m > 0) level_base.push_back(level_base[m - 1] + Ntot);
         BlockInfoAll[m].resize(Ntot, nullptr);
         Octree[m].resize(Ntot);
      }
#endif
      if (AllocateBlocks) _alloc();
   }

   virtual ~Grid() { _deallocAll(); }

   virtual Block *avail(int m, int n) { return (Block *)getBlockInfoAll(m, n).ptrBlock; }

   virtual int rank() const { return 0; }

#if DIMENSION == 3
   int getZforward(const int level, const int i, const int j, const int k) const
   {
      const int TwoPower = 1 << level;
      const int ix       = (i + TwoPower * NX) % (NX * TwoPower);
      const int iy       = (j + TwoPower * NY) % (NY * TwoPower);
      const int iz       = (k + TwoPower * NZ) % (NZ * TwoPower);
      return BlockInfo::forward(level, ix, iy, iz);
   }
   int getZchild(int level, int i, int j, int k) { return BlockInfo::child(level, i, j, k); }
   Block &operator()(int ix, int iy, int iz, int m)
   {
      const int n = getZforward(m, ix, iy, iz);
      return *(Block *)getBlockInfoAll(m, n).ptrBlock;
   }
   Block *avail1(int ix, int iy, int iz, int m)
   {
      int n = getZforward(m, ix, iy, iz);
      return avail(m, n);
   }
#else // DIMENSION = 2
   int getZforward(const int level, const int i, const int j) const
   {
      const int TwoPower = 1 << level;
      const int ix       = (i + TwoPower * NX) % (NX * TwoPower);
      const int iy       = (j + TwoPower * NY) % (NY * TwoPower);
      return BlockInfo::forward(level, ix, iy);
   }
   int getZchild(int level, int i, int j) { return BlockInfo::child(level, i, j); }
   Block &operator()(int ix, int iy, int m)
   {
      const int n = getZforward(m, ix, iy);
      return *(Block *)getBlockInfoAll(m, n).ptrBlock;
   }
   Block *avail1(int ix, int iy, int m)
   {
      const int n = getZforward(m, ix, iy);
      return avail(m, n);
   }
#endif

   std::array<int, 3> getMaxBlocks() const { return {NX, NY, NZ}; }

   inline int getlevelMax() { return levelMax; }
   inline int getlevelMax() const { return levelMax; }

   BlockInfo &getBlockInfoAll(int m, int n)
   {
#ifdef CUBISM_USE_MAP
      size_t aux  = level_base[m] + n;
      auto retval = BlockInfoAll.find(aux);
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
               double h0       = (maxextent / std::max(NX * Block::sizeX, std::max(NY * Block::sizeY, NZ * Block::sizeZ)));
               double h        = h0 / TwoPower;
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
               double h0        = (maxextent / std::max(NX * Block::sizeX, std::max(NY * Block::sizeY, NZ * Block::sizeZ)));
               double h         = h0 / TwoPower;
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
   void SortBlocks()
   {
      for (size_t i = 0; i < m_vInfo.size(); i++) m_vInfo[i].blockID = i;
      std::sort(m_vInfo.begin(), m_vInfo.end());
      std::vector<size_t> permutation(m_vInfo.size());
      for (size_t i = 0; i < m_vInfo.size(); i++) permutation[i] = m_vInfo[i].blockID;
      apply_permutation_in_place<Block *>(this->m_blocks, permutation);
      for (size_t i = 0; i < m_vInfo.size(); i++) m_vInfo[i].blockID = i;
      //
      //      for (size_t iii = 0 ; iii< m_vInfo.size(); iii++)
      //      {
      //         int level = m_vInfo[iii].level;
      //         int Z = m_vInfo[iii].Z;
      //         BlockInfo & dummy = getBlockInfoAll(level,Z);
      //         Tree(level,Z).setrank(0);
      //         int p[3];
      //         BlockInfo::inverse(Z,level,p[0],p[1]);
      //         if (level < levelMax - 1)
      //           for (int k = 0; k < 2; k++)
      //            for (int j = 0; j < 2; j++)
      //             for (int i = 0; i < 2; i++)
      //             {
      //                 int nc = getZforward(level + 1, 2 * p[0] + i, 2 * p[1] + j);
      //                 Tree(level + 1,nc).setCheckCoarser();
      //             if (level > 0)
      //             {
      //                int nf = getZforward(level - 1, p[0] / 2, p[1] / 2);
      //                Tree(level - 1,nf).setCheckFiner();
      //             }
      //             }
      //      }
      //
   }

   int getBlocksPerDimension(int idim) const
   {
      std::cout << "You called Grid::getBlocksPerDimension() in an AMR solver. Do you really need that?" << std::endl;
      abort();
   }
   virtual int get_world_size() const { return 1; }

   void UpdateMyGroups()
   {
      // if (!UpdateGroups) return; //TODO : does not work for CUP2D
      if (rank() == 0) std::cout << "Updating groups..." << std::endl;

      const unsigned int nX = BlockType::sizeX;
      const unsigned int nY = BlockType::sizeY;
      const size_t Ngrids   = getBlocksInfo().size();
      const auto &MyInfos   = getBlocksInfo();
      UpdateGroups          = false;
      MyGroups.clear();
      std::vector<bool> added(MyInfos.size(), false);

#if DIMENSION == 3
      const unsigned int nZ = BlockType::sizeZ;
      for (unsigned int m = 0; m < Ngrids; m++)
      {
         const BlockInfo &I = MyInfos[m];

         if (added[I.blockID]) continue;
         BlockGroup newGroup;

         newGroup.level = I.level;
         newGroup.h     = I.h;
         newGroup.Z.push_back(I.Z);

         std::vector<int> base(3);
         base[0] = I.index[0];
         base[1] = I.index[1];
         base[2] = I.index[2];
         std::vector<int> i_off(6, 0);
         std::vector<bool> ready_(6, false);

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

                     if (i0 < 0 || i1 < 0 || i2 < 0 || i0 >= blk[d0] * (1 << I.level) || i1 >= blk[d1] * (1 << I.level) || i2 >= blk[d2] * (1 << I.level))
                     {
                        valid = false;
                        break;
                     }
                     int n;
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
                        int n;
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
         } while (ready_[0] == false || ready_[1] == false || ready_[2] == false || ready_[3] == false || ready_[4] == false || ready_[5] == false);

         const int ix_min = base[0] - i_off[0];
         const int iy_min = base[1] - i_off[1];
         const int iz_min = base[2] - i_off[2];
         const int ix_max = base[0] + i_off[3];
         const int iy_max = base[1] + i_off[4];
         const int iz_max = base[2] + i_off[5];

         int n_base = getZforward(I.level, ix_min, iy_min, iz_min);

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
   }
#else
      for (unsigned int m = 0; m < Ngrids; m++)
      {
         const BlockInfo &I = MyInfos[m];

         if (added[I.blockID]) continue;
         BlockGroup newGroup;

         newGroup.level = I.level;
         newGroup.h     = I.h;
         newGroup.Z.push_back(I.Z);

         std::vector<int> base(3);
         base[0] = I.index[0];
         base[1] = I.index[1];
         base[2] = 0; // I.index[2];
         std::vector<int> i_off(4, 0);
         std::vector<bool> ready_(4, false);

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
                  int n = (d == 0 || d == 2) ? getZforward(I.level, i0, i1) : getZforward(I.level, i1, i0);

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
                     int n = (d == 0 || d == 2) ? getZforward(I.level, i0, i1) : getZforward(I.level, i1, i0);
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

         int n_base = getZforward(I.level, ix_min, iy_min);

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
   }

#endif
};

} // namespace cubism
