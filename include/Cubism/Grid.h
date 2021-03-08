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

struct TreePosition
{
    int position{-3};
    bool CheckCoarser() const {return position == -2;}
    bool CheckFiner  () const {return position == -1;}
    bool Exists      () const {return position >=  0;}
    int  rank        () const {return position;}
    void setrank(const int r) {position = r;}
    void setCheckCoarser()    {position = -2;}
    void setCheckFiner  ()    {position = -1;}
};

template <typename Block, template <typename X> class allocator = std::allocator>
class Grid
{
 protected:
   #ifdef CUBISM_USE_MAP
    std::unordered_map<size_t, BlockInfo*> BlockInfoAll;
    std::unordered_map<size_t,TreePosition> Octree;
   #else
    std::vector<std::vector<BlockInfo*>> BlockInfoAll;
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

   const bool xperiodic; // grid periodicity in x-direction
   const bool yperiodic; // grid periodicity in y-direction
   const bool zperiodic; // grid periodicity in z-direction

   TreePosition & Tree(int m, int n)
   {
      #ifdef CUBISM_USE_MAP
        size_t aux = level_base[m]+n;
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
          return Tree(m,n);
        }
        else
        {
          return retval->second;
        }
      #else
        return Octree[m][n];
      #endif
   }
   TreePosition & Tree(BlockInfo & info) {return Tree(info.level,info.Z);}
   TreePosition & Tree(const BlockInfo & info) {return Tree(info.level,info.Z);}
   
   bool UpdateFluxCorrection{true};

   void _alloc() // called in class constructor
   {
      int m        = levelStart;
      int TwoPower = 1 << m;      
      for (int n = 0; n < NX * NY * NZ * pow(TwoPower, DIMENSION); n++)
      {
         Tree(m,n).setrank(0);
         _alloc(m, n);
      }
      if (m - 1 >= 0)
      {
        for (int n = 0; n < NX * NY * NZ * pow( (1<<(m-1)), DIMENSION); n++)
          Tree(m-1,n).setCheckFiner();
      }
      if (m + 1 < levelMax)
      {
        for (int n = 0; n < NX * NY * NZ * pow( (1<<(m+1)), DIMENSION); n++)
          Tree(m+1,n).setCheckCoarser();
      }
      FillPos();
   }

   virtual void _alloc(int m, int n) // called whenever the grid is refined
   {
      allocator<Block> alloc;
      getBlockInfoAll(m,n).ptrBlock = alloc.allocate(1);
      getBlockInfoAll(m,n).changed  = true;
      getBlockInfoAll(m,n).h_gridpoint = getBlockInfoAll(m,n).h;
      m_blocks.push_back((Block *)getBlockInfoAll(m,n).ptrBlock);
      m_vInfo.push_back(getBlockInfoAll(m,n));
   }

   void _deallocAll() // called in class destructor
   {
      allocator<Block> alloc;
      for (size_t i = 0 ; i < m_vInfo.size() ; i ++)
      {
         int m = m_vInfo[i].level;
         int n = m_vInfo[i].Z;
         alloc.deallocate((Block *)getBlockInfoAll(m,n).ptrBlock, 1);
      }
      for (int m = 0 ; m < levelMax; m++)
      {
        const size_t nmax = getMaxBlocks()[0]*getMaxBlocks()[1]*getMaxBlocks()[2]*pow(1<<m,3);
        for (size_t n = 0 ; n < nmax ; n++)
        {
          if (Tree(m,n).position != -3)
          {
            #ifndef CUBISM_USE_MAP
              delete BlockInfoAll[m][n];
            #else
              size_t aux = level_base[m]+n;
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
      alloc.deallocate((Block *)getBlockInfoAll(m,n).ptrBlock, 1);
      Tree(m,n).setCheckCoarser();
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
            getBlockInfoAll(m_new,n_new).state   = Leave;
            getBlockInfoAll(m_new,n_new).changed = true;
            m_vInfo[j]                         = getBlockInfoAll(m_new,n_new);
            m_blocks[j]                        = (Block *)getBlockInfoAll(m_new,n_new).ptrBlock;
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
            m_vInfo[j] = getBlockInfoAll(m,n);

            assert(getBlockInfoAll(m,n).state == m_vInfo[j].state);
            //assert(getBlockInfoAll(m,n).TreePos == Exists);
            assert(Tree(m,n).Exists());

            m_blocks[j] = (Block *)getBlockInfoAll(m,n).ptrBlock;
         }
      else
         for (size_t j = 0; j < m_vInfo.size(); j++)
         {
            int m            = m_vInfo[j].level;
            int n            = m_vInfo[j].Z;
            m_vInfo[j].state = getBlockInfoAll(m,n).state;
            //assert(getBlockInfoAll(m,n).TreePos == Exists);
            assert(Tree(m,n).Exists());
            m_blocks[j] = (Block *)getBlockInfoAll(m,n).ptrBlock;
         }
      for (size_t j = 0; j < m_vInfo.size(); j++)
      {
         int m            = m_vInfo[j].level;
         int n            = m_vInfo[j].Z;
         m_vInfo[j].blockID = j;
         getBlockInfoAll(m,n).blockID = j;
      }
   }

   Grid(const unsigned int _NX, const unsigned int _NY = 1, const unsigned int _NZ = 1,
        const double _maxextent = 1, const unsigned int _levelStart = 0,
        const unsigned int _levelMax = 1, const bool AllocateBlocks = true,
        const bool a_xperiodic = true, const bool a_yperiodic = true, const bool a_zperiodic = true)
       : NX(_NX), NY(_NY), NZ(_NZ), maxextent(_maxextent), levelMax(_levelMax),
         levelStart(_levelStart), xperiodic(a_xperiodic), yperiodic(a_yperiodic),
         zperiodic(a_zperiodic)
   {
      BlockInfo dummy;
     #if DIMENSION == 3
      int nx     = dummy.blocks_per_dim(0, NX, NY, NZ);
      int ny     = dummy.blocks_per_dim(1, NX, NY, NZ);
      int nz     = dummy.blocks_per_dim(2, NX, NY, NZ);
     #else
      int nx     = dummy.blocks_per_dim(0, NX, NY);
      int ny     = dummy.blocks_per_dim(1, NX, NY);
      int nz     = 1;
     #endif      
      int lvlMax = dummy.levelMax(levelMax);

      #ifdef CUBISM_USE_MAP
        for (int m = 0; m < lvlMax; m++)
        {
          int TwoPower            = 1 << m;
          const unsigned int Ntot = nx * ny * nz * pow(TwoPower, DIMENSION);
          if (m==0)level_base.push_back(Ntot);
          if (m>0) level_base.push_back(level_base[m-1] + Ntot);
        }
      #else
        BlockInfoAll.resize(lvlMax);
        Octree.resize(lvlMax);
        for (int m = 0; m < lvlMax; m++)
        {
          int TwoPower            = 1 << m;
          const unsigned int Ntot = nx * ny * nz * pow(TwoPower, DIMENSION);
          if (m==0)level_base.push_back(Ntot);
          if (m>0) level_base.push_back(level_base[m-1] + Ntot);
          BlockInfoAll[m].resize(Ntot,nullptr);
          Octree[m].resize(Ntot);
        }
      #endif
      if (AllocateBlocks) _alloc();
   }

   virtual ~Grid() { _deallocAll(); }

   virtual Block *avail(int m, int n){ return (Block *)getBlockInfoAll(m,n).ptrBlock; }

#if DIMENSION == 3
   virtual Block *avail1(int ix, int iy, int iz, int m)
   {
      int n = getZforward(m, ix, iy, iz);
      return avail(m, n);
   }
#endif
#if DIMENSION == 2
   virtual Block *avail1(int ix, int iy, int m)
   {
      int n = getZforward(m, ix, iy);
      return avail(m, n);
   }
#endif

   virtual int rank() const { return 0; }

#if DIMENSION == 3
   int getZforward(const int level, const int i, const int j, const int k) const
   {
      int TwoPower = 1 << level;
      int ix       = (i + TwoPower * NX) % (NX * TwoPower);
      int iy       = (j + TwoPower * NY) % (NY * TwoPower);
      int iz       = (k + TwoPower * NZ) % (NZ * TwoPower);
      return BlockInfo::forward(level, ix, iy, iz);
   }
   int getZchild(int level, int i, int j, int k) { return BlockInfo::child(level, i, j, k); }
   virtual Block &operator()(int ix, int iy, int iz, int m)
   {
      int n = getZforward(m, ix, iy, iz);
      return *(Block *)getBlockInfoAll(m,n).ptrBlock;
   }
#endif
#if DIMENSION == 2
   int getZforward(const int level, const int i, const int j) const
   {
      int TwoPower = 1 << level;
      int ix       = (i + TwoPower * NX) % (NX * TwoPower);
      int iy       = (j + TwoPower * NY) % (NY * TwoPower);
      return BlockInfo::forward(level, ix, iy);
   }
   int getZchild(int level, int i, int j) { return BlockInfo::child(level, i, j); }
   virtual Block &operator()(int ix, int iy, int m)
   {
      int n = getZforward(m, ix, iy);
      return *(Block *)getBlockInfoAll(m,n).ptrBlock;
   }
#endif

   virtual std::array<int, 3> getMaxBlocks() const { return {NX, NY, NZ}; }

   inline int getlevelMax() { return levelMax; }
   inline int getlevelMax() const { return levelMax; }

   
   BlockInfo &getBlockInfoAll(int m, int n)
   {
      #ifdef CUBISM_USE_MAP
        size_t aux = level_base[m]+n;
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
                  BlockInfo * dumm = new BlockInfo();
                  int TwoPower = 1 << m;
                  double h0 = (maxextent / std::max(NX * Block::sizeX, std::max(NY * Block::sizeY, NZ * Block::sizeZ)));
                  double h = h0 / TwoPower;
                  double origin[3];
                  int i,j,k;
                #if DIMENSION == 3
                  BlockInfo::inverse(n,m,i,j,k);
                #else
                  BlockInfo::inverse(n,m,i,j);
                  k = 0;
                #endif
                  origin[0]  = i * Block::sizeX * h;
                  origin[1]  = j * Block::sizeY * h;
                  origin[2]  = k * Block::sizeZ * h;
                  dumm->setup(m, h, origin, n);
                  BlockInfoAll[aux] = dumm;
              }
          }
          return getBlockInfoAll(m,n);
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
                    BlockInfoAll[m][n] = new BlockInfo();                            
                    int TwoPower = 1 << m;
                    double h0 = (maxextent / std::max(NX * Block::sizeX, std::max(NY * Block::sizeY, NZ * Block::sizeZ)));
                    double h = h0 / TwoPower;
                    double origin[3];
                    int i,j,k;
                  #if DIMENSION == 3
                    BlockInfo::inverse(n,m,i,j,k);           
                  #else
                    BlockInfo::inverse(n,m,i,j);
                    k = 0;           
                  #endif
                    origin[0]  = i * Block::sizeX * h;
                    origin[1]  = j * Block::sizeY * h;
                    origin[2]  = k * Block::sizeZ * h;
                    BlockInfoAll[m][n]->setup(m, h, origin, n);
                }
            }
            return *BlockInfoAll[m][n];
        }
      #endif
   }

   inline        std::vector<Block *> &GetBlocks()       { return m_blocks; }
   inline const  std::vector<Block *> &GetBlocks() const { return m_blocks; }

         std::vector<BlockInfo> &getBlocksInfo()       { return m_vInfo; }
   const std::vector<BlockInfo> &getBlocksInfo() const { return m_vInfo; }

   template <typename T>
   void apply_permutation_in_place(std::vector<T>& vec, std::vector<std::size_t>& p)
   {
       std::vector<bool> done(vec.size(),false);
       for (std::size_t i = 0; i < vec.size(); ++i)
       {
           if (done[i])
           {
               continue;
           }
           done[i] = true;
           std::size_t prev_j = i;
           std::size_t j = p[i];
           while (i != j)
           {
               std::swap(vec[prev_j], vec[j]);
               done[j] = true;
               prev_j = j;
               j = p[j];
           }
       }
   }
   void SortBlocks()
   {
      for (size_t i = 0 ; i < m_vInfo.size(); i++)
         m_vInfo[i].blockID = i;
      std::sort(m_vInfo.begin(), m_vInfo.end());
      std::vector<size_t> permutation(m_vInfo.size());
      for (size_t i = 0 ; i< m_vInfo.size(); i++)
         permutation[i] = m_vInfo[i].blockID;
      apply_permutation_in_place<Block *>(this->m_blocks,permutation);
      for (size_t i = 0 ; i< m_vInfo.size(); i++)
         m_vInfo[i].blockID = i;
   }

   virtual int getBlocksPerDimension(int idim) const
   {
      std::cout <<"You called Grid::getBlocksPerDimension() in an AMR solver. Do you really need that?"<<std::endl;
      abort();
   }
   virtual int get_world_size() const {return 1;}
};

} // namespace cubism
