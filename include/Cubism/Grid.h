#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>
#include <omp.h>

#ifdef CUBISM_USE_NUMA
   #include <numa.h>
#endif

#include "BlockInfo.h"

namespace cubism // AMR_CUBISM
{

template <typename Block, template <typename X> class allocator = std::allocator>
class Grid
{
 protected:
   std::vector<std::vector<BlockInfo*>> BlockInfoAll;
   std::vector<std::vector<bool>> ready;
   std::vector<BlockInfo> m_vInfo; // meta-data for blocks that belong to this rank
   std::vector<Block *> m_blocks;  // pointers to blocks that belong to this rank

   #if DIMENSION == 3
   std::vector<std::vector<std::vector<std::vector<int>>>> Zsave;
   #else
   std::vector<std::vector<std::vector<int>>> Zsave;
   #endif

   const int NX;           // Total # of blocks for level 0 in X-direction
   const int NY;           // Total # of blocks for level 0 in Y-direction
   const int NZ;           // Total # of blocks for level 0 in Z-direction
   const double maxextent; // Maximum domain extent
   const int levelMax;     // Maximum refinement level allowed
   const int levelStart;   // Initial refinement level

 public:
   int N; // Current number of blocks
   typedef Block BlockType;
   typedef typename Block::RealType Real; // Block MUST provide `RealType`.

   const bool xperiodic;
   const bool yperiodic;
   const bool zperiodic;

   bool UpdateFluxCorrection{true};

   void _alloc() // called in class constructor
   {
      int m        = levelStart;
      int TwoPower = 1 << m;      
     #if DIMENSION == 3
      for (int n = 0; n < NX * NY * NZ * pow(TwoPower, 3); n++)
     #else
      for (int n = 0; n < NX * NY * pow(TwoPower, 2); n++)
     #endif  
      {
         getBlockInfoAll(m,n).TreePos = Exists;
         _alloc(m, n);
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
      m_vInfo.push_back(*BlockInfoAll[m][n]);
      N++;
   }

   virtual void _deallocAll() // called in class destructor
   {
      m_vInfo.clear();
      allocator<Block> alloc;
      for (size_t j = 0; j < m_vInfo.size(); j++) alloc.deallocate(m_blocks[j], 1);
      BlockInfoAll.clear();
   }

   void _dealloc(int m, int n) // called whenever the grid is compressed
   {
      N--;
      allocator<Block> alloc;
      alloc.deallocate((Block *)getBlockInfoAll(m,n).ptrBlock, 1);
      getBlockInfoAll(m,n).myrank = -1;
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
            assert(getBlockInfoAll(m,n).TreePos == Exists);

            m_blocks[j] = (Block *)getBlockInfoAll(m,n).ptrBlock;
         }
      else
         for (size_t j = 0; j < m_vInfo.size(); j++)
         {
            int m            = m_vInfo[j].level;
            int n            = m_vInfo[j].Z;
            m_vInfo[j].state = getBlockInfoAll(m,n).state;
            assert(getBlockInfoAll(m,n).TreePos == Exists);
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
     #endif      
      int lvlMax = dummy.levelMax(levelMax);

      N                = 0;

      BlockInfoAll.resize(lvlMax);
      ready.resize(lvlMax);
      Zsave.resize(lvlMax);

#if DIMENSION == 3
      for (int m = 0; m < lvlMax; m++)
      {
        int TwoPower            = 1 << m;
        const unsigned int Ntot = nx * ny * nz * pow(TwoPower, 3);
        BlockInfoAll[m].resize(Ntot);
        ready[m].resize(Ntot,false);
        Zsave[m].resize(nx * TwoPower);
        for (int ix = 0; ix < nx * TwoPower; ix++)
        {
            Zsave[m][ix].resize(ny * TwoPower);
            for (int iy = 0; iy < ny * TwoPower; iy++)
            {
               Zsave[m][ix][iy].resize(nz * TwoPower);
            }
        }
        for (int i = 0; i < nx * TwoPower; i++)
        for (int j = 0; j < ny * TwoPower; j++)
        for (int k = 0; k < nz * TwoPower; k++)
        {
            int n = BlockInfo::forward(m, i, j, k);
            Zsave[m][i][j][k] = n; 
        }
      }
#endif
#if DIMENSION == 2      
      for (int m = 0; m < lvlMax; m++)
      {
        int TwoPower            = 1 << m;
        const unsigned int Ntot = nx * ny * pow(TwoPower, 2);
        BlockInfoAll[m].resize(Ntot);
        ready[m].resize(Ntot,false);
        Zsave[m].resize(NX * TwoPower);
        for (int ix = 0; ix < nx * TwoPower; ix++)
        {
            Zsave[m][ix].resize(ny * TwoPower);
        }

        for (int i = 0; i < NX * TwoPower; i++)
        for (int j = 0; j < NY * TwoPower; j++)
        {
            int n = BlockInfo::forward(m, i, j);
            Zsave[m][i][j] = n;
        }
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
      return Zsave[level][ix][iy][iz];
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
      return Zsave[level][ix][iy];
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

   
   virtual BlockInfo &getBlockInfoAll(int m, int n)
   {
        if (ready[m][n])
        {
            return *BlockInfoAll[m][n];
        }
        else
        {
            #pragma omp critical
            {
                if (ready[m][n]==false)        
                {
                    BlockInfoAll[m][n] = new BlockInfo();                            
                    int blockrank = 0;
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
                    TreePosition TreePos;
                    if      (m == levelStart) TreePos = Exists;
                    else if (m <  levelStart) TreePos = CheckFiner;
                    else                      TreePos = CheckCoarser;
                    BlockInfoAll[m][n]->setup(m, h, origin, n, blockrank, TreePos);
                    ready[m][n] = true;
                }
            }
            return *BlockInfoAll[m][n];
        }
   }

   virtual std::vector<std::vector<BlockInfo*>> &getBlockInfoAll() { return BlockInfoAll; }

   inline        std::vector<Block *> &GetBlocks()       { return m_blocks; }
   inline const  std::vector<Block *> &GetBlocks() const { return m_blocks; }

   virtual       std::vector<BlockInfo> &getBlocksInfo()       { return m_vInfo; }
   virtual const std::vector<BlockInfo> &getBlocksInfo() const { return m_vInfo; }

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