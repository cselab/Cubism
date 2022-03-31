#pragma once

#include "Grid.h"
#include "Matrix3D.h"
#include <cstring>
#include <math.h>
#include <string>

namespace cubism // AMR_CUBISM
{
#ifdef __bgq__
#include <builtins.h>
#define memcpy2(a, b, c) __bcopy((b), (a), (c))
#else
#define memcpy2(a, b, c) memcpy((a), (b), (c))
#endif

//default coarse-fine interpolation stencil
#if DIMENSION == 3
    const int default_start [3] = {-1,-1,-1};
    const int default_end   [3] = {2,2,2}; 
#else
    const int default_start [3] = {-1,-1,0};
    const int default_end   [3] = {2,2,1}; 
#endif
/*
   Working copy of Block + Ghosts.
   Data of original block is copied (!) here. So when changing something in
   the lab we are not changing the original data.
   Works for (inner) blocks on the same rank. The blocks may have a different resolution; however
   two adjacent blocks are assumed to differ by at most one level of resolution (no adjacent blocks
   with grid spacing h and h/4 are allowed). Refinement ratio is (of course) 2.
*/

template <typename TBlock, template <typename X> class allocator = std::allocator,
          typename ElementTypeT = typename TBlock::ElementType>
class BlockLab
{
 public:
   typedef ElementTypeT ElementType;
   typedef typename ElementTypeT::RealType Real; // Element type MUST provide `RealType`.

 protected:
   typedef TBlock BlockType;
   typedef typename BlockType::ElementType ElementTypeBlock;

   Matrix3D<ElementType, true, allocator> *m_cacheBlock; // This is filled by the Blocklab
   int m_stencilStart[3], m_stencilEnd[3];
   bool istensorial;
   bool use_averages;

   Grid<BlockType, allocator> *m_refGrid;
   int NX, NY, NZ;
   std::array<BlockType *, 27> myblocks;
   std::array<int, 27> coarsened_nei_codes;
   int coarsened_nei_codes_size;

   // Extra stuff for AMR:
   Matrix3D<ElementType, true, allocator> *m_CoarsenedBlock; // coarsened version of given block
   int m_InterpStencilStart[3], m_InterpStencilEnd[3];       // stencil used for refinement (assumed tensorial)
   bool coarsened;                                           // will be true if block has at least one coarser neighbor
   int CoarseBlockSize[3];                                   // size of coarsened block (nX/2,nY/2,nZ/2)
   const double d_coef_plus[9] = {-0.09375, 0.4375,0.15625,  //starting point (+2,+1,0)
                                   0.15625,-0.5625,0.90625,  //last point     (-2,-1,0)
                                  -0.09375, 0.4375,0.15625}; //central point  (-1,0,+1)

   const double d_coef_minus[9]= { 0.15625,-0.5625, 0.90625, //starting point (+2,+1,0)
                                  -0.09375, 0.4375, 0.15625, //last point     (-2,-1,0)
                                   0.15625, 0.4375,-0.09375};//central point  (-1,0,+1)

   virtual void _apply_bc(const BlockInfo &info, const Real t = 0, bool coarse = false) {}

   template <typename T>
   void _release(T *&t)
   {
      if (t != NULL)
      {
         allocator<T>().destroy(t);
         allocator<T>().deallocate(t, 1);
      }
      t = NULL;
   }

 public:
   BlockLab(): m_cacheBlock(nullptr), m_refGrid(nullptr),m_CoarsenedBlock(nullptr)
   {
      m_stencilStart[0] = m_stencilStart[1] = m_stencilStart[2] = 0;
      m_stencilEnd[0] = m_stencilEnd[1] = m_stencilEnd[2] = 0;
      m_InterpStencilStart[0] = m_InterpStencilStart[1] = m_InterpStencilStart[2] = 0;
      m_InterpStencilEnd[0] = m_InterpStencilEnd[1] = m_InterpStencilEnd[2] = 0;

      CoarseBlockSize[0] = (int)BlockType::sizeX / 2;
      CoarseBlockSize[1] = (int)BlockType::sizeY / 2;
      CoarseBlockSize[2] = (int)BlockType::sizeZ / 2;
      if (CoarseBlockSize[0] == 0) CoarseBlockSize[0] = 1;
      if (CoarseBlockSize[1] == 0) CoarseBlockSize[1] = 1;
      if (CoarseBlockSize[2] == 0) CoarseBlockSize[2] = 1;
   }

   virtual std::string name() const { return "BlockLab"; }
   virtual bool is_xperiodic() { return true; }
   virtual bool is_yperiodic() { return true; }
   virtual bool is_zperiodic() { return true; }

   ~BlockLab()
   {
      _release(m_cacheBlock);
      _release(m_CoarsenedBlock);
   }

   bool UseCoarseStencil(const BlockInfo &a, const int *b_index)
   {
      if (a.level == 0|| (!use_averages)) return false;

      int imin[3];
      int imax[3];
      for (int d = 0; d < 3; d++)
      {
        imin[d] = (a.index[d] < b_index[d]) ? 0 : -1;
        imax[d] = (a.index[d] > b_index[d]) ? 0 : +1;
      }

      const int aux = 1 << a.level;
      std::array<int, 3> blocksPerDim = m_refGrid->getMaxBlocks();
      if (is_xperiodic())
      {
        if (a.index[0] == 0 && b_index[0] == blocksPerDim[0] * aux - 1) imin[0] = -1;
        if (b_index[0] == 0 && a.index[0] == blocksPerDim[0] * aux - 1) imax[0] = +1;
      }
      if (is_yperiodic())
      {
        if (a.index[1] == 0 && b_index[1] == blocksPerDim[1] * aux - 1) imin[1] = -1;
        if (b_index[1] == 0 && a.index[1] == blocksPerDim[1] * aux - 1) imax[1] = +1;
      }
      if (is_zperiodic())
      {
        if (a.index[2] == 0 && b_index[2] == blocksPerDim[2] * aux - 1) imin[2] = -1;
        if (b_index[2] == 0 && a.index[2] == blocksPerDim[2] * aux - 1) imax[2] = +1;
      }

      for (int i2 = imin[2]; i2 <= imax[2]; i2++)
      for (int i1 = imin[1]; i1 <= imax[1]; i1++)
      for (int i0 = imin[0]; i0 <= imax[0]; i0++)
      {
         const long long n = a.Znei_(i0, i1, i2);
         if ((m_refGrid->Tree(a.level, n)).CheckCoarser())
         {
            return true;
            break;
         }
      }
      return false;
   }

   void prepare(Grid<BlockType, allocator> &grid, int startX, int endX, int startY, int endY,
                int startZ, int endZ, const bool _istensorial, int IstartX = default_start[0], int IendX = default_end[0],
                int IstartY = default_start[1], int IendY = default_end[1], int IstartZ = default_start[2], int IendZ = default_end[2])
   {
      const int ss[3]  = {startX, startY, startZ};
      const int se[3]  = {endX, endY, endZ};
      const int Iss[3] = {IstartX, IstartY, IstartZ};
      const int Ise[3] = {IendX, IendY, IendZ};
      prepare(grid, ss, se, _istensorial, Iss, Ise);
   }

   /**
    * Prepare the extended block.
    * @param collection    Collection of blocks in the grid (e.g. result of
    * Grid::getBlockCollection()).
    * @param boundaryInfo  Info on the boundaries of the grid (e.g. result of
    * Grid::getBoundaryInfo()).
    * @param stencil_start Maximal stencil used for computations at lower boundary.
    *                      Defines how many ghosts we will get in extended block.
    * @param stencil_end   Maximal stencil used for computations at lower boundary.
    *                      Defines how many ghosts we will get in extended block.
    */
   void prepare(Grid<BlockType, allocator> &grid, const int stencil_start[3],
                const int stencil_end[3], const bool _istensorial,
                const int Istencil_start[3]=default_start,
                const int Istencil_end[3]=default_end)
   {
      istensorial = _istensorial;

      m_stencilStart[0] = stencil_start[0];
      m_stencilStart[1] = stencil_start[1];
      m_stencilStart[2] = stencil_start[2];
      m_stencilEnd[0]   = stencil_end[0];
      m_stencilEnd[1]   = stencil_end[1];
      m_stencilEnd[2]   = stencil_end[2];

      m_refGrid = &grid;

      assert(stencil_start[0] >= -BlockType::sizeX);
      assert(stencil_start[1] >= -BlockType::sizeY);
      assert(stencil_start[2] >= -BlockType::sizeZ);
      assert(stencil_end[0] < BlockType::sizeX * 2);
      assert(stencil_end[1] < BlockType::sizeY * 2);
      assert(stencil_end[2] < BlockType::sizeZ * 2);
      assert(stencil_start[0] <= stencil_end[0]);
      assert(stencil_start[1] <= stencil_end[1]);
      assert(stencil_start[2] <= stencil_end[2]);

      if (m_cacheBlock == NULL ||
          (int)m_cacheBlock->getSize()[0] != (int)BlockType::sizeX + m_stencilEnd[0] - m_stencilStart[0] - 1 ||
          (int)m_cacheBlock->getSize()[1] != (int)BlockType::sizeY + m_stencilEnd[1] - m_stencilStart[1] - 1 ||
          (int)m_cacheBlock->getSize()[2] != (int)BlockType::sizeZ + m_stencilEnd[2] - m_stencilStart[2] - 1)
      {
         if (m_cacheBlock != NULL) _release(m_cacheBlock);

         m_cacheBlock = allocator<Matrix3D<ElementType, true, allocator>>().allocate(1);

         allocator<Matrix3D<ElementType, true, allocator>>().construct(m_cacheBlock);

         m_cacheBlock->_Setup(BlockType::sizeX + m_stencilEnd[0] - m_stencilStart[0] - 1,
                              BlockType::sizeY + m_stencilEnd[1] - m_stencilStart[1] - 1,
                              BlockType::sizeZ + m_stencilEnd[2] - m_stencilStart[2] - 1);
      }

      coarsened               = false;
      m_InterpStencilStart[0] = Istencil_start[0];
      m_InterpStencilStart[1] = Istencil_start[1];
      m_InterpStencilStart[2] = Istencil_start[2];

      m_InterpStencilEnd[0] = Istencil_end[0];
      m_InterpStencilEnd[1] = Istencil_end[1];
      m_InterpStencilEnd[2] = Istencil_end[2];

      assert(m_InterpStencilStart[0] <= m_InterpStencilEnd[0]);
      assert(m_InterpStencilStart[1] <= m_InterpStencilEnd[1]);
      assert(m_InterpStencilStart[2] <= m_InterpStencilEnd[2]);

      const int e[3] = {(m_stencilEnd[0]) / 2 + 1 + m_InterpStencilEnd[0] - 1,
                        (m_stencilEnd[1]) / 2 + 1 + m_InterpStencilEnd[1] - 1,
                        (m_stencilEnd[2]) / 2 + 1 + m_InterpStencilEnd[2] - 1};

      const int s[3] = {(m_stencilStart[0] - 1) / 2 + m_InterpStencilStart[0],
                        (m_stencilStart[1] - 1) / 2 + m_InterpStencilStart[1],
                        (m_stencilStart[2] - 1) / 2 + m_InterpStencilStart[2]};

      if (m_CoarsenedBlock == NULL ||
          (int)m_CoarsenedBlock->getSize()[0] != CoarseBlockSize[0] + e[0] - s[0] - 1 ||
          (int)m_CoarsenedBlock->getSize()[1] != CoarseBlockSize[1] + e[1] - s[1] - 1 ||
          (int)m_CoarsenedBlock->getSize()[2] != CoarseBlockSize[2] + e[2] - s[2] - 1)
      {
         if (m_CoarsenedBlock != NULL) _release(m_CoarsenedBlock);

         m_CoarsenedBlock = allocator<Matrix3D<ElementType, true, allocator>>().allocate(1);

         allocator<Matrix3D<ElementType, true, allocator>>().construct(m_CoarsenedBlock);

         m_CoarsenedBlock->_Setup(CoarseBlockSize[0] + e[0] - s[0] - 1,
                                  CoarseBlockSize[1] + e[1] - s[1] - 1,
                                  CoarseBlockSize[2] + e[2] - s[2] - 1);
      }

      #if DIMENSION == 3
         use_averages = (m_refGrid->FiniteDifferences == false || istensorial
                        || m_stencilStart[0]< -2 || m_stencilStart[1] < -2 || m_stencilStart[2] < -2 
                        || m_stencilEnd  [0]>  3 || m_stencilEnd  [1] >  3 || m_stencilEnd  [2] >  3);
      #else
         use_averages = (m_refGrid->FiniteDifferences == false || istensorial
                        || m_stencilStart[0]< -2 || m_stencilStart[1] < -2 
                        || m_stencilEnd  [0]>  3 || m_stencilEnd  [1] >  3);
      #endif
   }

   void load(const BlockInfo & info, const Real t = 0, const bool applybc = true)
   {
      const int nX                    = BlockType::sizeX;
      const int nY                    = BlockType::sizeY;
      const int nZ                    = BlockType::sizeZ;
      const bool xperiodic            = is_xperiodic();
      const bool yperiodic            = is_yperiodic();
      const bool zperiodic            = is_zperiodic();

      std::array<int, 3> blocksPerDim = m_refGrid->getMaxBlocks();

      const int aux = 1 << info.level;
      NX      = blocksPerDim[0] * aux; // needed for apply_bc
      NY      = blocksPerDim[1] * aux; // needed for apply_bc
      NZ      = blocksPerDim[2] * aux; // needed for apply_bc

      assert(m_cacheBlock != NULL);

      // 1.load the block into the cache
      {
         BlockType &block            = *(BlockType *)info.ptrBlock;
         ElementType *ptrSource = &block(0);

         #if 0 // original
            for(int iz=0; iz<nZ; iz++)
            for(int iy=0; iy<nY; iy++)
            {
              ElementType * ptrDestination = &m_cacheBlock->Access(0-m_stencilStart[0], iy-m_stencilStart[1], iz-m_stencilStart[2]);
              memcpy2((char *)ptrDestination, (char *)ptrSource, sizeof(ElementType)*nX);
              ptrSource+= nX;
            }
         #else
            const int nbytes = sizeof(ElementType) * nX;
            const int _iz0   = -m_stencilStart[2];
            const int _iz1   = _iz0 + nZ;
            const int _iy0   = -m_stencilStart[1];
            const int _iy1   = _iy0 + nY;
            const int m_vSize0 = m_cacheBlock->getSize(0);
            const int m_nElemsPerSlice = m_cacheBlock->getNumberOfElementsPerSlice();
            const int my_ix = -m_stencilStart[0];
            #pragma GCC ivdep
            for (int iz = _iz0; iz < _iz1; iz++)
            {
               const int my_izx = iz * m_nElemsPerSlice + my_ix;
               #pragma GCC ivdep
               for (int iy = _iy0; iy < _iy1; iy += 4)
               {
                  ElementType * __restrict__ ptrDestination0 = &m_cacheBlock->LinAccess(my_izx + (iy    )*m_vSize0);
                  ElementType * __restrict__ ptrDestination1 = &m_cacheBlock->LinAccess(my_izx + (iy + 1)*m_vSize0);
                  ElementType * __restrict__ ptrDestination2 = &m_cacheBlock->LinAccess(my_izx + (iy + 2)*m_vSize0);
                  ElementType * __restrict__ ptrDestination3 = &m_cacheBlock->LinAccess(my_izx + (iy + 3)*m_vSize0);
                  memcpy2(ptrDestination0, (ptrSource         ), nbytes);
                  memcpy2(ptrDestination1, (ptrSource +     nX), nbytes);
                  memcpy2(ptrDestination2, (ptrSource + 2 * nX), nbytes);
                  memcpy2(ptrDestination3, (ptrSource + 3 * nX), nbytes);
                  ptrSource += 4 * nX;
               }
            }
         #endif
      }

      // 2. put the ghosts into the cache
      {
         coarsened = false;

         const bool xskin = info.index[0] == 0 || info.index[0] == NX - 1;
         const bool yskin = info.index[1] == 0 || info.index[1] == NY - 1;
         const bool zskin = info.index[2] == 0 || info.index[2] == NZ - 1;
         const int xskip  = info.index[0] == 0 ? -1 : 1;
         const int yskip  = info.index[1] == 0 ? -1 : 1;
         const int zskip  = info.index[2] == 0 ? -1 : 1;

         int icodes[DIMENSION == 2 ? 8 : 26];  // Could be uint8_t?
         int k = 0;
         coarsened_nei_codes_size = 0;

         for (int icode = (DIMENSION == 2 ? 9 : 0); icode < (DIMENSION == 2 ? 18 : 27); icode++)
         {
            myblocks[icode] = nullptr;
            if (icode == 1 * 1 + 3 * 1 + 9 * 1) continue;
            const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, icode / 9 - 1};

            if (!xperiodic && code[0] == xskip && xskin) continue;
            if (!yperiodic && code[1] == yskip && yskin) continue;
            if (!zperiodic && code[2] == zskip && zskin) continue;

            const auto & TreeNei = m_refGrid->Tree(info.level,info.Znei_(code[0], code[1], code[2]));
            if (TreeNei.Exists())
            {
               icodes[k++] = icode;
               if (!coarsened)
               {
                  const int infoNei_index[3] ={(info.index[0]+code[0]+NX)%NX,
                                               (info.index[1]+code[1]+NY)%NY,
                                               (info.index[2]+code[2]+NZ)%NZ};
                  coarsened = UseCoarseStencil(info, infoNei_index);
               }
            }
            else if (TreeNei.CheckCoarser())
            {
               coarsened_nei_codes[coarsened_nei_codes_size++] = icode;
               CoarseFineExchange(info, code);
            }

            if (!istensorial && abs(code[0]) + abs(code[1]) + abs(code[2]) > 1) continue;

            // s and e correspond to start and end of this lab's cells that are filled by neighbors
            const int s[3] = {code[0] < 1 ? (code[0] < 0 ? m_stencilStart[0] : 0) : nX,
                              code[1] < 1 ? (code[1] < 0 ? m_stencilStart[1] : 0) : nY,
                              code[2] < 1 ? (code[2] < 0 ? m_stencilStart[2] : 0) : nZ};

            const int e[3] = {code[0] < 1 ? (code[0] < 0 ? 0 : nX) : nX + m_stencilEnd[0] - 1,
                              code[1] < 1 ? (code[1] < 0 ? 0 : nY) : nY + m_stencilEnd[1] - 1,
                              code[2] < 1 ? (code[2] < 0 ? 0 : nZ) : nZ + m_stencilEnd[2] - 1};

            if      (TreeNei.Exists()    ) SameLevelExchange   (info, code, s, e);
            else if (TreeNei.CheckFiner()) FineToCoarseExchange(info, code, s, e);
         } // icode = 0,...,26 (3D) or 9,...,17 (2D)
         if (coarsened)
            for (int i = 0; i < k; ++i)
            {
               const int icode = icodes[i];
               const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, icode / 9 - 1};
               FillCoarseVersion(info, code);
            }

         if (m_refGrid->get_world_size() == 1)
         {
            post_load(info, t, applybc);
         }
      }
   }

   void post_load(const BlockInfo &info, const Real t = 0, bool applybc = true)
   {
      #if DIMENSION == 3
         const int nX = BlockType::sizeX;
         const int nY = BlockType::sizeY;
         const int nZ = BlockType::sizeZ;
         if (coarsened)
         {
            const int offset[3] = {(m_stencilStart[0] - 1) / 2 + m_InterpStencilStart[0],
                                   (m_stencilStart[1] - 1) / 2 + m_InterpStencilStart[1],
                                   (m_stencilStart[2] - 1) / 2 + m_InterpStencilStart[2]};
            #pragma GCC ivdep
            for (int k = 0; k < nZ / 2; k++)
            {
               #pragma GCC ivdep
               for (int j = 0; j < nY / 2; j++)
               {
                  #pragma GCC ivdep
                  for (int i = 0; i < nX / 2; i++)
                  {
                     if (i > -m_InterpStencilStart[0] && i < nX / 2 - m_InterpStencilEnd[0] &&
                         j > -m_InterpStencilStart[1] && j < nY / 2 - m_InterpStencilEnd[1] &&
                         k > -m_InterpStencilStart[2] && k < nZ / 2 - m_InterpStencilEnd[2]) continue;
                     const int ix = 2 * i - m_stencilStart[0];
                     const int iy = 2 * j - m_stencilStart[1];
                     const int iz = 2 * k - m_stencilStart[2];
                     ElementType &coarseElement = m_CoarsenedBlock->Access(i - offset[0], j - offset[1], k - offset[2]);
                     coarseElement = AverageDown( m_cacheBlock->Read(ix  ,iy  ,iz  ), 
                                                  m_cacheBlock->Read(ix+1,iy  ,iz  ),
                                                  m_cacheBlock->Read(ix  ,iy+1,iz  ),
                                                  m_cacheBlock->Read(ix+1,iy+1,iz  ),
                                                  m_cacheBlock->Read(ix  ,iy  ,iz+1),
                                                  m_cacheBlock->Read(ix+1,iy  ,iz+1),
                                                  m_cacheBlock->Read(ix  ,iy+1,iz+1),
                                                  m_cacheBlock->Read(ix+1,iy+1,iz+1));
                  }
               }
            }
         }
      #else
         const int nX = BlockType::sizeX;
         const int nY = BlockType::sizeY;
         if (coarsened)
         {
            const int offset[2] = {(m_stencilStart[0] - 1) / 2 + m_InterpStencilStart[0],
                                   (m_stencilStart[1] - 1) / 2 + m_InterpStencilStart[1]};
            #pragma GCC ivdep
            for (int j = 0; j < nY / 2; j++)
            {
               #pragma GCC ivdep
               for (int i = 0; i < nX / 2; i++)
               {
                  if (i > -m_InterpStencilStart[0] && i < nX / 2 - m_InterpStencilEnd[0] &&
                      j > -m_InterpStencilStart[1] && j < nY / 2 - m_InterpStencilEnd[1]) continue;
                  const int ix = 2 * i - m_stencilStart[0];
                  const int iy = 2 * j - m_stencilStart[1];
                  ElementType &coarseElement = m_CoarsenedBlock->Access(i - offset[0], j - offset[1],0);
                  coarseElement = AverageDown( m_cacheBlock->Read(ix  ,iy  ,0),
                                               m_cacheBlock->Read(ix+1,iy  ,0),
                                               m_cacheBlock->Read(ix  ,iy+1,0),
                                               m_cacheBlock->Read(ix+1,iy+1,0));              
               }
            }
         }
      #endif
      if (applybc) _apply_bc(info, t, true); // apply BC to coarse block
      CoarseFineInterpolation(info);
      if (applybc) _apply_bc(info, t); 
   }

   void SameLevelExchange(const BlockInfo &info, const int *const code, const int *const s, const int *const e)
   {
      const int bytes = (e[0] - s[0]) * sizeof(ElementType);
      if (!bytes) return;

      const int icode = (code[0]+1)+3*(code[1]+1)+9*(code[2]+1);
      myblocks[icode] = m_refGrid->avail(info.level, info.Znei_(code[0], code[1], code[2]));
      if (myblocks[icode] == nullptr) return;
      const BlockType &b = *myblocks[icode];

      const int nX = BlockType::sizeX;
      const int nY = BlockType::sizeY;
      const int nZ = BlockType::sizeZ;
      const int m_vSize0         = m_cacheBlock->getSize(0);
      const int m_nElemsPerSlice = m_cacheBlock->getNumberOfElementsPerSlice();
      const int my_ix            = s[0] - m_stencilStart[0];
      const int mod = (e[1] - s[1]) % 4;

      #pragma GCC ivdep
      for (int iz = s[2]; iz < e[2]; iz++)
      {
         const int my_izx = (iz - m_stencilStart[2]) * m_nElemsPerSlice + my_ix;
         #pragma GCC ivdep
         for (int iy = s[1]; iy < e[1]-mod; iy += 4)
         {
            ElementType * __restrict__ ptrDest0 = &m_cacheBlock->LinAccess(my_izx + (iy     - m_stencilStart[1]) * m_vSize0);
            ElementType * __restrict__ ptrDest1 = &m_cacheBlock->LinAccess(my_izx + (iy + 1 - m_stencilStart[1]) * m_vSize0);
            ElementType * __restrict__ ptrDest2 = &m_cacheBlock->LinAccess(my_izx + (iy + 2 - m_stencilStart[1]) * m_vSize0);
            ElementType * __restrict__ ptrDest3 = &m_cacheBlock->LinAccess(my_izx + (iy + 3 - m_stencilStart[1]) * m_vSize0);
            const ElementType * ptrSrc0 = &b(s[0] - code[0] * nX, iy     - code[1] * nY, iz - code[2] * nZ);
            const ElementType * ptrSrc1 = &b(s[0] - code[0] * nX, iy + 1 - code[1] * nY, iz - code[2] * nZ);
            const ElementType * ptrSrc2 = &b(s[0] - code[0] * nX, iy + 2 - code[1] * nY, iz - code[2] * nZ);
            const ElementType * ptrSrc3 = &b(s[0] - code[0] * nX, iy + 3 - code[1] * nY, iz - code[2] * nZ);
            memcpy2(ptrDest0, ptrSrc0, bytes);
            memcpy2(ptrDest1, ptrSrc1, bytes);
            memcpy2(ptrDest2, ptrSrc2, bytes);
            memcpy2(ptrDest3, ptrSrc3, bytes);
         }
         #pragma GCC ivdep
         for (int iy = e[1]-mod; iy < e[1]; iy++)
         {
            ElementType * __restrict__ ptrDest = &m_cacheBlock->LinAccess(my_izx + (iy - m_stencilStart[1]) * m_vSize0);
            const ElementType * ptrSrc = &b(s[0] - code[0] * nX, iy - code[1] * nY, iz - code[2] * nZ);
            memcpy2(ptrDest, ptrSrc, bytes);
         }
      }
   }

 #if DIMENSION == 3
   ElementType AverageDown(const ElementType &e0, const ElementType &e1, 
                           const ElementType &e2, const ElementType &e3,
                           const ElementType &e4, const ElementType &e5,
                           const ElementType &e6, const ElementType &e7)
   {
      return 0.125 * (e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7);
   }
   virtual void TestInterp(ElementType *C[3][3][3], ElementType *R, int x, int y, int z)
   {
      const ElementType dudx   = 0.125*( (*C[2][1][1]) - (*C[0][1][1]) );
      const ElementType dudy   = 0.125*( (*C[1][2][1]) - (*C[1][0][1]) );
      const ElementType dudz   = 0.125*( (*C[1][1][2]) - (*C[1][1][0]) );
      const ElementType dudxdy = 0.015625*((*C[0][0][1]) + (*C[2][2][1]) - (*C[2][0][1]) - (*C[0][2][1]));
      const ElementType dudxdz = 0.015625*((*C[0][1][0]) + (*C[2][1][2]) - (*C[2][1][0]) - (*C[0][1][2]));
      const ElementType dudydz = 0.015625*((*C[1][0][0]) + (*C[1][2][2]) - (*C[1][2][0]) - (*C[1][0][2]));
      const ElementType lap    = *C[1][1][1] + 0.03125*((*C[0][1][1]) + (*C[2][1][1]) + (*C[1][0][1]) + (*C[1][2][1]) + (*C[1][1][0]) + (*C[1][1][2]) + (-6.0)*(*C[1][1][1]));
      R[0] = lap - dudx - dudy - dudz + dudxdy + dudxdz + dudydz;
      R[1] = lap + dudx - dudy - dudz - dudxdy - dudxdz + dudydz;
      R[2] = lap - dudx + dudy - dudz - dudxdy + dudxdz - dudydz;
      R[3] = lap + dudx + dudy - dudz + dudxdy - dudxdz - dudydz;
      R[4] = lap - dudx - dudy + dudz + dudxdy - dudxdz - dudydz;
      R[5] = lap + dudx - dudy + dudz - dudxdy + dudxdz - dudydz;
      R[6] = lap - dudx + dudy + dudz - dudxdy - dudxdz + dudydz;
      R[7] = lap + dudx + dudy + dudz + dudxdy + dudxdz + dudydz;
   }
 #else
   ElementType AverageDown(const ElementType &e0, const ElementType &e1,
                           const ElementType &e2, const ElementType &e3)
   {
      return 0.25 * ((e0 + e3) + (e1 + e2));
   }  
   void LI(ElementType & a, ElementType b, ElementType c)
   {
      auto kappa = ((4.0/15.0)*a+(6.0/15.0)*c)+(-10.0/15.0)*b;
      auto lambda = (b - c) - kappa;
      a = (4.0*kappa+2.0*lambda)+c;
   }
   void LE(ElementType & a, ElementType b, ElementType c)
   {
      auto kappa = ((4.0/15.0)*a+(6.0/15.0)*c)+(-10.0/15.0)*b;
      auto lambda = (b - c) - kappa;
      a = (9.0*kappa+3.0*lambda)+c;
   }
   virtual void TestInterp(ElementType *C[3][3], ElementType &R, int x, int y)
   {
      const double dx = 0.25*(2*x-1);
      const double dy = 0.25*(2*y-1);
      ElementType dudx   = 0.5*( (*C[2][1]) - (*C[0][1]) );
      ElementType dudy   = 0.5*( (*C[1][2]) - (*C[1][0]) );
      ElementType dudxdy = 0.25*(((*C[0][0]) + (*C[2][2])) - ((*C[2][0]) + (*C[0][2])));
      ElementType dudx2  = ((*C[0][1]) + (*C[2][1])) -2.0*(*C[1][1]);
      ElementType dudy2  = ((*C[1][0]) + (*C[1][2])) -2.0*(*C[1][1]);
      R = (*C[1][1] + (dx*dudx + dy*dudy)) + ( ((0.5*dx*dx)*dudx2+(0.5*dy*dy)*dudy2) +(dx*dy)*dudxdy );
   }
 #endif

   void FineToCoarseExchange(const BlockInfo &info, const int *const code, const int *const s, const int *const e)
   {
      const int bytes = (abs(code[0]) * (e[0] - s[0]) + (1 - abs(code[0])) * ((e[0] - s[0]) / 2)) * sizeof(ElementType);
      if (!bytes) return;

      const int nX = BlockType::sizeX;
      const int nY = BlockType::sizeY;
      const int nZ = BlockType::sizeZ;
      const int m_vSize0         = m_cacheBlock->getSize(0);
      const int m_nElemsPerSlice = m_cacheBlock->getNumberOfElementsPerSlice();
      const int yStep            = (code[1] == 0) ? 2 : 1;
      const int zStep            = (code[2] == 0) ? 2 : 1;
      const int mod              = ((e[1] - s[1]) / yStep) % 4;

      int Bstep = 1;                                                    // face
      if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 2)) Bstep = 3; // edge
      else if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 3)) Bstep = 4; // corner

      /*
        A corner has one finer block.
        An edge has two finer blocks, corresponding to B=0 and B=3. The block B=0 is the one closer
        to the origin (0,0,0). A face has four finer blocks. They are numbered as follows, depending
        on whether the face lies on the xy- , yz- or xz- plane

        y                                  z                                  z
        ^                                  ^                                  ^
        |                                  |                                  |
        |                                  |                                  |
        |_________________                 |_________________                 |_________________
        |        |        |                |        |        |                |        |        |
        |    2   |   3    |                |    2   |   3    |                |    2   |   3    |
        |________|________|                |________|________|                |________|________|
        |        |        |                |        |        |                |        |        |
        |    0   |    1   |                |    0   |    1   |                |    0   |    1   |
        |________|________|------------->x |________|________|------------->x |________|________|------------->y

      */
      // loop over blocks that make up face/edge/corner (respectively 4,2 or 1 blocks)
      for (int B = 0; B <= 3; B += Bstep)
      {
         const int aux = (abs(code[0]) == 1) ? (B % 2) : (B / 2);

         #if DIMENSION == 3
            BlockType *b_ptr = m_refGrid->avail1(2 * info.index[0] + max(code[0], 0) + code[0] + (B % 2) * max(0, 1 - abs(code[0])),
                                                 2 * info.index[1] + max(code[1], 0) + code[1] + aux     * max(0, 1 - abs(code[1])),
                                                 2 * info.index[2] + max(code[2], 0) + code[2] + (B / 2) * max(0, 1 - abs(code[2])),
                                                 info.level + 1);
         #else
            BlockType *b_ptr = m_refGrid->avail1(2 * info.index[0] + max(code[0], 0) + code[0] + (B % 2) * max(0, 1 - abs(code[0])),
                                                 2 * info.index[1] + max(code[1], 0) + code[1] + aux     * max(0, 1 - abs(code[1])),
                                                 info.level + 1);
         #endif
         if (b_ptr == nullptr) continue;
         BlockType &b = *b_ptr;

         const int my_ix = abs(code[0]) * (s[0] - m_stencilStart[0]) + (1 - abs(code[0])) * (s[0] - m_stencilStart[0] + (B % 2) * (e[0] - s[0]) / 2);
         const int XX = s[0] - code[0] * nX + min(0, code[0]) * (e[0] - s[0]);

         #pragma GCC ivdep
         for (int iz = s[2]; iz < e[2]; iz += zStep)
         {
            const int ZZ = (abs(code[2]) == 1) ? 2 * (iz - code[2] * nZ) + min(0, code[2]) * nZ : iz;
            const int my_izx = (abs(code[2]) * (iz - m_stencilStart[2]) + (1 - abs(code[2])) * (iz / 2 - m_stencilStart[2] + (B / 2) * (e[2] - s[2]) / 2)) * m_nElemsPerSlice + my_ix;

            #pragma GCC ivdep
            for (int iy = s[1]; iy < e[1]-mod; iy += 4 * yStep)
            {
               ElementType * __restrict__ ptrDest0 = &m_cacheBlock->LinAccess(my_izx + (abs(code[1]) * (iy + 0 * yStep - m_stencilStart[1]) + (1 - abs(code[1])) * ((iy + 0 * yStep) / 2 - m_stencilStart[1] + aux * (e[1] - s[1]) / 2)) * m_vSize0);
               ElementType * __restrict__ ptrDest1 = &m_cacheBlock->LinAccess(my_izx + (abs(code[1]) * (iy + 1 * yStep - m_stencilStart[1]) + (1 - abs(code[1])) * ((iy + 1 * yStep) / 2 - m_stencilStart[1] + aux * (e[1] - s[1]) / 2)) * m_vSize0);
               ElementType * __restrict__ ptrDest2 = &m_cacheBlock->LinAccess(my_izx + (abs(code[1]) * (iy + 2 * yStep - m_stencilStart[1]) + (1 - abs(code[1])) * ((iy + 2 * yStep) / 2 - m_stencilStart[1] + aux * (e[1] - s[1]) / 2)) * m_vSize0);
               ElementType * __restrict__ ptrDest3 = &m_cacheBlock->LinAccess(my_izx + (abs(code[1]) * (iy + 3 * yStep - m_stencilStart[1]) + (1 - abs(code[1])) * ((iy + 3 * yStep) / 2 - m_stencilStart[1] + aux * (e[1] - s[1]) / 2)) * m_vSize0);
               const int YY0 = (abs(code[1]) == 1) ? 2 * (iy + 0 * yStep - code[1] * nY) + min(0, code[1]) * nY : iy + 0 * yStep;
               const int YY1 = (abs(code[1]) == 1) ? 2 * (iy + 1 * yStep - code[1] * nY) + min(0, code[1]) * nY : iy + 1 * yStep;
               const int YY2 = (abs(code[1]) == 1) ? 2 * (iy + 2 * yStep - code[1] * nY) + min(0, code[1]) * nY : iy + 2 * yStep;
               const int YY3 = (abs(code[1]) == 1) ? 2 * (iy + 3 * yStep - code[1] * nY) + min(0, code[1]) * nY : iy + 3 * yStep;
               #if DIMENSION == 3
                  const ElementType * ptrSrc_00 = &b(XX  ,YY0  ,ZZ  );
                  const ElementType * ptrSrc_10 = &b(XX  ,YY0  ,ZZ+1);
                  const ElementType * ptrSrc_20 = &b(XX  ,YY0+1,ZZ  );
                  const ElementType * ptrSrc_30 = &b(XX  ,YY0+1,ZZ+1);
                  const ElementType * ptrSrc_01 = &b(XX  ,YY1  ,ZZ  );
                  const ElementType * ptrSrc_11 = &b(XX  ,YY1  ,ZZ+1);
                  const ElementType * ptrSrc_21 = &b(XX  ,YY1+1,ZZ  );
                  const ElementType * ptrSrc_31 = &b(XX  ,YY1+1,ZZ+1);
                  const ElementType * ptrSrc_02 = &b(XX  ,YY2  ,ZZ  );
                  const ElementType * ptrSrc_12 = &b(XX  ,YY2  ,ZZ+1);
                  const ElementType * ptrSrc_22 = &b(XX  ,YY2+1,ZZ  );
                  const ElementType * ptrSrc_32 = &b(XX  ,YY2+1,ZZ+1);
                  const ElementType * ptrSrc_03 = &b(XX  ,YY3  ,ZZ  );
                  const ElementType * ptrSrc_13 = &b(XX  ,YY3  ,ZZ+1);
                  const ElementType * ptrSrc_23 = &b(XX  ,YY3+1,ZZ  );
                  const ElementType * ptrSrc_33 = &b(XX  ,YY3+1,ZZ+1);
                  #pragma GCC ivdep
                  for (int ee = 0; ee < (abs(code[0]) * (e[0] - s[0]) + (1 - abs(code[0])) * ((e[0] - s[0]) / 2)); ee++)
                  {
                     ptrDest0[ee] = 0.125*(ptrSrc_00[2*ee  ]+ptrSrc_10[2*ee  ]+ptrSrc_20[2*ee  ]+ptrSrc_30[2*ee  ]
                                          +ptrSrc_00[2*ee+1]+ptrSrc_10[2*ee+1]+ptrSrc_20[2*ee+1]+ptrSrc_30[2*ee+1]);
                     ptrDest1[ee] = 0.125*(ptrSrc_01[2*ee  ]+ptrSrc_11[2*ee  ]+ptrSrc_21[2*ee  ]+ptrSrc_31[2*ee  ]
                                          +ptrSrc_01[2*ee+1]+ptrSrc_11[2*ee+1]+ptrSrc_21[2*ee+1]+ptrSrc_31[2*ee+1]);
                     ptrDest2[ee] = 0.125*(ptrSrc_02[2*ee  ]+ptrSrc_12[2*ee  ]+ptrSrc_22[2*ee  ]+ptrSrc_32[2*ee  ]
                                          +ptrSrc_02[2*ee+1]+ptrSrc_12[2*ee+1]+ptrSrc_22[2*ee+1]+ptrSrc_32[2*ee+1]);
                     ptrDest3[ee] = 0.125*(ptrSrc_03[2*ee  ]+ptrSrc_13[2*ee  ]+ptrSrc_23[2*ee  ]+ptrSrc_33[2*ee  ]
                                          +ptrSrc_03[2*ee+1]+ptrSrc_13[2*ee+1]+ptrSrc_23[2*ee+1]+ptrSrc_33[2*ee+1]);
                  }
               #else
                 const ElementType *ptrSrc_00 = &b(XX,YY0  ,ZZ);
                 const ElementType *ptrSrc_10 = &b(XX,YY0+1,ZZ);
                 const ElementType *ptrSrc_01 = &b(XX,YY1  ,ZZ);
                 const ElementType *ptrSrc_11 = &b(XX,YY1+1,ZZ);
                 const ElementType *ptrSrc_02 = &b(XX,YY2  ,ZZ);
                 const ElementType *ptrSrc_12 = &b(XX,YY2+1,ZZ);
                 const ElementType *ptrSrc_03 = &b(XX,YY3  ,ZZ);
                 const ElementType *ptrSrc_13 = &b(XX,YY3+1,ZZ);    
                 #pragma GCC ivdep
                 for (int ee = 0; ee < (abs(code[0]) * (e[0] - s[0]) + (1 - abs(code[0])) * ((e[0] - s[0]) / 2)); ee++)
                 {
                    ptrDest0[ee] = AverageDown(*(ptrSrc_00 + 2 * ee    ), *(ptrSrc_10 + 2 * ee    ),
                                               *(ptrSrc_00 + 2 * ee + 1), *(ptrSrc_10 + 2 * ee + 1));
                    ptrDest1[ee] = AverageDown(*(ptrSrc_01 + 2 * ee    ), *(ptrSrc_11 + 2 * ee    ),
                                               *(ptrSrc_01 + 2 * ee + 1), *(ptrSrc_11 + 2 * ee + 1));
                    ptrDest2[ee] = AverageDown(*(ptrSrc_02 + 2 * ee), *(ptrSrc_12 + 2 * ee),
                                               *(ptrSrc_02 + 2 * ee + 1), *(ptrSrc_12 + 2 * ee + 1));
                    ptrDest3[ee] = AverageDown(*(ptrSrc_03 + 2 * ee), *(ptrSrc_13 + 2 * ee),
                                               *(ptrSrc_03 + 2 * ee + 1), *(ptrSrc_13 + 2 * ee + 1));
                  }
               #endif
            }
            #pragma GCC ivdep
            for (int iy = e[1]-mod; iy < e[1]; iy += yStep)
            {
               ElementType *ptrDest = (ElementType *)&m_cacheBlock->LinAccess(my_izx + (abs(code[1]) * (iy - m_stencilStart[1]) + (1 - abs(code[1])) *(iy / 2 - m_stencilStart[1] + aux * (e[1] - s[1]) / 2)) * m_vSize0);
               const int YY = (abs(code[1]) == 1) ? 2 * (iy - code[1] * nY) + min(0, code[1]) * nY : iy;
               #if DIMENSION == 3
                  const ElementType * ptrSrc_0 = &b(XX, YY, ZZ);
                  const ElementType * ptrSrc_1 = &b(XX, YY, ZZ + 1);
                  const ElementType * ptrSrc_2 = &b(XX, YY + 1, ZZ);
                  const ElementType * ptrSrc_3 = &b(XX, YY + 1, ZZ + 1);
                  const ElementType * ptrSrc_0_1 = &b(XX+1, YY, ZZ);
                  const ElementType * ptrSrc_1_1 = &b(XX+1, YY, ZZ + 1);
                  const ElementType * ptrSrc_2_1 = &b(XX+1, YY + 1, ZZ);
                  const ElementType * ptrSrc_3_1 = &b(XX+1, YY + 1, ZZ + 1);
                  // average down elements of block b to send to coarser neighbor
                  #pragma GCC ivdep
                  for (int ee = 0; ee < (abs(code[0]) * (e[0] - s[0]) + (1 - abs(code[0])) * ((e[0] - s[0]) / 2)); ee++)
                  {
                     ptrDest[ee] = 0.125f*(ptrSrc_0  [2*ee]+ptrSrc_1  [2*ee]+ptrSrc_2  [2*ee]+ptrSrc_3  [2*ee]
                                          +ptrSrc_0_1[2*ee]+ptrSrc_1_1[2*ee]+ptrSrc_2_1[2*ee]+ptrSrc_3_1[2*ee]);
                  }
               #else
                  const ElementType * ptrSrc_0 = &b(XX, YY    , ZZ);
                  const ElementType * ptrSrc_1 = &b(XX, YY + 1, ZZ);
                  // average down elements of block b to send to coarser neighbor
                  #pragma GCC ivdep
                  for (int ee = 0; ee < (abs(code[0]) * (e[0] - s[0]) + (1 - abs(code[0])) * ((e[0] - s[0]) / 2)); ee++)
                  {
                     ptrDest[ee] = AverageDown(*(ptrSrc_0 + 2 * ee    ), *(ptrSrc_1 + 2 * ee    ),
                                               *(ptrSrc_0 + 2 * ee + 1), *(ptrSrc_1 + 2 * ee + 1));
                  }
               #endif
            }
         }
      } // B
   }

   void CoarseFineExchange(const BlockInfo &info, const int *const code)
   {
      // Coarse neighbors send their cells. Those are stored in m_CoarsenedBlock and are later used
      // in function CoarseFineInterpolation to interpolate fine values.

      const int infoNei_index[3] ={(info.index[0]+code[0]+NX)%NX,
                                   (info.index[1]+code[1]+NY)%NY,
                                   (info.index[2]+code[2]+NZ)%NZ};
      #if DIMENSION == 3
         BlockType *b_ptr = m_refGrid->avail1((infoNei_index[0]) / 2,
                                              (infoNei_index[1]) / 2,
                                              (infoNei_index[2]) / 2, info.level - 1);
      #else
         BlockType *b_ptr = m_refGrid->avail1((infoNei_index[0]) / 2,
                                              (infoNei_index[1]) / 2, info.level - 1);
      #endif

      if (b_ptr == nullptr) return;
      const BlockType &b = *b_ptr;

      const int nX = BlockType::sizeX;
      const int nY = BlockType::sizeY;
      const int nZ = BlockType::sizeZ;

      const int offset[3] = {(m_stencilStart[0] - 1) / 2 + m_InterpStencilStart[0],
                             (m_stencilStart[1] - 1) / 2 + m_InterpStencilStart[1],
                             (m_stencilStart[2] - 1) / 2 + m_InterpStencilStart[2]};

      const int s[3] = {code[0] < 1 ? (code[0] < 0 ? offset[0] : 0) : CoarseBlockSize[0],
                        code[1] < 1 ? (code[1] < 0 ? offset[1] : 0) : CoarseBlockSize[1],
                        code[2] < 1 ? (code[2] < 0 ? offset[2] : 0) : CoarseBlockSize[2]};

      const int e[3] = {code[0] < 1 ? (code[0] < 0 ? 0 : CoarseBlockSize[0]) : CoarseBlockSize[0] + (m_stencilEnd[0]) / 2 + m_InterpStencilEnd[0] - 1,
                        code[1] < 1 ? (code[1] < 0 ? 0 : CoarseBlockSize[1]) : CoarseBlockSize[1] + (m_stencilEnd[1]) / 2 + m_InterpStencilEnd[1] - 1,
                        code[2] < 1 ? (code[2] < 0 ? 0 : CoarseBlockSize[2]) : CoarseBlockSize[2] + (m_stencilEnd[2]) / 2 + m_InterpStencilEnd[2] - 1};

      const int bytes = (e[0] - s[0]) * sizeof(ElementType);
      if (!bytes) return;

      const int base[3] = {(info.index[0] + code[0]) % 2, (info.index[1] + code[1]) % 2, (info.index[2] + code[2]) % 2};

      int CoarseEdge[3];
      CoarseEdge[0] = (code[0] == 0) ? 0 : (((info.index[0] % 2 == 0) && (infoNei_index[0] > info.index[0])) ||
                                            ((info.index[0] % 2 == 1) && (infoNei_index[0] < info.index[0]))) ? 1 : 0;
      CoarseEdge[1] = (code[1] == 0) ? 0 : (((info.index[1] % 2 == 0) && (infoNei_index[1] > info.index[1])) ||
                                            ((info.index[1] % 2 == 1) && (infoNei_index[1] < info.index[1]))) ? 1 : 0;
      CoarseEdge[2] = (code[2] == 0) ? 0 : (((info.index[2] % 2 == 0) && (infoNei_index[2] > info.index[2])) ||
                                            ((info.index[2] % 2 == 1) && (infoNei_index[2] < info.index[2]))) ? 1 : 0;

      const int start[3] = {max(code[0], 0) * nX / 2 + (1 - abs(code[0])) * base[0] * nX / 2 - code[0] * nX + CoarseEdge[0] * code[0] * nX / 2,
                            max(code[1], 0) * nY / 2 + (1 - abs(code[1])) * base[1] * nY / 2 - code[1] * nY + CoarseEdge[1] * code[1] * nY / 2,
                            max(code[2], 0) * nZ / 2 + (1 - abs(code[2])) * base[2] * nZ / 2 - code[2] * nZ + CoarseEdge[2] * code[2] * nZ / 2};

      const int m_vSize0         = m_CoarsenedBlock->getSize(0);
      const int m_nElemsPerSlice = m_CoarsenedBlock->getNumberOfElementsPerSlice();
      const int my_ix            = s[0] - offset[0];
      const int mod              = (e[1] - s[1]) % 4;

      #pragma GCC ivdep
      for (int iz = s[2]; iz < e[2]; iz++)
      {
         const int my_izx = (iz - offset[2]) * m_nElemsPerSlice + my_ix;
         #pragma GCC ivdep
         for (int iy = s[1]; iy < e[1]-mod; iy += 4)
         {
            ElementType __restrict__ *ptrDest0 = &m_CoarsenedBlock->LinAccess(my_izx + (iy + 0 - offset[1]) * m_vSize0);
            ElementType __restrict__ *ptrDest1 = &m_CoarsenedBlock->LinAccess(my_izx + (iy + 1 - offset[1]) * m_vSize0);
            ElementType __restrict__ *ptrDest2 = &m_CoarsenedBlock->LinAccess(my_izx + (iy + 2 - offset[1]) * m_vSize0);
            ElementType __restrict__ *ptrDest3 = &m_CoarsenedBlock->LinAccess(my_izx + (iy + 3 - offset[1]) * m_vSize0);
            const ElementType *ptrSrc0 = &b(s[0] + start[0], iy + 0 + start[1], iz + start[2]);
            const ElementType *ptrSrc1 = &b(s[0] + start[0], iy + 1 + start[1], iz + start[2]);
            const ElementType *ptrSrc2 = &b(s[0] + start[0], iy + 2 + start[1], iz + start[2]);
            const ElementType *ptrSrc3 = &b(s[0] + start[0], iy + 3 + start[1], iz + start[2]);
            memcpy2(ptrDest0, ptrSrc0, bytes);
            memcpy2(ptrDest1, ptrSrc1, bytes);
            memcpy2(ptrDest2, ptrSrc2, bytes);
            memcpy2(ptrDest3, ptrSrc3, bytes);
         }
         #pragma GCC ivdep
         for (int iy = e[1]-mod; iy < e[1]; iy++)
         {
            ElementType *ptrDest = &m_CoarsenedBlock->LinAccess(my_izx + (iy - offset[1]) * m_vSize0);
            const ElementType *ptrSrc = &b(s[0] + start[0], iy + start[1], iz + start[2]);
            memcpy2(ptrDest, ptrSrc, bytes);
         }
      }
   }

   void FillCoarseVersion(const BlockInfo &info, const int *const code)
   {
      // If a neighboring block is on the same level it might need to average down some cells and
      // use them to fill the coarsened version of this block. Those cells are needed to refine the
      // coarsened version and obtain ghosts from coarser neighbors (those cells are inside the
      // interpolation stencil for refinement).

      const int icode = (code[0]+1)+3*(code[1]+1)+9*(code[2]+1);
      if (myblocks[icode] == nullptr) return;
      const BlockType &b = *myblocks[icode];

      const int nX = BlockType::sizeX;
      const int nY = BlockType::sizeY;
      const int nZ = BlockType::sizeZ;

      const int eC[3] = {(m_stencilEnd[0]) / 2 + m_InterpStencilEnd[0],
                         (m_stencilEnd[1]) / 2 + m_InterpStencilEnd[1],
                         (m_stencilEnd[2]) / 2 + m_InterpStencilEnd[2]};

      const int sC[3] = {(m_stencilStart[0] - 1) / 2 + m_InterpStencilStart[0],
                         (m_stencilStart[1] - 1) / 2 + m_InterpStencilStart[1],
                         (m_stencilStart[2] - 1) / 2 + m_InterpStencilStart[2]};

      const int s[3] = {code[0] < 1 ? (code[0] < 0 ? sC[0] : 0) : CoarseBlockSize[0],
                        code[1] < 1 ? (code[1] < 0 ? sC[1] : 0) : CoarseBlockSize[1],
                        code[2] < 1 ? (code[2] < 0 ? sC[2] : 0) : CoarseBlockSize[2]};

      const int e[3] = {code[0] < 1 ? (code[0] < 0 ? 0 : CoarseBlockSize[0]) : CoarseBlockSize[0] + eC[0] - 1,
                        code[1] < 1 ? (code[1] < 0 ? 0 : CoarseBlockSize[1]) : CoarseBlockSize[1] + eC[1] - 1,
                        code[2] < 1 ? (code[2] < 0 ? 0 : CoarseBlockSize[2]) : CoarseBlockSize[2] + eC[2] - 1};

      const int bytes = (e[0] - s[0]) * sizeof(ElementType);
      if (!bytes) return;

      const int start[3] = {
          s[0] + max(code[0], 0) * CoarseBlockSize[0] - code[0] * nX + min(0, code[0]) * (e[0] - s[0]),
          s[1] + max(code[1], 0) * CoarseBlockSize[1] - code[1] * nY + min(0, code[1]) * (e[1] - s[1]),
          s[2] + max(code[2], 0) * CoarseBlockSize[2] - code[2] * nZ + min(0, code[2]) * (e[2] - s[2])};

      const int m_vSize0         = m_CoarsenedBlock->getSize(0);
      const int m_nElemsPerSlice = m_CoarsenedBlock->getNumberOfElementsPerSlice();
      const int my_ix            = s[0] - sC[0];
      const int XX               = start[0];

      #pragma GCC ivdep
      for (int iz = s[2]; iz < e[2]; iz++)
      {
         const int ZZ     = 2 * (iz - s[2]) + start[2];
         const int my_izx = (iz - sC[2]) * m_nElemsPerSlice + my_ix;

         #pragma GCC ivdep
         for (int iy = s[1]; iy < e[1]; iy++)
         {
            if (code[1] == 0 && code[2] == 0 && iy > -m_InterpStencilStart[1] &&
                iy < nY / 2 - m_InterpStencilEnd[1] && iz > -m_InterpStencilStart[2] &&
                iz < nZ / 2 - m_InterpStencilEnd[2])
               continue;

            ElementType __restrict__ *ptrDest1 = &m_CoarsenedBlock->LinAccess(my_izx + (iy - sC[1]) * m_vSize0);

            const int YY = 2 * (iy - s[1]) + start[1];
            #if DIMENSION == 3
               const ElementType *ptrSrc_0 = &b(XX, YY, ZZ);
               const ElementType *ptrSrc_1 = &b(XX, YY, ZZ + 1);
               const ElementType *ptrSrc_2 = &b(XX, YY + 1, ZZ);
               const ElementType *ptrSrc_3 = &b(XX, YY + 1, ZZ + 1);
               // average down elements of block b to send to coarser neighbor
               #pragma GCC ivdep
               for (int ee = 0; ee < e[0] - s[0]; ee++)
               {
                  ptrDest1[ee] = AverageDown(*(ptrSrc_0 + 2 * ee), 
                                             *(ptrSrc_1 + 2 * ee),
                                             *(ptrSrc_2 + 2 * ee), 
                                             *(ptrSrc_3 + 2 * ee),
                                             *(ptrSrc_0 + 2 * ee + 1), 
                                             *(ptrSrc_1 + 2 * ee + 1),
                                             *(ptrSrc_2 + 2 * ee + 1), 
                                             *(ptrSrc_3 + 2 * ee + 1));
               }
            #else
               const ElementType *ptrSrc_0 = (const ElementType *)&b(XX, YY, ZZ);
               const ElementType *ptrSrc_1 = (const ElementType *)&b(XX, YY + 1, ZZ);
               // average down elements of block b to send to coarser neighbor
               #pragma GCC ivdep
               for (int ee = 0; ee < e[0] - s[0]; ee++)
               {
                  ptrDest1[ee] = AverageDown(*(ptrSrc_0 + 2 * ee), 
                                             *(ptrSrc_1 + 2 * ee),
                                             *(ptrSrc_0 + 2 * ee + 1), 
                                             *(ptrSrc_1 + 2 * ee + 1));
               }
            #endif
         }
      }
   }

   void CoarseFineInterpolation(const BlockInfo &info)
   {
      const int nX         = BlockType::sizeX;
      const int nY         = BlockType::sizeY;
      const int nZ         = BlockType::sizeZ;
      const bool xperiodic = is_xperiodic();
      const bool yperiodic = is_yperiodic();
      const bool zperiodic = is_zperiodic();
      const std::array<int, 3> blocksPerDim = m_refGrid->getMaxBlocks();
      const int aux    = 1 << info.level;
      const bool xskin = info.index[0] == 0 || info.index[0] == blocksPerDim[0] * aux - 1;
      const bool yskin = info.index[1] == 0 || info.index[1] == blocksPerDim[1] * aux - 1;
      const bool zskin = info.index[2] == 0 || info.index[2] == blocksPerDim[2] * aux - 1;
      const int xskip  = info.index[0] == 0 ? -1 : 1;
      const int yskip  = info.index[1] == 0 ? -1 : 1;
      const int zskip  = info.index[2] == 0 ? -1 : 1;

      const int offset[3] = {(m_stencilStart[0] - 1) / 2 + m_InterpStencilStart[0],
                             (m_stencilStart[1] - 1) / 2 + m_InterpStencilStart[1],
                             (m_stencilStart[2] - 1) / 2 + m_InterpStencilStart[2]};
      
      for (int ii = 0; ii < coarsened_nei_codes_size; ++ii)
      {
         const int icode = coarsened_nei_codes[ii];
         if (icode == 1 * 1 + 3 * 1 + 9 * 1) continue;
         const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, (icode / 9) % 3 - 1};

         #if DIMENSION == 2
         if (code[2] != 0) continue;
         #endif

         if (!xperiodic && code[0] == xskip && xskin) continue;
         if (!yperiodic && code[1] == yskip && yskin) continue;
         if (!zperiodic && code[2] == zskip && zskin) continue;
         if (!istensorial && abs(code[0]) + abs(code[1]) + abs(code[2]) > 1) continue;

         // s and e correspond to start and end of this lab's cells that are filled by neighbors
         const int s[3] = {code[0] < 1 ? (code[0] < 0 ? m_stencilStart[0] : 0) : nX,
                           code[1] < 1 ? (code[1] < 0 ? m_stencilStart[1] : 0) : nY,
                           code[2] < 1 ? (code[2] < 0 ? m_stencilStart[2] : 0) : nZ};
         const int e[3] = {code[0] < 1 ? (code[0] < 0 ? 0 : nX) : nX + m_stencilEnd[0] - 1,
                           code[1] < 1 ? (code[1] < 0 ? 0 : nY) : nY + m_stencilEnd[1] - 1,
                           code[2] < 1 ? (code[2] < 0 ? 0 : nZ) : nZ + m_stencilEnd[2] - 1};

         const int sC[3] = {
             code[0] < 1 ? (code[0] < 0 ? ((m_stencilStart[0] - 1) / 2) : 0) : CoarseBlockSize[0],
             code[1] < 1 ? (code[1] < 0 ? ((m_stencilStart[1] - 1) / 2) : 0) : CoarseBlockSize[1],
             code[2] < 1 ? (code[2] < 0 ? ((m_stencilStart[2] - 1) / 2) : 0) : CoarseBlockSize[2]};
         // /*comment to silence warnings*/const int eC[3] = {
         // /*comment to silence warnings*/code[0]<1? (code[0]<0 ? 0:nX/2 ) :
         // nX/2+(m_stencilEnd[0])/2
         // + 1 + 0*(m_InterpStencilEnd[0] -1),
         // /*comment to silence warnings*/code[1]<1? (code[1]<0 ? 0:nY/2 ) :
         // nY/2+(m_stencilEnd[1])/2
         // + 1 + 0*(m_InterpStencilEnd[1] -1),
         // /*comment to silence warnings*/code[2]<1? (code[2]<0 ? 0:nZ/2 ) :
         // nZ/2+(m_stencilEnd[2])/2
         // + 1 + 0*(m_InterpStencilEnd[2] -1)};

         const int bytes = (e[0] - s[0]) * sizeof(ElementType);
         if (!bytes) continue;

         #if DIMENSION == 3
            ElementType retval[8];
            if (use_averages)
               for (int iz = s[2]; iz < e[2]; iz += 2)
               {
                  const int ZZ = (iz - s[2] - min(0, code[2]) * ((e[2] - s[2]) % 2)) / 2 + sC[2];
                  const int z = abs(iz - s[2] - min(0, code[2]) * ((e[2] - s[2]) % 2)) % 2;
                  const int izp = (abs(iz) % 2 == 1) ?  -1 : 1;
                  const int rzp = (izp == 1) ? 1:0;
                  const int rz  = (izp == 1) ? 0:1;

                  #pragma GCC ivdep   
                  for (int iy = s[1]; iy < e[1]; iy += 2)
                  {
                     const int YY = (iy - s[1] - min(0, code[1]) * ((e[1] - s[1]) % 2)) / 2 + sC[1];
                     const int y = abs(iy - s[1] - min(0, code[1]) * ((e[1] - s[1]) % 2)) % 2;
                     const int iyp = (abs(iy) % 2 == 1) ?  -1 : 1;
                     const int ryp = (iyp == 1) ? 1:0;
                     const int ry  = (iyp == 1) ? 0:1;

                     #pragma GCC ivdep      
                     for (int ix = s[0]; ix < e[0]; ix += 2)
                     {
                        const int XX = (ix - s[0] - min(0, code[0]) * ((e[0] - s[0]) % 2)) / 2 + sC[0];
                        const int x = abs(ix - s[0] - min(0, code[0]) * ((e[0] - s[0]) % 2)) % 2;
                        const int ixp = (abs(ix) % 2 == 1) ?  -1 : 1;
                        const int rxp = (ixp == 1) ? 1:0;
                        const int rx  = (ixp == 1) ? 0:1;
   
                        ElementType *Test[3][3][3];
                        for (int i = 0; i < 3; i++)
                           for (int j = 0; j < 3; j++)
                              for (int k = 0; k < 3; k++)
                                 Test[i][j][k] = &m_CoarsenedBlock->Access(XX - 1 + i - offset[0], YY - 1 + j - offset[1], ZZ - 1 + k - offset[2]);
   
                        TestInterp(Test,retval,x,y,z);
      
                        if (ix       >= s[0] && ix        < e[0] && iy       >= s[1] && iy        < e[1] && iz       >= s[2] && iz       < e[2])
                          m_cacheBlock->Access(ix     - m_stencilStart[0],iy       - m_stencilStart[1],iz       - m_stencilStart[2]) = retval[ rx  +2*ry  +4*rz ];
                        if (ix + ixp >= s[0] && ix + ixp  < e[0] && iy       >= s[1] && iy        < e[1] && iz       >= s[2] && iz       < e[2])
                          m_cacheBlock->Access(ix+ixp - m_stencilStart[0],iy       - m_stencilStart[1],iz       - m_stencilStart[2]) = retval[ rxp +2*ry  +4*rz ];
                        if (ix       >= s[0] && ix        < e[0] && iy + iyp >= s[1] && iy + iyp  < e[1] && iz       >= s[2] && iz       < e[2])
                          m_cacheBlock->Access(ix     - m_stencilStart[0],iy + iyp - m_stencilStart[1],iz       - m_stencilStart[2]) = retval[ rx  +2*ryp +4*rz ];
                        if (ix + ixp >= s[0] && ix + ixp  < e[0] && iy + iyp >= s[1] && iy + iyp  < e[1] && iz       >= s[2] && iz       < e[2])
                          m_cacheBlock->Access(ix+ixp - m_stencilStart[0],iy + iyp - m_stencilStart[1],iz       - m_stencilStart[2]) = retval[ rxp +2*ryp +4*rz ];
                        if (ix       >= s[0] && ix        < e[0] && iy       >= s[1] && iy        < e[1] && iz + izp >= s[2] && iz + izp < e[2])
                          m_cacheBlock->Access(ix     - m_stencilStart[0],iy       - m_stencilStart[1],iz + izp - m_stencilStart[2]) = retval[ rx  +2*ry  +4*rzp];
                        if (ix + ixp >= s[0] && ix + ixp  < e[0] && iy       >= s[1] && iy        < e[1] && iz + izp >= s[2] && iz + izp < e[2])
                          m_cacheBlock->Access(ix+ixp - m_stencilStart[0],iy       - m_stencilStart[1],iz + izp - m_stencilStart[2]) = retval[ rxp +2*ry  +4*rzp];
                        if (ix       >= s[0] && ix        < e[0] && iy + iyp >= s[1] && iy + iyp  < e[1] && iz + izp >= s[2] && iz + izp < e[2])
                          m_cacheBlock->Access(ix     - m_stencilStart[0],iy + iyp - m_stencilStart[1],iz + izp - m_stencilStart[2]) = retval[ rx  +2*ryp +4*rzp];
                        if (ix + ixp >= s[0] && ix + ixp  < e[0] && iy + iyp >= s[1] && iy + iyp  < e[1] && iz + izp >= s[2] && iz + izp < e[2])
                          m_cacheBlock->Access(ix+ixp - m_stencilStart[0],iy + iyp - m_stencilStart[1],iz + izp - m_stencilStart[2]) = retval[ rxp +2*ryp +4*rzp];
                     }
                  }
               }
            if (m_refGrid->FiniteDifferences && abs(code[0]) + abs(code[1]) + abs(code[2]) == 1) //Correct stencil points +-1 and +-2 at faces
            {
               const int coef_ixyz [3] = {min(0, code[0]) * ((e[0] - s[0]) % 2),
                                          min(0, code[1]) * ((e[1] - s[1]) % 2),
                                          min(0, code[2]) * ((e[2] - s[2]) % 2)}; 
               const int min_iz = max(s[2],-2);
               const int min_iy = max(s[1],-2);
               const int min_ix = max(s[0],-2);
               const int max_iz = min(e[2],nZ+2);
               const int max_iy = min(e[1],nY+2);
               const int max_ix = min(e[0],nX+2);

               for (int iz = min_iz; iz < max_iz; iz ++)
               {
                  const int ZZ  =    (iz - s[2] - coef_ixyz[2])/2 + sC[2] - offset[2];
                  const int z   = abs(iz - s[2] - coef_ixyz[2])%2;
                  const double dz = 0.25*(2*z-1);
                  const double * dz_coef = dz > 0 ? &d_coef_plus[0] : &d_coef_minus[0];
                  const bool zinner = (ZZ+offset[2] != 0) && (ZZ+offset[2] != CoarseBlockSize[2] - 1);
                  const bool zstart = (ZZ+offset[2] == 0);

                  #pragma GCC ivdep
                  for (int iy = min_iy; iy < max_iy; iy ++)
                  {
                     const int YY =    (iy - s[1] - coef_ixyz[1])/2 + sC[1] - offset[1];
                     const int y  = abs(iy - s[1] - coef_ixyz[1])%2;
                     const double dy = 0.25*(2*y-1);
                     const double * dy_coef = dy > 0 ? &d_coef_plus[0] : &d_coef_minus[0];
                     const bool yinner = (YY+offset[1] != 0) && (YY+offset[1] != CoarseBlockSize[1] - 1);
                     const bool ystart = (YY+offset[1] == 0);

                     #pragma GCC ivdep
                     for (int ix = min_ix; ix < max_ix; ix ++)
                     {
                        const int XX =    (ix - s[0] - coef_ixyz[0])/2 + sC[0] - offset[0];
                        const int x  = abs(ix - s[0] - coef_ixyz[0])%2;
                        const double dx = 0.25*(2*x-1);
                        const double * dx_coef = dx > 0 ? &d_coef_plus[0] : &d_coef_minus[0];
                        const bool xinner = (XX+offset[0] != 0) && (XX+offset[0] != CoarseBlockSize[0] - 1);
                        const bool xstart = (XX+offset[0] == 0);

                        auto & a = m_cacheBlock->Access(ix - m_stencilStart[0],iy - m_stencilStart[1],iz - m_stencilStart[2]);
                        if (code[0] != 0) //X-face
                        {
                           ElementType x1D,x2D,mixed;
                           if (yinner && zinner)
                           {
                              x1D = dy_coef[6] *  m_CoarsenedBlock->Access(XX,YY-1,ZZ  ) +
                                    dy_coef[7] *  m_CoarsenedBlock->Access(XX,YY  ,ZZ  ) +
                                    dy_coef[8] *  m_CoarsenedBlock->Access(XX,YY+1,ZZ  );
                              x2D = dz_coef[6] *  m_CoarsenedBlock->Access(XX,YY  ,ZZ-1) +
                                    dz_coef[7] *  m_CoarsenedBlock->Access(XX,YY  ,ZZ  ) +
                                    dz_coef[8] *  m_CoarsenedBlock->Access(XX,YY  ,ZZ+1);
                              mixed = dy*dz*0.25*(m_CoarsenedBlock->Access(XX,YY-1,ZZ-1)
                                                 +m_CoarsenedBlock->Access(XX,YY+1,ZZ+1)
                                                 -m_CoarsenedBlock->Access(XX,YY+1,ZZ-1)
                                                 -m_CoarsenedBlock->Access(XX,YY-1,ZZ+1));
                           }
                           else if (yinner)
                           {
                              x1D = dy_coef[6] * m_CoarsenedBlock->Access(XX,YY-1,ZZ) +
                                    dy_coef[7] * m_CoarsenedBlock->Access(XX,YY  ,ZZ) +
                                    dy_coef[8] * m_CoarsenedBlock->Access(XX,YY+1,ZZ);
                              if (zstart)
                              {
                                 x2D = dz_coef[0] *  m_CoarsenedBlock->Access(XX,YY  ,ZZ+2) +
                                       dz_coef[1] *  m_CoarsenedBlock->Access(XX,YY  ,ZZ+1) +
                                       dz_coef[2] *  m_CoarsenedBlock->Access(XX,YY  ,ZZ  );
                                 mixed = dy*dz*0.5 *(m_CoarsenedBlock->Access(XX,YY-1,ZZ  )
                                                    +m_CoarsenedBlock->Access(XX,YY+1,ZZ+1)
                                                    -m_CoarsenedBlock->Access(XX,YY+1,ZZ  )
                                                    -m_CoarsenedBlock->Access(XX,YY-1,ZZ+1));
                              }
                              else
                              {
                                 x2D = dz_coef[3] *  m_CoarsenedBlock->Access(XX,YY  ,ZZ-2) +
                                       dz_coef[4] *  m_CoarsenedBlock->Access(XX,YY  ,ZZ-1) +
                                       dz_coef[5] *  m_CoarsenedBlock->Access(XX,YY  ,ZZ  );
                                 mixed = dy*dz*0.5 *(m_CoarsenedBlock->Access(XX,YY-1,ZZ-1)
                                                    +m_CoarsenedBlock->Access(XX,YY+1,ZZ  )
                                                    -m_CoarsenedBlock->Access(XX,YY+1,ZZ-1)
                                                    -m_CoarsenedBlock->Access(XX,YY-1,ZZ  ));
                              }
                           }
                           else if (zinner)
                           {
                              x2D = dz_coef[6] * m_CoarsenedBlock->Access(XX,YY,ZZ-1) +
                                    dz_coef[7] * m_CoarsenedBlock->Access(XX,YY,ZZ  ) +
                                    dz_coef[8] * m_CoarsenedBlock->Access(XX,YY,ZZ+1);
                              if (ystart)
                              {
                                 x1D = dy_coef[0] *  m_CoarsenedBlock->Access(XX,YY+2,ZZ  ) +
                                       dy_coef[1] *  m_CoarsenedBlock->Access(XX,YY+1,ZZ  ) +
                                       dy_coef[2] *  m_CoarsenedBlock->Access(XX,YY  ,ZZ  );
                                 mixed = dy*dz*0.5 *(m_CoarsenedBlock->Access(XX,YY  ,ZZ-1)
                                                    +m_CoarsenedBlock->Access(XX,YY+1,ZZ+1)
                                                    -m_CoarsenedBlock->Access(XX,YY+1,ZZ-1)
                                                    -m_CoarsenedBlock->Access(XX,YY  ,ZZ+1));
                              }
                              else
                              {
                                 x1D = dy_coef[3] *  m_CoarsenedBlock->Access(XX,YY-2,ZZ  ) +
                                       dy_coef[4] *  m_CoarsenedBlock->Access(XX,YY-1,ZZ  ) +
                                       dy_coef[5] *  m_CoarsenedBlock->Access(XX,YY  ,ZZ  );
                                 mixed = dy*dz*0.5 *(m_CoarsenedBlock->Access(XX,YY-1,ZZ-1)
                                                    +m_CoarsenedBlock->Access(XX,YY  ,ZZ+1)
                                                    -m_CoarsenedBlock->Access(XX,YY  ,ZZ-1)
                                                    -m_CoarsenedBlock->Access(XX,YY-1,ZZ+1));
                              }
                           }
                           else if (zstart)
                           {
                              x2D = dz_coef[0] * m_CoarsenedBlock->Access(XX,YY,ZZ+2) +
                                    dz_coef[1] * m_CoarsenedBlock->Access(XX,YY,ZZ+1) +
                                    dz_coef[2] * m_CoarsenedBlock->Access(XX,YY,ZZ  );
                              if (ystart)
                              {
                                 x1D = dy_coef[0] * m_CoarsenedBlock->Access(XX,YY+2,ZZ  ) +
                                       dy_coef[1] * m_CoarsenedBlock->Access(XX,YY+1,ZZ  ) +
                                       dy_coef[2] * m_CoarsenedBlock->Access(XX,YY  ,ZZ  );
                                 mixed = dy*dz*    (m_CoarsenedBlock->Access(XX,YY  ,ZZ  )
                                                   +m_CoarsenedBlock->Access(XX,YY+1,ZZ+1)
                                                   -m_CoarsenedBlock->Access(XX,YY+1,ZZ  )
                                                   -m_CoarsenedBlock->Access(XX,YY  ,ZZ+1));
                              }
                              else
                              {
                                 x1D = dy_coef[3] * m_CoarsenedBlock->Access(XX,YY-2,ZZ  ) +
                                       dy_coef[4] * m_CoarsenedBlock->Access(XX,YY-1,ZZ  ) +
                                       dy_coef[5] * m_CoarsenedBlock->Access(XX,YY  ,ZZ  );
                                 mixed = dy*dz*    (m_CoarsenedBlock->Access(XX,YY-1,ZZ  )
                                                   +m_CoarsenedBlock->Access(XX,YY  ,ZZ+1)
                                                   -m_CoarsenedBlock->Access(XX,YY  ,ZZ  )
                                                   -m_CoarsenedBlock->Access(XX,YY-1,ZZ+1));
                              }
                           }
                           else if (ystart) // and !zstart
                           {
                              x1D = dy_coef[0] * m_CoarsenedBlock->Access(XX,YY+2,ZZ  ) +
                                    dy_coef[1] * m_CoarsenedBlock->Access(XX,YY+1,ZZ  ) +
                                    dy_coef[2] * m_CoarsenedBlock->Access(XX,YY  ,ZZ  );
                              x2D = dz_coef[3] * m_CoarsenedBlock->Access(XX,YY  ,ZZ-2) +
                                    dz_coef[4] * m_CoarsenedBlock->Access(XX,YY  ,ZZ-1) +
                                    dz_coef[5] * m_CoarsenedBlock->Access(XX,YY  ,ZZ  );
                              mixed = dy*dz*    (m_CoarsenedBlock->Access(XX,YY  ,ZZ-1)
                                                +m_CoarsenedBlock->Access(XX,YY+1,ZZ  )
                                                -m_CoarsenedBlock->Access(XX,YY+1,ZZ-1)
                                                -m_CoarsenedBlock->Access(XX,YY  ,ZZ  ));
                           }
                           else // !ystart and !zstart
                           {
                              x1D = dy_coef[3] * m_CoarsenedBlock->Access(XX,YY-2,ZZ  ) +
                                    dy_coef[4] * m_CoarsenedBlock->Access(XX,YY-1,ZZ  ) +
                                    dy_coef[5] * m_CoarsenedBlock->Access(XX,YY  ,ZZ  );
                              x2D = dz_coef[3] * m_CoarsenedBlock->Access(XX,YY  ,ZZ-2) +
                                    dz_coef[4] * m_CoarsenedBlock->Access(XX,YY  ,ZZ-1) +
                                    dz_coef[5] * m_CoarsenedBlock->Access(XX,YY  ,ZZ  );
                              mixed = dy*dz*    (m_CoarsenedBlock->Access(XX,YY  ,ZZ  )
                                                +m_CoarsenedBlock->Access(XX,YY-1,ZZ-1)
                                                -m_CoarsenedBlock->Access(XX,YY-1,ZZ  )
                                                -m_CoarsenedBlock->Access(XX,YY  ,ZZ-1));
                           }
                           a = x1D + x2D + mixed;
                        }
                        else if (code[1] != 0) //Y-face
                        {
                           ElementType x1D,x2D,mixed;
                           if (xinner && zinner)
                           {
                              x1D = dx_coef[6] *  m_CoarsenedBlock->Access(XX-1,YY,ZZ  ) +
                                    dx_coef[7] *  m_CoarsenedBlock->Access(XX  ,YY,ZZ  ) +
                                    dx_coef[8] *  m_CoarsenedBlock->Access(XX+1,YY,ZZ  );
                              x2D = dz_coef[6] *  m_CoarsenedBlock->Access(XX  ,YY,ZZ-1) +
                                    dz_coef[7] *  m_CoarsenedBlock->Access(XX  ,YY,ZZ  ) +
                                    dz_coef[8] *  m_CoarsenedBlock->Access(XX  ,YY,ZZ+1);
                              mixed = dx*dz*0.25*(m_CoarsenedBlock->Access(XX-1,YY,ZZ-1)
                                                 +m_CoarsenedBlock->Access(XX+1,YY,ZZ+1)
                                                 -m_CoarsenedBlock->Access(XX+1,YY,ZZ-1)
                                                 -m_CoarsenedBlock->Access(XX-1,YY,ZZ+1));
                           }
                           else if (xinner)
                           {
                              x1D = dx_coef[6] * m_CoarsenedBlock->Access(XX-1,YY,ZZ) +
                                    dx_coef[7] * m_CoarsenedBlock->Access(XX  ,YY,ZZ) +
                                    dx_coef[8] * m_CoarsenedBlock->Access(XX+1,YY,ZZ);
                              if (zstart)
                              {
                                 x2D = dz_coef[0] *  m_CoarsenedBlock->Access(XX  ,YY,ZZ+2) +
                                       dz_coef[1] *  m_CoarsenedBlock->Access(XX  ,YY,ZZ+1) +
                                       dz_coef[2] *  m_CoarsenedBlock->Access(XX  ,YY,ZZ  );
                                 mixed = dx*dz*0.5 *(m_CoarsenedBlock->Access(XX-1,YY,ZZ  )
                                                    +m_CoarsenedBlock->Access(XX+1,YY,ZZ+1)
                                                    -m_CoarsenedBlock->Access(XX+1,YY,ZZ  )
                                                    -m_CoarsenedBlock->Access(XX-1,YY,ZZ+1));
                              }
                              else
                              {
                                 x2D = dz_coef[3] *  m_CoarsenedBlock->Access(XX  ,YY,ZZ-2) +
                                       dz_coef[4] *  m_CoarsenedBlock->Access(XX  ,YY,ZZ-1) +
                                       dz_coef[5] *  m_CoarsenedBlock->Access(XX  ,YY,ZZ  );
                                 mixed = dx*dz*0.5 *(m_CoarsenedBlock->Access(XX-1,YY,ZZ-1)
                                                    +m_CoarsenedBlock->Access(XX+1,YY,ZZ  )
                                                    -m_CoarsenedBlock->Access(XX+1,YY,ZZ-1)
                                                    -m_CoarsenedBlock->Access(XX-1,YY,ZZ  ));
                              }
                           }
                           else if (zinner)
                           {
                              x2D = dz_coef[6] * m_CoarsenedBlock->Access(XX,YY,ZZ-1) +
                                    dz_coef[7] * m_CoarsenedBlock->Access(XX,YY,ZZ  ) +
                                    dz_coef[8] * m_CoarsenedBlock->Access(XX,YY,ZZ+1);
                              if (xstart)
                              {
                                 x1D = dx_coef[0] *  m_CoarsenedBlock->Access(XX+2,YY,ZZ  ) +
                                       dx_coef[1] *  m_CoarsenedBlock->Access(XX+1,YY,ZZ  ) +
                                       dx_coef[2] *  m_CoarsenedBlock->Access(XX  ,YY,ZZ  );
                                 mixed = dx*dz*0.5 *(m_CoarsenedBlock->Access(XX  ,YY,ZZ-1)
                                                    +m_CoarsenedBlock->Access(XX+1,YY,ZZ+1)
                                                    -m_CoarsenedBlock->Access(XX+1,YY,ZZ-1)
                                                    -m_CoarsenedBlock->Access(XX  ,YY,ZZ+1));
                              }
                              else
                              {
                                 x1D = dx_coef[3] *  m_CoarsenedBlock->Access(XX-2,YY,ZZ  ) +
                                       dx_coef[4] *  m_CoarsenedBlock->Access(XX-1,YY,ZZ  ) +
                                       dx_coef[5] *  m_CoarsenedBlock->Access(XX  ,YY,ZZ  );
                                 mixed = dx*dz*0.5 *(m_CoarsenedBlock->Access(XX-1,YY,ZZ-1)
                                                    +m_CoarsenedBlock->Access(XX  ,YY,ZZ+1)
                                                    -m_CoarsenedBlock->Access(XX  ,YY,ZZ-1)
                                                    -m_CoarsenedBlock->Access(XX-1,YY,ZZ+1));
                              }
                           }
                           else if (zstart)
                           {
                              x2D = dz_coef[0] * m_CoarsenedBlock->Access(XX,YY,ZZ+2) +
                                    dz_coef[1] * m_CoarsenedBlock->Access(XX,YY,ZZ+1) +
                                    dz_coef[2] * m_CoarsenedBlock->Access(XX,YY,ZZ  );
                              if (xstart)
                              {
                                 x1D = dx_coef[0] * m_CoarsenedBlock->Access(XX+2,YY,ZZ  ) +
                                       dx_coef[1] * m_CoarsenedBlock->Access(XX+1,YY,ZZ  ) +
                                       dx_coef[2] * m_CoarsenedBlock->Access(XX  ,YY,ZZ  );
                                 mixed = dx*dz*    (m_CoarsenedBlock->Access(XX  ,YY,ZZ  )
                                                   +m_CoarsenedBlock->Access(XX+1,YY,ZZ+1)
                                                   -m_CoarsenedBlock->Access(XX+1,YY,ZZ  )
                                                   -m_CoarsenedBlock->Access(XX  ,YY,ZZ+1));
                              }
                              else
                              {
                                 x1D = dx_coef[3] * m_CoarsenedBlock->Access(XX-2,YY,ZZ  ) +
                                       dx_coef[4] * m_CoarsenedBlock->Access(XX-1,YY,ZZ  ) +
                                       dx_coef[5] * m_CoarsenedBlock->Access(XX  ,YY,ZZ  );
                                 mixed = dx*dz*    (m_CoarsenedBlock->Access(XX-1,YY,ZZ  )
                                                   +m_CoarsenedBlock->Access(XX  ,YY,ZZ+1)
                                                   -m_CoarsenedBlock->Access(XX  ,YY,ZZ  )
                                                   -m_CoarsenedBlock->Access(XX-1,YY,ZZ+1));
                              }
                           }
                           else if (xstart) // and !zstart
                           {
                              x1D = dx_coef[0] * m_CoarsenedBlock->Access(XX+2,YY,ZZ  ) +
                                    dx_coef[1] * m_CoarsenedBlock->Access(XX+1,YY,ZZ  ) +
                                    dx_coef[2] * m_CoarsenedBlock->Access(XX  ,YY,ZZ  );
                              x2D = dz_coef[3] * m_CoarsenedBlock->Access(XX  ,YY,ZZ-2) +
                                    dz_coef[4] * m_CoarsenedBlock->Access(XX  ,YY,ZZ-1) +
                                    dz_coef[5] * m_CoarsenedBlock->Access(XX  ,YY,ZZ  );
                              mixed = dx*dz*    (m_CoarsenedBlock->Access(XX  ,YY,ZZ-1)
                                                +m_CoarsenedBlock->Access(XX+1,YY,ZZ  )
                                                -m_CoarsenedBlock->Access(XX+1,YY,ZZ-1)
                                                -m_CoarsenedBlock->Access(XX  ,YY,ZZ  ));
                           }
                           else // !xstart and !zstart
                           {
                              x1D = dx_coef[3] * m_CoarsenedBlock->Access(XX-2,YY,ZZ  ) +
                                    dx_coef[4] * m_CoarsenedBlock->Access(XX-1,YY,ZZ  ) +
                                    dx_coef[5] * m_CoarsenedBlock->Access(XX  ,YY,ZZ  );
                              x2D = dz_coef[3] * m_CoarsenedBlock->Access(XX  ,YY,ZZ-2) +
                                    dz_coef[4] * m_CoarsenedBlock->Access(XX  ,YY,ZZ-1) +
                                    dz_coef[5] * m_CoarsenedBlock->Access(XX  ,YY,ZZ  );
                              mixed = dx*dz*    (m_CoarsenedBlock->Access(XX  ,YY,ZZ  )
                                                +m_CoarsenedBlock->Access(XX-1,YY,ZZ-1)
                                                -m_CoarsenedBlock->Access(XX-1,YY,ZZ  )
                                                -m_CoarsenedBlock->Access(XX  ,YY,ZZ-1));
                           }
                           a = x1D + x2D + mixed;                     
                        }
                        else if (code[2] != 0) //Z-face
                        {
                           ElementType x1D,x2D,mixed;
                           if (xinner && yinner)
                           {
                              x1D = dx_coef[6] *  m_CoarsenedBlock->Access(XX-1,YY  ,ZZ) +
                                    dx_coef[7] *  m_CoarsenedBlock->Access(XX  ,YY  ,ZZ) +
                                    dx_coef[8] *  m_CoarsenedBlock->Access(XX+1,YY  ,ZZ);
                              x2D = dy_coef[6] *  m_CoarsenedBlock->Access(XX  ,YY-1,ZZ) +
                                    dy_coef[7] *  m_CoarsenedBlock->Access(XX  ,YY  ,ZZ) +
                                    dy_coef[8] *  m_CoarsenedBlock->Access(XX  ,YY+1,ZZ);
                              mixed = dx*dy*0.25*(m_CoarsenedBlock->Access(XX-1,YY-1,ZZ)
                                                 +m_CoarsenedBlock->Access(XX+1,YY+1,ZZ)
                                                 -m_CoarsenedBlock->Access(XX+1,YY-1,ZZ)
                                                 -m_CoarsenedBlock->Access(XX-1,YY+1,ZZ));
                           }
                           else if (xinner)
                           {
                              x1D = dx_coef[6] * m_CoarsenedBlock->Access(XX-1,YY,ZZ) +
                                    dx_coef[7] * m_CoarsenedBlock->Access(XX  ,YY,ZZ) +
                                    dx_coef[8] * m_CoarsenedBlock->Access(XX+1,YY,ZZ);
                              if (ystart)
                              {
                                 x2D = dy_coef[0] *  m_CoarsenedBlock->Access(XX  ,YY+2,ZZ) +
                                       dy_coef[1] *  m_CoarsenedBlock->Access(XX  ,YY+1,ZZ) +
                                       dy_coef[2] *  m_CoarsenedBlock->Access(XX  ,YY  ,ZZ);
                                 mixed = dx*dy*0.5 *(m_CoarsenedBlock->Access(XX-1,YY  ,ZZ)
                                                    +m_CoarsenedBlock->Access(XX+1,YY+1,ZZ)
                                                    -m_CoarsenedBlock->Access(XX+1,YY  ,ZZ)
                                                    -m_CoarsenedBlock->Access(XX-1,YY+1,ZZ));
                              }
                              else
                              {
                                 x2D = dy_coef[3] *  m_CoarsenedBlock->Access(XX  ,YY-2,ZZ) +
                                       dy_coef[4] *  m_CoarsenedBlock->Access(XX  ,YY-1,ZZ) +
                                       dy_coef[5] *  m_CoarsenedBlock->Access(XX  ,YY  ,ZZ);
                                 mixed = dx*dy*0.5 *(m_CoarsenedBlock->Access(XX-1,YY-1,ZZ)
                                                    +m_CoarsenedBlock->Access(XX+1,YY  ,ZZ)
                                                    -m_CoarsenedBlock->Access(XX+1,YY-1,ZZ)
                                                    -m_CoarsenedBlock->Access(XX-1,YY  ,ZZ));
                              }
                           }
                           else if (yinner)
                           {
                              x2D = dy_coef[6] * m_CoarsenedBlock->Access(XX,YY-1,ZZ) +
                                    dy_coef[7] * m_CoarsenedBlock->Access(XX,YY  ,ZZ) +
                                    dy_coef[8] * m_CoarsenedBlock->Access(XX,YY+1,ZZ);
                              if (xstart)
                              {
                                 x1D = dx_coef[0] *  m_CoarsenedBlock->Access(XX+2,YY  ,ZZ) +
                                       dx_coef[1] *  m_CoarsenedBlock->Access(XX+1,YY  ,ZZ) +
                                       dx_coef[2] *  m_CoarsenedBlock->Access(XX  ,YY  ,ZZ);
                                 mixed = dx*dy*0.5 *(m_CoarsenedBlock->Access(XX  ,YY-1,ZZ)
                                                    +m_CoarsenedBlock->Access(XX+1,YY+1,ZZ)
                                                    -m_CoarsenedBlock->Access(XX+1,YY-1,ZZ)
                                                    -m_CoarsenedBlock->Access(XX  ,YY+1,ZZ));
                              }
                              else
                              {
                                 x1D = dx_coef[3] *  m_CoarsenedBlock->Access(XX-2,YY  ,ZZ) +
                                       dx_coef[4] *  m_CoarsenedBlock->Access(XX-1,YY  ,ZZ) +
                                       dx_coef[5] *  m_CoarsenedBlock->Access(XX  ,YY  ,ZZ);
                                 mixed = dx*dy*0.5 *(m_CoarsenedBlock->Access(XX-1,YY-1,ZZ)
                                                    +m_CoarsenedBlock->Access(XX  ,YY+1,ZZ)
                                                    -m_CoarsenedBlock->Access(XX  ,YY-1,ZZ)
                                                    -m_CoarsenedBlock->Access(XX-1,YY+1,ZZ));
                              }
                           }
                           else if (ystart)
                           {
                              x2D = dy_coef[0] * m_CoarsenedBlock->Access(XX,YY+2,ZZ) +
                                    dy_coef[1] * m_CoarsenedBlock->Access(XX,YY+1,ZZ) +
                                    dy_coef[2] * m_CoarsenedBlock->Access(XX,YY  ,ZZ);
                              if (xstart)
                              {
                                 x1D = dx_coef[0] * m_CoarsenedBlock->Access(XX+2,YY  ,ZZ) +
                                       dx_coef[1] * m_CoarsenedBlock->Access(XX+1,YY  ,ZZ) +
                                       dx_coef[2] * m_CoarsenedBlock->Access(XX  ,YY  ,ZZ);
                                 mixed = dx*dy*    (m_CoarsenedBlock->Access(XX  ,YY  ,ZZ)
                                                   +m_CoarsenedBlock->Access(XX+1,YY+1,ZZ)
                                                   -m_CoarsenedBlock->Access(XX+1,YY  ,ZZ)
                                                   -m_CoarsenedBlock->Access(XX  ,YY+1,ZZ));
                              }
                              else
                              {
                                 x1D = dx_coef[3] * m_CoarsenedBlock->Access(XX-2,YY  ,ZZ) +
                                       dx_coef[4] * m_CoarsenedBlock->Access(XX-1,YY  ,ZZ) +
                                       dx_coef[5] * m_CoarsenedBlock->Access(XX  ,YY  ,ZZ);
                                 mixed = dx*dy*    (m_CoarsenedBlock->Access(XX-1,YY  ,ZZ)
                                                   +m_CoarsenedBlock->Access(XX  ,YY+1,ZZ)
                                                   -m_CoarsenedBlock->Access(XX  ,YY  ,ZZ)
                                                   -m_CoarsenedBlock->Access(XX-1,YY+1,ZZ));
                              }
                           }
                           else if (xstart) // and !ystart
                           {
                              x1D = dx_coef[0] * m_CoarsenedBlock->Access(XX+2,YY  ,ZZ) +
                                    dx_coef[1] * m_CoarsenedBlock->Access(XX+1,YY  ,ZZ) +
                                    dx_coef[2] * m_CoarsenedBlock->Access(XX  ,YY  ,ZZ);
                              x2D = dy_coef[3] * m_CoarsenedBlock->Access(XX  ,YY-2,ZZ) +
                                    dy_coef[4] * m_CoarsenedBlock->Access(XX  ,YY-1,ZZ) +
                                    dy_coef[5] * m_CoarsenedBlock->Access(XX  ,YY  ,ZZ);
                              mixed = dx*dy*    (m_CoarsenedBlock->Access(XX  ,YY-1,ZZ)
                                                +m_CoarsenedBlock->Access(XX+1,YY  ,ZZ)
                                                -m_CoarsenedBlock->Access(XX+1,YY-1,ZZ)
                                                -m_CoarsenedBlock->Access(XX  ,YY  ,ZZ));
                           }
                           else // !xstart and !ystart
                           {
                              x1D = dx_coef[3] * m_CoarsenedBlock->Access(XX-2,YY  ,ZZ) +
                                    dx_coef[4] * m_CoarsenedBlock->Access(XX-1,YY  ,ZZ) +
                                    dx_coef[5] * m_CoarsenedBlock->Access(XX  ,YY  ,ZZ);
                              x2D = dy_coef[3] * m_CoarsenedBlock->Access(XX  ,YY-2,ZZ) +
                                    dy_coef[4] * m_CoarsenedBlock->Access(XX  ,YY-1,ZZ) +
                                    dy_coef[5] * m_CoarsenedBlock->Access(XX  ,YY  ,ZZ);
                              mixed = dx*dy*    (m_CoarsenedBlock->Access(XX  ,YY  ,ZZ)
                                                +m_CoarsenedBlock->Access(XX-1,YY-1,ZZ)
                                                -m_CoarsenedBlock->Access(XX-1,YY  ,ZZ)
                                                -m_CoarsenedBlock->Access(XX  ,YY-1,ZZ));
                           }
                           a = x1D + x2D + mixed;
                        }
                        const auto & b = m_cacheBlock->Access(ix - m_stencilStart[0] + (-3*code[0]+1)/2 - x*abs(code[0]),
                                                              iy - m_stencilStart[1] + (-3*code[1]+1)/2 - y*abs(code[1]),
                                                              iz - m_stencilStart[2] + (-3*code[2]+1)/2 - z*abs(code[2]));
                        const auto & c = m_cacheBlock->Access(ix - m_stencilStart[0] + (-5*code[0]+1)/2 - x*abs(code[0]),
                                                              iy - m_stencilStart[1] + (-5*code[1]+1)/2 - y*abs(code[1]),
                                                              iz - m_stencilStart[2] + (-5*code[2]+1)/2 - z*abs(code[2]));
                        const int ccc  = code[0] + code[1] + code[2];
                        const int xyz  = abs(code[0])*x+abs(code[1])*y+abs(code[2])*z;
                        if (ccc == 1)     a = (xyz==0)?(1.0/15.0)*(8.0*a+10.0*b-3.0*c):(1.0/15.0)*(24.0*a-15.0*b+6*c);
                        else /*(ccc=-1)*/ a = (xyz==1)?(1.0/15.0)*(8.0*a+10.0*b-3.0*c):(1.0/15.0)*(24.0*a-15.0*b+6*c);
                     }
                  }
               }
            }
         #else

            if (use_averages)
            {
               #pragma GCC ivdep
               for (int iy = s[1]; iy < e[1]; iy += 1)
               {
                  const int YY = (iy - s[1] - min(0, code[1]) * ((e[1] - s[1]) % 2)) / 2 + sC[1];
                  #pragma GCC ivdep
                  for (int ix = s[0]; ix < e[0]; ix += 1)
                  {
                     const int XX = (ix - s[0] - min(0, code[0]) * ((e[0] - s[0]) % 2)) / 2 + sC[0];
                     ElementType *Test[3][3];
                     for (int i = 0; i < 3; i++)
                        for (int j = 0; j < 3; j++)
                              Test[i][j] = &m_CoarsenedBlock->Access(XX - 1 + i - offset[0],
                                                                     YY - 1 + j - offset[1],0);
                     TestInterp(Test,m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1],0),
                                abs(ix - s[0] - min(0, code[0]) * ((e[0] - s[0]) % 2)) % 2,
                                abs(iy - s[1] - min(0, code[1]) * ((e[1] - s[1]) % 2)) % 2);
                  }
               }
            }
            if (m_refGrid->FiniteDifferences && abs(code[0]) + abs(code[1]) == 1) //Correct stencil points +-1 and +-2 at faces
            {
               #pragma GCC ivdep
               for (int iy = s[1]; iy < e[1]; iy += 2)
               {
                  const int YY = (iy - s[1] - min(0, code[1]) * ((e[1] - s[1]) % 2)) / 2 + sC[1]- offset[1];
                  const int y = abs(iy - s[1] - min(0, code[1]) * ((e[1] - s[1]) % 2)) % 2;
                  const int iyp = (abs(iy) % 2 == 1) ?  -1 : 1;
                  const double dy = 0.25*(2*y-1);

                  #pragma GCC ivdep
                  for (int ix = s[0]; ix < e[0]; ix += 2)
                  {
                     const int XX = (ix - s[0] - min(0, code[0]) * ((e[0] - s[0]) % 2)) / 2 + sC[0]- offset[0];
                     const int x = abs(ix - s[0] - min(0, code[0]) * ((e[0] - s[0]) % 2)) % 2;
                     const int ixp = (abs(ix) % 2 == 1) ?  -1 : 1;
                     const double dx = 0.25*(2*x-1);
                     if (ix < -2 || iy < -2 || ix > nX+1 || iy > nY+1) continue;

                     if (code[0] != 0)
                     {
                        ElementType dudy,dudy2;
                        if (YY+offset[1] == 0)
                        {
                           dudy  = (-0.5*m_CoarsenedBlock->Access(XX,YY+2,0) - 1.5*m_CoarsenedBlock->Access(XX,YY,0))+ 2.0*m_CoarsenedBlock->Access(XX,YY+1,0) ;
                           dudy2 = (m_CoarsenedBlock->Access(XX,YY+2,0)+m_CoarsenedBlock->Access(XX,YY,0))-2.0*m_CoarsenedBlock->Access(XX,YY+1,0);
                        }
                        else if (YY+offset[1] == CoarseBlockSize[1] - 1)
                        {
                           dudy  = (0.5*m_CoarsenedBlock->Access(XX,YY-2,0) + 1.5*m_CoarsenedBlock->Access(XX,YY,0) )- 2.0*m_CoarsenedBlock->Access(XX,YY-1,0) ;
                           dudy2 = (m_CoarsenedBlock->Access(XX,YY-2,0)+m_CoarsenedBlock->Access(XX,YY,0))-2.0*m_CoarsenedBlock->Access(XX,YY-1,0);
                        }
                        else
                        {
                           dudy  = 0.5*(m_CoarsenedBlock->Access(XX,YY+1,0)-m_CoarsenedBlock->Access(XX,YY-1,0));
                           dudy2 = (m_CoarsenedBlock->Access(XX,YY+1,0)+m_CoarsenedBlock->Access(XX,YY-1,0))-2.0*m_CoarsenedBlock->Access(XX,YY,0);
                        }
                        m_cacheBlock->Access(ix - m_stencilStart[0]    , iy - m_stencilStart[1]    ,0) = m_CoarsenedBlock->Access(XX,YY,0) + dy*dudy + (0.5*dy*dy)*dudy2;
                        if (iy + iyp >= s[1] && iy + iyp  < e[1]) m_cacheBlock->Access(ix - m_stencilStart[0]    , iy - m_stencilStart[1]+iyp,0) = m_CoarsenedBlock->Access(XX,YY,0) - dy*dudy + (0.5*dy*dy)*dudy2;
                        if (ix + ixp >= s[0] && ix + ixp  < e[0]) m_cacheBlock->Access(ix - m_stencilStart[0]+ixp, iy - m_stencilStart[1]    ,0) = m_CoarsenedBlock->Access(XX,YY,0) + dy*dudy + (0.5*dy*dy)*dudy2;
                        if (ix + ixp >= s[0] && ix + ixp  < e[0] && iy + iyp >= s[1] && iy + iyp  < e[1]) m_cacheBlock->Access(ix - m_stencilStart[0]+ixp, iy - m_stencilStart[1]+iyp,0) = m_CoarsenedBlock->Access(XX,YY,0) - dy*dudy + (0.5*dy*dy)*dudy2;
                     }
                     else //if (code[1] != 0)
                     {
                        ElementType dudx,dudx2;
                        if (XX+offset[0] == 0)
                        {
                           dudx  = (-0.5*m_CoarsenedBlock->Access(XX+2,YY,0)- 1.5*m_CoarsenedBlock->Access(XX,YY,0)) + 2.0*m_CoarsenedBlock->Access(XX+1,YY,0) ;
                           dudx2 = (m_CoarsenedBlock->Access(XX+2,YY,0)+m_CoarsenedBlock->Access(XX,YY,0))-2.0*m_CoarsenedBlock->Access(XX+1,YY,0);
                        }
                        else if (XX+offset[0] == CoarseBlockSize[0] - 1)
                        {
                           dudx  = (0.5*m_CoarsenedBlock->Access(XX-2,YY,0)+ 1.5*m_CoarsenedBlock->Access(XX,YY,0)) - 2.0*m_CoarsenedBlock->Access(XX-1,YY,0) ;
                           dudx2 = (m_CoarsenedBlock->Access(XX-2,YY,0)+m_CoarsenedBlock->Access(XX,YY,0))-2.0*m_CoarsenedBlock->Access(XX-1,YY,0);
                        }
                        else
                        {
                           dudx  = 0.5*(m_CoarsenedBlock->Access(XX+1,YY,0)-m_CoarsenedBlock->Access(XX-1,YY,0));
                           dudx2 = (m_CoarsenedBlock->Access(XX+1,YY,0)+m_CoarsenedBlock->Access(XX-1,YY,0))-2.0*m_CoarsenedBlock->Access(XX,YY,0);
                        }
                        m_cacheBlock->Access(ix - m_stencilStart[0]    , iy - m_stencilStart[1]    ,0) = m_CoarsenedBlock->Access(XX,YY,0) + dx*dudx + (0.5*dx*dx)*dudx2;
                        if (iy + iyp >= s[1] && iy + iyp  < e[1]) m_cacheBlock->Access(ix - m_stencilStart[0]    , iy - m_stencilStart[1]+iyp,0) = m_CoarsenedBlock->Access(XX,YY,0) + dx*dudx + (0.5*dx*dx)*dudx2;
                        if (ix + ixp >= s[0] && ix + ixp  < e[0]) m_cacheBlock->Access(ix - m_stencilStart[0]+ixp, iy - m_stencilStart[1]    ,0) = m_CoarsenedBlock->Access(XX,YY,0) - dx*dudx + (0.5*dx*dx)*dudx2;
                        if (ix + ixp >= s[0] && ix + ixp  < e[0] && iy + iyp >= s[1] && iy + iyp  < e[1]) m_cacheBlock->Access(ix - m_stencilStart[0]+ixp, iy - m_stencilStart[1]+iyp,0) = m_CoarsenedBlock->Access(XX,YY,0) - dx*dudx + (0.5*dx*dx)*dudx2;
                     }
                  }

               }

               for (int iy = s[1]; iy < e[1]; iy += 1)
               {
               #pragma GCC ivdep
               for (int ix = s[0]; ix < e[0]; ix += 1)
               {
                  if (ix < -2 || iy < -2 || ix > nX+1 || iy > nY+1) continue;
                  const int x = abs(ix - s[0] - min(0, code[0]) * ((e[0] - s[0]) % 2)) % 2;
                  const int y = abs(iy - s[1] - min(0, code[1]) * ((e[1] - s[1]) % 2)) % 2;

                  auto & a = m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1],0);

                  if (code[0] == 0 && code[1] == 1)
                  {
                     if (y==0) //interpolation
                     {
                        auto & b = m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1] - 1,0);
                        auto & c = m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1] - 2,0);
                        LI(a,b,c);
                     }
                     else if (y==1) //extrapolation
                     {
                        auto & b = m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1] - 2,0);
                        auto & c = m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1] - 3,0);
                        LE(a,b,c);
                     }
                  }
                  else if (code[0] == 0 && code[1] == -1)
                  {
                     if (y==1) //interpolation
                     {
                        auto & b = m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1] + 1,0);
                        auto & c = m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1] + 2,0);
                        LI(a,b,c);
                     }
                     else if (y==0) //extrapolation
                     {
                        auto & b = m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1] + 2,0);
                        auto & c = m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1] + 3,0);
                        LE(a,b,c);
                     }
                  }
                  else if (code[1] == 0 && code[0] == 1)
                  {
                     if (x==0) //interpolation
                     {
                        auto & b = m_cacheBlock->Access(ix - m_stencilStart[0] - 1, iy - m_stencilStart[1],0);
                        auto & c = m_cacheBlock->Access(ix - m_stencilStart[0] - 2, iy - m_stencilStart[1],0);
                        LI(a,b,c);
                     }
                     else if (x==1) //extrapolation
                     {
                        auto & b = m_cacheBlock->Access(ix - m_stencilStart[0] - 2, iy - m_stencilStart[1],0);
                        auto & c = m_cacheBlock->Access(ix - m_stencilStart[0] - 3, iy - m_stencilStart[1],0);
                        LE(a,b,c);
                     }
                  }
                  else if (code[1] == 0 && code[0] == -1)
                  {
                     if (x==1) //interpolation
                     {
                        auto & b = m_cacheBlock->Access(ix - m_stencilStart[0] + 1, iy - m_stencilStart[1],0);
                        auto & c = m_cacheBlock->Access(ix - m_stencilStart[0] + 2, iy - m_stencilStart[1],0);
                        LI(a,b,c);
                     }
                     else if (x==0) //extrapolation
                     {
                        auto & b = m_cacheBlock->Access(ix - m_stencilStart[0] + 2, iy - m_stencilStart[1],0);
                        auto & c = m_cacheBlock->Access(ix - m_stencilStart[0] + 3, iy - m_stencilStart[1],0);
                        LE(a,b,c);
                     }
                  }
               }
               }
            }
         #endif
      }
   }

   /**
    * Get a single element from the block.
    * stencil_start and stencil_end refer to the values passed in BlockLab::prepare().
    * ix: Index in x-direction (stencil_start[0] <= ix < BlockType::sizeX + stencil_end[0] - 1).
    * iy: Index in y-direction (stencil_start[1] <= iy < BlockType::sizeY + stencil_end[1] - 1).
    * iz: Index in z-direction (stencil_start[2] <= iz < BlockType::sizeZ + stencil_end[2] - 1).
    */
   ElementType &operator()(int ix, int iy = 0, int iz = 0)
   {
      assert(ix - m_stencilStart[0] >= 0 && ix - m_stencilStart[0] < (int)m_cacheBlock->getSize()[0]);
      assert(iy - m_stencilStart[1] >= 0 && iy - m_stencilStart[1] < (int)m_cacheBlock->getSize()[1]);
      assert(iz - m_stencilStart[2] >= 0 && iz - m_stencilStart[2] < (int)m_cacheBlock->getSize()[2]);
      return m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1], iz - m_stencilStart[2]);
   }

   const ElementType &operator()(int ix, int iy = 0, int iz = 0) const
   {
      assert(ix - m_stencilStart[0] >= 0 && ix - m_stencilStart[0] < (int)m_cacheBlock->getSize()[0]);
      assert(iy - m_stencilStart[1] >= 0 && iy - m_stencilStart[1] < (int)m_cacheBlock->getSize()[1]);
      assert(iz - m_stencilStart[2] >= 0 && iz - m_stencilStart[2] < (int)m_cacheBlock->getSize()[2]);
      return m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1], iz - m_stencilStart[2]);
   }

   /** Just as BlockLab::operator() but returning a const. */
   const ElementType &read(int ix, int iy = 0, int iz = 0) const
   {
      assert(ix - m_stencilStart[0] >= 0 && ix - m_stencilStart[0] < (int)m_cacheBlock->getSize()[0]);
      assert(iy - m_stencilStart[1] >= 0 && iy - m_stencilStart[1] < (int)m_cacheBlock->getSize()[1]);
      assert(iz - m_stencilStart[2] >= 0 && iz - m_stencilStart[2] < (int)m_cacheBlock->getSize()[2]);
      return m_cacheBlock->Access(ix - m_stencilStart[0], iy - m_stencilStart[1], iz - m_stencilStart[2]);
   }

   void release()
   {
      _release(m_cacheBlock);
      _release(m_CoarsenedBlock);
   }

 private:
   BlockLab(const BlockLab &) = delete;
   BlockLab &operator=(const BlockLab &) = delete;
};

} // namespace cubism
