#pragma once
#include "BlockInfo.h"
#include "BlockLab.h"
#include "Grid.h"
#include "Matrix3D.h"

#include <algorithm>
#include <cstring>
#include <omp.h>
#include <string>

namespace cubism
{

template <typename TGrid, typename TLab>
class MeshAdaptation
{
 public:
   typedef typename TGrid::BlockType BlockType;
   typedef typename TGrid::BlockType::ElementType ElementType;
 protected:
   TGrid *m_refGrid;
   int s[3];
   int e[3];
   bool istensorial{true};
   int Is[3];
   int Ie[3];
   std::vector<int> components;
   TLab *labs;
   double time;
   bool LabsPrepared;
   bool flag;
   bool CallValidStates;
   bool basic_refinement;
   double tolerance_for_refinement;
   double tolerance_for_compression;
 public:
   MeshAdaptation(TGrid &grid, const double Rtol, const double Ctol)
   {
      m_refGrid = &grid;

      for (int i = 0 ; i < ElementType::DIM ; i++) components.push_back(i);

      s[0] = -2;
      e[0] =  3;
      s[1] = -2;
      e[1] =  3;
      #if DIMENSION == 3
         s[2] = -2;
         e[2] =  3;
      #else
         s[2] = 0;
         e[2] = 1;
      #endif
      Is[0] = s[0]; Is[1] = s[1]; Is[2] = s[2];
      Ie[0] = e[0]; Ie[1] = e[1]; Ie[2] = e[2];
      istensorial = true;
      tolerance_for_refinement  = Rtol;
      tolerance_for_compression = Ctol;
      flag = true;
   }

   virtual ~MeshAdaptation() {}

   virtual void Tag(double t = 0)
   {
      time = t;

      vector<BlockInfo> & I = m_refGrid->getBlocksInfo();

      const int nthreads = omp_get_max_threads();

      labs = new TLab[nthreads];

      for (int i = 0; i < nthreads; i++)
         labs[i].prepare(*m_refGrid, s[0], e[0], s[1], e[1], s[2], e[2], istensorial, Is[0], Ie[0], Is[1], Ie[1], Is[2], Ie[2]);

      LabsPrepared = true;

      CallValidStates = false;

      #pragma omp parallel
      {
         const int tid = omp_get_thread_num();
         TLab &mylab = labs[tid];
         #pragma omp for
         for (size_t i = 0; i < I.size(); i++)
         {
            mylab.load(I[i], t);
            BlockInfo &info = m_refGrid->getBlockInfoAll(I[i].level, I[i].Z);
            I[i].state      = TagLoadedBlock(labs[tid], info);
            info.state      = I[i].state;
            #pragma omp critical
            {
               if (info.state != Leave) CallValidStates = true;
            }
         }
      }
      if (CallValidStates) ValidStates();
   }

   void TagLike(const std::vector<BlockInfo> & I1)
   {
      std::vector<BlockInfo> &I2 = m_refGrid->getBlocksInfo();
      for (size_t i1 = 0; i1 < I2.size(); i1++)
      {
         BlockInfo &ary0      = I2[i1];
         BlockInfo &info      = m_refGrid->getBlockInfoAll(ary0.level, ary0.Z);
         for (int i = 2 * (info.index[0] / 2); i <= 2 * (info.index[0] / 2) + 1; i++)
         for (int j = 2 * (info.index[1] / 2); j <= 2 * (info.index[1] / 2) + 1; j++)
         #if DIMENSION == 3
         for (int k = 2 * (info.index[2] / 2); k <= 2 * (info.index[2] / 2) + 1; k++)
         {
            const long long n = m_refGrid->getZforward(info.level, i, j, k);
            BlockInfo &infoNei = m_refGrid->getBlockInfoAll(info.level, n);
            infoNei.state = Leave;
         }
         #else
         {
            const long long n = m_refGrid->getZforward(info.level, i, j);
            BlockInfo &infoNei = m_refGrid->getBlockInfoAll(info.level, n);
            infoNei.state = Leave;
         }
         #endif
         info.state = Leave;
         ary0.state = Leave;
      }
      #pragma omp parallel for
      for (size_t i = 0 ; i < I1.size(); i++)
      {
         const BlockInfo & info1 = I1[i];
         BlockInfo & info2 = I2[i];
         BlockInfo & info3 = m_refGrid->getBlockInfoAll(info2.level, info2.Z);
         info2.state = info1.state;
         info3.state = info1.state;
         if (info2.state == Compress)
         {
            const int i2 = 2 * (info2.index[0] / 2);
            const int j2 = 2 * (info2.index[1] / 2);
            #if DIMENSION == 3
            const int k2 = 2 * (info2.index[2] / 2);
            const long long n = m_refGrid->getZforward(info2.level, i2, j2, k2);
            #else
            const long long n = m_refGrid->getZforward(info2.level, i2, j2);
            #endif
            BlockInfo &infoNei = m_refGrid->getBlockInfoAll(info2.level, n);
            infoNei.state = Compress;
         }
      }
      LabsPrepared = false;
   }

   virtual void Adapt(double t = 0, bool verbosity = false, bool basic = false)
   {
      basic_refinement = basic;
      if (LabsPrepared == false)
      {
         const int nthreads = omp_get_max_threads();
         labs = new TLab[nthreads];
         for (int i = 0; i < nthreads; i++)
            labs[i].prepare(*m_refGrid, s[0], e[0], s[1], e[1], s[2], e[2], true, Is[0], Ie[0], Is[1], Ie[1], Is[2], Ie[2]);
      }

      // Refinement/compression of blocks
      /*************************************************/
      int r = 0;
      int c = 0;

      std::vector<long long> n_com;
      std::vector<long long> n_ref;
      std::vector<int> m_com;
      std::vector<int> m_ref;

      std::vector<BlockInfo> &I = m_refGrid->getBlocksInfo();
      for (auto &i : I)
      {
         BlockInfo &info = m_refGrid->getBlockInfoAll(i.level, i.Z);
         if (info.state == Refine)
         {
            m_ref.push_back(info.level);
            n_ref.push_back(info.Z);
         }
         else if (info.state == Compress && info.index[0]%2 == 0
            && info.index[1]%2 == 0 && info.index[2]%2 == 0)
         {
            m_com.push_back(info.level);
            n_com.push_back(info.Z);
         }
      }
      //#pragma omp parallel
      {
         //#pragma omp for
         for (size_t i = 0; i < m_ref.size(); i++)
         {
            refine_1(m_ref[i], n_ref[i]);
            //#pragma omp atomic
            r++;
         }
         //#pragma omp for
         for (size_t i = 0; i < m_ref.size(); i++)
         {
            refine_2(m_ref[i], n_ref[i]);
         }
      }
      //#pragma omp parallel for
      for (size_t i = 0; i < m_com.size(); i++)
      {
         compress(m_com[i], n_com[i]);
         //#pragma omp atomic
         c++;
      }
      if (verbosity) std::cout << "Blocks refined:" << r << " blocks compressed:" << c << std::endl;
      m_refGrid->FillPos();
      delete[] labs;
      if (r>0 || c>0 || CallValidStates)
      {
         m_refGrid->UpdateGroups         = true;
         m_refGrid->UpdateFluxCorrection = true;
         flag                            = false;
      }
      LabsPrepared = false;
   }

 protected:
   void refine_1(const int level, const long long Z)
   {
      const int tid = omp_get_thread_num();

      BlockInfo &parent = m_refGrid->getBlockInfoAll(level, Z);
      parent.state      = Leave;
      if (basic_refinement == false)
         labs[tid].load(parent, time, true);

      const int p[3] = {parent.index[0], parent.index[1], parent.index[2]};

      assert(parent.ptrBlock != NULL);
      assert(level <= m_refGrid->getlevelMax() - 1);
      #if DIMENSION == 3
      BlockType *Blocks[8];
      for (int k = 0; k < 2; k++)
         for (int j = 0; j < 2; j++)
            for (int i = 0; i < 2; i++)
            {
               const long long nc = m_refGrid->getZforward(level + 1, 2 * p[0] + i, 2 * p[1] + j, 2 * p[2] + k);
               BlockInfo &Child   = m_refGrid->getBlockInfoAll(level + 1, nc);
               Child.state        = Leave;
               #pragma omp critical
               {
                  m_refGrid->_alloc(level + 1, nc);
                  m_refGrid->Tree(level + 1, nc).setCheckCoarser();
               }
               Blocks[k * 4 + j * 2 + i] = (BlockType *)Child.ptrBlock;
            }
      #else
      BlockType *Blocks[4];
      for (int j = 0; j < 2; j++)
         for (int i = 0; i < 2; i++)
         {
            const long long nc = m_refGrid->getZforward(level + 1, 2 * p[0] + i, 2 * p[1] + j);
            BlockInfo &Child = m_refGrid->getBlockInfoAll(level + 1, nc);
            Child.state = Leave;
            #pragma omp critical
            {
               m_refGrid->_alloc(level + 1, nc);
               m_refGrid->Tree(level + 1, nc).setCheckCoarser();
            }
            Blocks[j * 2 + i] = (BlockType *)Child.ptrBlock;
         }
      #endif
      if (basic_refinement == false)
         RefineBlocks(Blocks, parent);
   }

   void refine_2(const int level, const long long Z)
   {
      #pragma omp critical
      {
         m_refGrid->_dealloc(level, Z);
      }

      BlockInfo &parent = m_refGrid->getBlockInfoAll(level, Z);
      m_refGrid->Tree(parent).setCheckFiner();
      parent.state = Leave;

      int p[3] = {parent.index[0], parent.index[1], parent.index[2]};
      #if DIMENSION == 3
      for (int k = 0; k < 2; k++)
         for (int j = 0; j < 2; j++)
            for (int i = 0; i < 2; i++)
            {
               const long long nc = m_refGrid->getZforward(level + 1, 2 * p[0] + i, 2 * p[1] + j, 2 * p[2] + k);
               BlockInfo &Child   = m_refGrid->getBlockInfoAll(level + 1, nc);
               m_refGrid->Tree(Child).setrank(m_refGrid->rank());

               if (level + 2 < m_refGrid->getlevelMax())
                  for (int i0 = 0; i0 < 2; i0++)
                     for (int i1 = 0; i1 < 2; i1++)
                        for (int i2 = 0; i2 < 2; i2++)
                           m_refGrid->Tree(level + 2, Child.Zchild[i0][i1][i2]).setCheckCoarser();
            }
      #endif
      #if DIMENSION == 2
      for (int j = 0; j < 2; j++)
         for (int i = 0; i < 2; i++)
         {
            const long long nc = m_refGrid->getZforward(level + 1, 2 * p[0] + i, 2 * p[1] + j);
            BlockInfo &Child   = m_refGrid->getBlockInfoAll(level + 1, nc);
            m_refGrid->Tree(Child).setrank(m_refGrid->rank());

            if (level + 2 < m_refGrid->getlevelMax())
               for (int i0 = 0; i0 < 2; i0++)
                  for (int i1 = 0; i1 < 2; i1++) m_refGrid->Tree(level + 2, Child.Zchild[i0][i1][1]).setCheckCoarser();
         }
      #endif
   }

   void compress(const int level, const long long Z)
   {
      assert(level > 0);

      BlockInfo &info = m_refGrid->getBlockInfoAll(level, Z);

      assert(info.state == Compress);

      #if DIMENSION == 3
      BlockType *Blocks[8];
      for (int K = 0; K < 2; K++)
         for (int J = 0; J < 2; J++)
            for (int I = 0; I < 2; I++)
            {
               const int blk = K * 4 + J * 2 + I;
               const long long n =
                   m_refGrid->getZforward(level, info.index[0] + I, info.index[1] + J, info.index[2] + K);
               Blocks[blk] = (BlockType *)(m_refGrid->getBlockInfoAll(level, n)).ptrBlock;
            }

      const int nx         = BlockType::sizeX;
      const int ny         = BlockType::sizeY;
      const int nz         = BlockType::sizeZ;
      const int offsetX[2] = {0, nx / 2};
      const int offsetY[2] = {0, ny / 2};
      const int offsetZ[2] = {0, nz / 2};
      if (basic_refinement == false)
      for (int K = 0; K < 2; K++)
         for (int J = 0; J < 2; J++)
            for (int I = 0; I < 2; I++)
            {
               BlockType &b = *Blocks[K * 4 + J * 2 + I];
               for (int k = 0; k < nz; k += 2)
                  for (int j = 0; j < ny; j += 2)
                     for (int i = 0; i < nx; i += 2)
                     {
                        ElementType average =
                            0.125 * ( (b(i    , j    , k  ) + b(i + 1, j + 1, k + 1)) 
                                    + (b(i + 1, j    , k  ) + b(i    , j + 1, k + 1))
                                    + (b(i    , j + 1, k  ) + b(i + 1, j    , k + 1))
                                    + (b(i + 1, j + 1, k  ) + b(i    , j    , k + 1)) );
                        (*Blocks[0])(i / 2 + offsetX[I], j / 2 + offsetY[J], k / 2 + offsetZ[K]) = average;
                     }
            }

      const long long np = m_refGrid->getZforward(level - 1, info.index[0] / 2, info.index[1] / 2, info.index[2] / 2);
      BlockInfo &parent  = m_refGrid->getBlockInfoAll(level - 1, np);
      m_refGrid->Tree(parent.level, parent.Z).setrank(m_refGrid->rank());
      parent.ptrBlock    = info.ptrBlock;
      parent.state       = Leave;
      if (level - 2 >= 0) m_refGrid->Tree(level - 2, parent.Zparent).setCheckFiner();

      #pragma omp critical
      {
         for (int K = 0; K < 2; K++)
            for (int J = 0; J < 2; J++)
               for (int I = 0; I < 2; I++)
               {
                  const long long n =
                      m_refGrid->getZforward(level, info.index[0] + I, info.index[1] + J, info.index[2] + K);
                  if (I + J + K == 0)
                  {
                     m_refGrid->FindBlockInfo(level, n, level - 1, np);
                  }
                  else
                  {
                     m_refGrid->_dealloc(level, n);
                  }
                  m_refGrid->Tree(level, n).setCheckCoarser();
                  m_refGrid->getBlockInfoAll(level, n).state = Leave;
               }
      }
      #endif
      #if DIMENSION == 2
      BlockType *Blocks[4];
      for (int J = 0; J < 2; J++)
         for (int I = 0; I < 2; I++)
         {
            const int blk     = J * 2 + I;
            const long long n = m_refGrid->getZforward(level, info.index[0] + I, info.index[1] + J);
            Blocks[blk]       = (BlockType *)(m_refGrid->getBlockInfoAll(level, n)).ptrBlock;
         }

      const int nx         = BlockType::sizeX;
      const int ny         = BlockType::sizeY;
      const int offsetX[2] = {0, nx / 2};
      const int offsetY[2] = {0, ny / 2};
      if (basic_refinement == false)
      for (int J = 0; J < 2; J++)
         for (int I = 0; I < 2; I++)
         {
            BlockType &b = *Blocks[J * 2 + I];
            for (int j = 0; j < ny; j += 2)
               for (int i = 0; i < nx; i += 2)
               {
                  ElementType average = 0.25 * ( (b(i, j, 0)+ b(i + 1, j + 1, 0)) + (b(i + 1, j, 0) + b(i, j + 1, 0)) );
                  (*Blocks[0])(i / 2 + offsetX[I], j / 2 + offsetY[J], 0) = average;
               }
         }
      const long long np = m_refGrid->getZforward(level - 1, info.index[0] / 2, info.index[1] / 2);
      BlockInfo &parent  = m_refGrid->getBlockInfoAll(level - 1, np);
      m_refGrid->Tree(parent.level, parent.Z).setrank(m_refGrid->rank());
      parent.ptrBlock    = info.ptrBlock;
      parent.state       = Leave;
      if (level - 2 >= 0) m_refGrid->Tree(level - 2, parent.Zparent).setCheckFiner();

      #pragma omp critical
      {
         for (int J = 0; J < 2; J++)
            for (int I = 0; I < 2; I++)
            {
               const long long n = m_refGrid->getZforward(level, info.index[0] + I, info.index[1] + J);
               if (I + J == 0)
               {
                  m_refGrid->FindBlockInfo(level, n, level - 1, np);
               }
               else
               {
                  m_refGrid->_dealloc(level, n);
               }
               m_refGrid->Tree(level, n).setCheckCoarser();
               m_refGrid->getBlockInfoAll(level, n).state = Leave;
            }
      }
      #endif
   }

   virtual void ValidStates()
   {
      const std::array<int, 3> blocksPerDim = m_refGrid->getMaxBlocks();
      const int levelMin              = 0;
      const int levelMax              = m_refGrid->getlevelMax();
      const bool xperiodic            = m_refGrid->xperiodic;
      const bool yperiodic            = m_refGrid->yperiodic;
      const bool zperiodic            = m_refGrid->zperiodic;

      std::vector<BlockInfo> &I = m_refGrid->getBlocksInfo();

      #pragma omp parallel for
      for (size_t j = 0; j < I.size(); j++)
      {
         BlockInfo &info = I[j];

         if ((info.state == Refine && info.level == levelMax - 1) || (info.state == Compress && info.level == levelMin))
         {
            info.state                                             = Leave;
            (m_refGrid->getBlockInfoAll(info.level, info.Z)).state = Leave;
         }
         if (info.state != Leave)
         {
            info.changed2                                             = true;
            (m_refGrid->getBlockInfoAll(info.level, info.Z)).changed2 = info.changed2;
         }
      }

      // 1.Change states of blocks next to finer resolution blocks
      // 2.Change states of blocks next to same resolution blocks
      // 3.Compress a block only if all blocks with the same parent need compression
      bool clean_boundary = true;
      for (int m = levelMax - 1; m >= levelMin; m--)
      {
         // 1.
         for (size_t j = 0; j < I.size(); j++)
         {
            BlockInfo &info = I[j];
            if (info.level == m && info.state != Refine && info.level != levelMax - 1)
            {
               const int TwoPower = 1 << info.level;
               const bool xskin   = info.index[0] == 0 || info.index[0] == blocksPerDim[0] * TwoPower - 1;
               const bool yskin   = info.index[1] == 0 || info.index[1] == blocksPerDim[1] * TwoPower - 1;
               const bool zskin   = info.index[2] == 0 || info.index[2] == blocksPerDim[2] * TwoPower - 1;
               const int xskip    = info.index[0] == 0 ? -1 : 1;
               const int yskip    = info.index[1] == 0 ? -1 : 1;
               const int zskip    = info.index[2] == 0 ? -1 : 1;

               for (int icode = 0; icode < 27; icode++)
               {
                  if (info.state == Refine) break;
                  if (icode == 1 * 1 + 3 * 1 + 9 * 1) continue;
                  const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, (icode / 9) % 3 - 1};
                  if (!xperiodic && code[0] == xskip && xskin) continue;
                  if (!yperiodic && code[1] == yskip && yskin) continue;
                  if (!zperiodic && code[2] == zskip && zskin) continue;
                  #if DIMENSION == 2
                  if (code[2] != 0) continue;
                  #endif


                  BlockInfo &infoNei = m_refGrid->getBlockInfoAll(info.level, info.Znei_(code[0], code[1], code[2]));
                  if (m_refGrid->Tree(infoNei).CheckFiner())
                  {
                     if (info.state == Compress)
                     {
                        info.state                                             = Leave;
                        (m_refGrid->getBlockInfoAll(info.level, info.Z)).state = Leave;
                     }
                     // if (info.level == levelMax - 1) break;

                     const int tmp = abs(code[0]) + abs(code[1]) + abs(code[2]); 
                     int Bstep = 1;// face
                     if (tmp == 2) Bstep = 3; //edge
                     else if (tmp == 3) Bstep = 4; //corner                                                    

                     //loop over blocks that make up face/edge/corner(respectively 4,2 or 1 blocks)
                     #if DIMENSION == 3
                     for (int B = 0; B <= 3; B += Bstep)
                     #else
                     for (int B = 0; B <= 1; B += Bstep)
                     #endif
                     {
                        const int aux = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
                        const int iNei = 2 * info.index[0] + max(code[0], 0) + code[0] + (B % 2) * max(0, 1 - abs(code[0]));
                        const int jNei = 2 * info.index[1] + max(code[1], 0) + code[1] + aux * max(0, 1 - abs(code[1]));
                        #if DIMENSION == 3
                           const int kNei = 2 * info.index[2] + max(code[2], 0) + code[2] + (B / 2) * max(0, 1 - abs(code[2]));
                           const long long zzz = m_refGrid->getZforward(m + 1, iNei, jNei, kNei);
                        #else
                           const long long zzz = m_refGrid->getZforward(m + 1, iNei, jNei);
                        #endif
                        BlockInfo &FinerNei = m_refGrid->getBlockInfoAll(m + 1, zzz);
                        State NeiState      = FinerNei.state;
                        if (NeiState == Refine)
                        {
                           info.state                                                = Refine;
                           (m_refGrid->getBlockInfoAll(info.level, info.Z)).state    = Refine;
                           info.changed2                                             = true;
                           (m_refGrid->getBlockInfoAll(info.level, info.Z)).changed2 = true;
                           break;
                        }
                     }
                  }
               }
            }
         }

         m_refGrid->UpdateBoundary(clean_boundary);
         clean_boundary = false;
         if (m == levelMin) break;

         // 2.
         for (size_t j = 0; j < I.size(); j++)
         {
            BlockInfo &info = I[j];
            if (info.level == m && info.state == Compress)
            {
               const int aux    = 1 << info.level;
               const bool xskin = info.index[0] == 0 || info.index[0] == blocksPerDim[0] * aux - 1;
               const bool yskin = info.index[1] == 0 || info.index[1] == blocksPerDim[1] * aux - 1;
               const bool zskin = info.index[2] == 0 || info.index[2] == blocksPerDim[2] * aux - 1;
               const int xskip  = info.index[0] == 0 ? -1 : 1;
               const int yskip  = info.index[1] == 0 ? -1 : 1;
               const int zskip  = info.index[2] == 0 ? -1 : 1;

               for (int icode = 0; icode < 27; icode++)
               {
                  if (icode == 1 * 1 + 3 * 1 + 9 * 1) continue;
                  const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, (icode / 9) % 3 - 1};
                  if (!xperiodic && code[0] == xskip && xskin) continue;
                  if (!yperiodic && code[1] == yskip && yskin) continue;
                  if (!zperiodic && code[2] == zskip && zskin) continue;
                  #if DIMENSION == 2
                  if (code[2] != 0) continue;
                  #endif

                  BlockInfo &infoNei = m_refGrid->getBlockInfoAll(info.level, info.Znei_(code[0], code[1], code[2]));
                  if (m_refGrid->Tree(infoNei).Exists() && infoNei.state == Refine)
                  {
                     info.state                                             = Leave;
                     (m_refGrid->getBlockInfoAll(info.level, info.Z)).state = Leave;
                     break;
                  }
               }
            }
         }
      } // m

      // 3.
      for (size_t jjj = 0; jjj < I.size(); jjj++)
      {
         BlockInfo &info = I[jjj];
         const int m     = info.level;
         bool found      = false;
         for (int i = 2 * (info.index[0] / 2); i <= 2 * (info.index[0] / 2) + 1; i++)
         for (int j = 2 * (info.index[1] / 2); j <= 2 * (info.index[1] / 2) + 1; j++)
         for (int k = 2 * (info.index[2] / 2); k <= 2 * (info.index[2] / 2) + 1; k++)
         {
            #if DIMENSION == 3
               const long long n = m_refGrid->getZforward(m, i, j, k);
            #else
               const long long n = m_refGrid->getZforward(m, i, j);
            #endif
            BlockInfo &infoNei = m_refGrid->getBlockInfoAll(m, n);
            if (m_refGrid->Tree(infoNei).Exists() == false || infoNei.state != Compress)
            {
               found = true;
               if (info.state == Compress)
               {
                  info.state                                             = Leave;
                  (m_refGrid->getBlockInfoAll(info.level, info.Z)).state = Leave;
               }
               break;
            }
         }
         if (found)
            for (int i = 2 * (info.index[0] / 2); i <= 2 * (info.index[0] / 2) + 1; i++)
            for (int j = 2 * (info.index[1] / 2); j <= 2 * (info.index[1] / 2) + 1; j++)
            for (int k = 2 * (info.index[2] / 2); k <= 2 * (info.index[2] / 2) + 1; k++)
            {
               #if DIMENSION == 3
                  const long long n = m_refGrid->getZforward(m, i, j, k);
               #else
                  const long long n = m_refGrid->getZforward(m, i, j);
               #endif
               BlockInfo &infoNei = m_refGrid->getBlockInfoAll(m, n);
               if (m_refGrid->Tree(infoNei).Exists() && infoNei.state == Compress) infoNei.state = Leave;
            }
      }
   }
   ////////////////////////////////////////////////////////////////////////////////////////////////
   // Virtual functions that can be overwritten by user
   ////////////////////////////////////////////////////////////////////////////////////////////////

   virtual void RefineBlocks(BlockType *B[8], BlockInfo parent)
   {
      int tid      = omp_get_thread_num();
      const int nx = BlockType::sizeX;
      const int ny = BlockType::sizeY;

      int offsetX[2] = {0, nx / 2};
      int offsetY[2] = {0, ny / 2};

      TLab &Lab = labs[tid];

      #if DIMENSION == 3
      const int nz   = BlockType::sizeZ;
      int offsetZ[2] = {0, nz / 2};

      for (int K = 0; K < 2; K++)
         for (int J = 0; J < 2; J++)
            for (int I = 0; I < 2; I++)
            {
               BlockType &b = *B[K * 4 + J * 2 + I];
               b.clear();

               for (int k = 0; k < nz; k += 2)
                  for (int j = 0; j < ny; j += 2)
                     for (int i = 0; i < nx; i += 2)
                     {
                        const int x = i / 2 + offsetX[I];
                        const int y = j / 2 + offsetY[J];
                        const int z = k / 2 + offsetZ[K];
                        ElementType dudx  = 0.5 * (Lab(x+1,y,z) - Lab(x-1,y,z));
                        ElementType dudy  = 0.5 * (Lab(x,y+1,z) - Lab(x,y-1,z));
                        ElementType dudz  = 0.5 * (Lab(x,y,z+1) - Lab(x,y,z-1));
                        ElementType dudx2 = (Lab(x+1,y,z)  + Lab(x-1,y,z)) - 2.0*Lab(x,y,z);
                        ElementType dudy2 = (Lab(x,y+1,z)  + Lab(x,y-1,z)) - 2.0*Lab(x,y,z);
                        ElementType dudz2 = (Lab(x,y,z+1)  + Lab(x,y,z-1)) - 2.0*Lab(x,y,z);
                        ElementType dudxdy = 0.25*((Lab(x+1,y+1,z)+Lab(x-1,y-1,z)) - (Lab(x+1,y-1,z)+Lab(x-1,y+1,z)));
                        ElementType dudxdz = 0.25*((Lab(x+1,y,z+1)+Lab(x-1,y,z-1)) - (Lab(x+1,y,z-1)+Lab(x-1,y,z+1)));
                        ElementType dudydz = 0.25*((Lab(x,y+1,z+1)+Lab(x,y-1,z-1)) - (Lab(x,y+1,z-1)+Lab(x,y-1,z+1)));

                        b(i  , j  , k  ) = Lab(x,y,z) + 0.25*(-(1.0)* dudx - dudy - dudz) + 0.03125 *(dudx2+dudy2+dudz2) + 0.0625*(       dudxdy + dudxdz + dudydz);
                        b(i+1, j  , k  ) = Lab(x,y,z) + 0.25*(        dudx - dudy - dudz) + 0.03125 *(dudx2+dudy2+dudz2) + 0.0625*(-(1.0)*dudxdy - dudxdz + dudydz);
                        b(i  , j+1, k  ) = Lab(x,y,z) + 0.25*(-(1.0)* dudx + dudy - dudz) + 0.03125 *(dudx2+dudy2+dudz2) + 0.0625*(-(1.0)*dudxdy + dudxdz - dudydz);
                        b(i+1, j+1, k  ) = Lab(x,y,z) + 0.25*(        dudx + dudy - dudz) + 0.03125 *(dudx2+dudy2+dudz2) + 0.0625*(       dudxdy - dudxdz - dudydz);
                        b(i  , j  , k+1) = Lab(x,y,z) + 0.25*(-(1.0)* dudx - dudy + dudz) + 0.03125 *(dudx2+dudy2+dudz2) + 0.0625*(       dudxdy - dudxdz - dudydz);
                        b(i+1, j  , k+1) = Lab(x,y,z) + 0.25*(        dudx - dudy + dudz) + 0.03125 *(dudx2+dudy2+dudz2) + 0.0625*(-(1.0)*dudxdy + dudxdz - dudydz);
                        b(i  , j+1, k+1) = Lab(x,y,z) + 0.25*(-(1.0)* dudx + dudy + dudz) + 0.03125 *(dudx2+dudy2+dudz2) + 0.0625*(-(1.0)*dudxdy - dudxdz + dudydz);
                        b(i+1, j+1, k+1) = Lab(x,y,z) + 0.25*(        dudx + dudy + dudz) + 0.03125 *(dudx2+dudy2+dudz2) + 0.0625*(       dudxdy + dudxdz + dudydz);
                     }
            }
      #else

      for (int J = 0; J < 2; J++)
         for (int I = 0; I < 2; I++)
         {
            BlockType &b = *B[J * 2 + I];
            b.clear();

            for (int j = 0; j < ny; j += 2)
               for (int i = 0; i < nx; i += 2)
               {
                  ElementType dudx = 0.5 * (Lab(i / 2 + offsetX[I] + 1, j / 2 + offsetY[J]) -
                                            Lab(i / 2 + offsetX[I] - 1, j / 2 + offsetY[J]));
                  ElementType dudy = 0.5 * (Lab(i / 2 + offsetX[I], j / 2 + offsetY[J] + 1) -
                                            Lab(i / 2 + offsetX[I], j / 2 + offsetY[J] - 1));

                  ElementType dudx2 = (Lab(i/2+offsetX[I]+1, j/2+offsetY[J]  )  + Lab(i/2+offsetX[I]-1,j/2+offsetY[J]  ))- 2.0*Lab(i/2+offsetX[I], j/2+offsetY[J]);
                  ElementType dudy2 = (Lab(i/2+offsetX[I]  , j/2+offsetY[J]+1)  + Lab(i/2+offsetX[I]  ,j/2+offsetY[J]-1))- 2.0*Lab(i/2+offsetX[I], j/2+offsetY[J]);  

                  ElementType dudxdy = 0.25*( (Lab(i/2+offsetX[I]+1, j/2+offsetY[J]+1)+Lab(i/2+offsetX[I]-1, j/2+offsetY[J]-1))
                                             -(Lab(i/2+offsetX[I]+1, j/2+offsetY[J]-1)+Lab(i/2+offsetX[I]-1, j/2+offsetY[J]+1)));

                  b(i  ,j  ,0) = (Lab(i/2+offsetX[I], j/2+offsetY[J]) + (- 0.25*dudx - 0.25*dudy) ) + ( (0.03125*dudx2 + 0.03125*dudy2) + 0.0625*dudxdy);
                  b(i+1,j  ,0) = (Lab(i/2+offsetX[I], j/2+offsetY[J]) + (+ 0.25*dudx - 0.25*dudy) ) + ( (0.03125*dudx2 + 0.03125*dudy2) - 0.0625*dudxdy);
                  b(i  ,j+1,0) = (Lab(i/2+offsetX[I], j/2+offsetY[J]) + (- 0.25*dudx + 0.25*dudy) ) + ( (0.03125*dudx2 + 0.03125*dudy2) - 0.0625*dudxdy);
                  b(i+1,j+1,0) = (Lab(i/2+offsetX[I], j/2+offsetY[J]) + (+ 0.25*dudx + 0.25*dudy) ) + ( (0.03125*dudx2 + 0.03125*dudy2) + 0.0625*dudxdy);
               }
         }
      #endif
   }

   virtual State TagLoadedBlock(TLab &Lab_, BlockInfo &info)
   {
      const int nx = BlockType::sizeX;
      const int ny = BlockType::sizeY;

      double Linf = 0.0;
      #if DIMENSION == 3
      const int nz = BlockType::sizeZ;
      for (int k = 0; k < nz; k++)
         for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++)
            {
               double s0 = std::fabs(Lab_(i, j, k).magnitude());
               Linf      = max(Linf, s0);
            }
      #endif
      #if DIMENSION == 2
      for (int j = 0; j < ny; j++)
         for (int i = 0; i < nx; i++)
         {
            double s0 = std::fabs(Lab_(i, j).magnitude());
            Linf      = max(Linf, s0);
         }
      #endif

      if (Linf > tolerance_for_refinement) return Refine;
      else if (Linf < tolerance_for_compression)
         return Compress;

      return Leave;
   }
};
} // namespace cubism
