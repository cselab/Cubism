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

#define WENOWAVELET 3

template <typename TGrid, typename otherTGRID = TGrid>
class MeshAdaptation_basic
{
 public:
   typedef typename TGrid::BlockType BlockType;

 protected:
   TGrid *m_refGrid;
   bool flag;

 public:
   MeshAdaptation_basic(TGrid &grid)
   {
      m_refGrid = &grid;
      flag      = true;
   }

   virtual ~MeshAdaptation_basic() {}

   virtual void AdaptLikeOther(otherTGRID &OtherGrid)
   {
      otherTGRID *m_OtherGrid = &OtherGrid;

      std::vector<BlockInfo> &I = m_refGrid->getBlocksInfo();

      const int Ninner = I.size();
      #pragma omp parallel for
      for (int i = 0; i < Ninner; i++)
      {
         BlockInfo &ary0      = I[i];
         BlockInfo &info      = m_refGrid->getBlockInfoAll(ary0.level, ary0.Z);
         BlockInfo &infoOther = m_OtherGrid->getBlockInfoAll(ary0.level, ary0.Z);
         if (m_OtherGrid->Tree(infoOther).Exists()) ary0.state = Leave;
         else if (m_OtherGrid->Tree(infoOther).CheckFiner())
            ary0.state = Refine;
         else if (m_OtherGrid->Tree(infoOther).CheckCoarser())
            ary0.state = Compress;
         info.state = ary0.state;
      }

      ValidStates();

      // Refinement/compression of blocks
      /*************************************************/
      int r = 0;
      int c = 0;

      std::vector<int> m_com;
      std::vector<int> m_ref;
      std::vector<long long> n_com;
      std::vector<long long> n_ref;

      for (auto &info : I)
      {
         if (info.state == Refine)
         {
            m_ref.push_back(info.level);
            n_ref.push_back(info.Z);
         }
         else if (info.state == Compress)
         {
            m_com.push_back(info.level);
            n_com.push_back(info.Z);
         }
      }
      #pragma omp parallel
      {
         #pragma omp for
         for (size_t i = 0; i < m_ref.size(); i++)
         {
            refine_1(m_ref[i], n_ref[i]);
            #pragma omp atomic
            r++;
         }
         #pragma omp for
         for (size_t i = 0; i < m_ref.size(); i++)
         {
            refine_2(m_ref[i], n_ref[i]);
         }
      }

      #pragma omp parallel for
      for (size_t i = 0; i < m_com.size(); i++)
      {
         compress(m_com[i], n_com[i]);
         #pragma omp atomic
         c++;
      }

      m_refGrid->FillPos();
      if (r > 0 || c > 0)
      {
         m_refGrid->UpdateFluxCorrection = flag;
         flag                            = false;
         m_refGrid->UpdateGroups         = true;
      }
   }

 protected:
   virtual void refine_1(const int level, const long long Z)
   {
      BlockInfo &parent = m_refGrid->getBlockInfoAll(level, Z);
      parent.state      = Leave;

      const int p[3] = {parent.index[0], parent.index[1], parent.index[2]};

      assert(parent.ptrBlock != NULL);
      assert(level <= m_refGrid->getlevelMax() - 1);

#if DIMENSION == 3
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
            }
#endif
#if DIMENSION == 2
      for (int j = 0; j < 2; j++)
         for (int i = 0; i < 2; i++)
         {
            const long long nc = m_refGrid->getZforward(level + 1, 2 * p[0] + i, 2 * p[1] + j);
            BlockInfo &Child   = m_refGrid->getBlockInfoAll(level + 1, nc);
            Child.state        = Leave;
            #pragma omp critical
            {
               m_refGrid->_alloc(level + 1, nc);
               m_refGrid->Tree(level + 1, nc).setCheckCoarser();
            }
         }
#endif
   }

   virtual void refine_2(const int level, const long long Z)
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
                  for (int i1 = 0; i1 < 2; i1++) m_refGrid->Tree(level + 2, Child.Zchild[i0][i1]).setCheckCoarser();
         }
#endif
   }

   virtual void compress(const int level, const long long Z)
   {
      assert(level > 0);

      BlockInfo &info = m_refGrid->getBlockInfoAll(level, Z);

      assert(info.state == Compress);

#if DIMENSION == 3
      const long long np = m_refGrid->getZforward(level - 1, info.index[0] / 2, info.index[1] / 2, info.index[2] / 2);
      BlockInfo &parent  = m_refGrid->getBlockInfoAll(level - 1, np);
      parent.ptrBlock    = info.ptrBlock;
      m_refGrid->Tree(parent).setrank(m_refGrid->rank());
      parent.h_gridpoint = parent.h;
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
                  m_refGrid->getBlockInfoAll(level, n).state = Leave;
                  m_refGrid->Tree(level, n).setCheckCoarser();
               }
      }
#endif
#if DIMENSION == 2

      const long long np = m_refGrid->getZforward(level - 1, info.index[0] / 2, info.index[1] / 2);
      BlockInfo &parent  = m_refGrid->getBlockInfoAll(level - 1, np);
      parent.ptrBlock    = info.ptrBlock;
      m_refGrid->Tree(parent).setrank(m_refGrid->rank());
      parent.h_gridpoint = parent.h;
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
               m_refGrid->getBlockInfoAll(level, n).state = Leave;
               m_refGrid->Tree(level, n).setCheckCoarser();
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
                  if (icode == 1 * 1 + 3 * 1 + 9 * 1) continue;
                  const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, (icode / 9) % 3 - 1};
                  if (!xperiodic && code[0] == xskip && xskin) continue;
                  if (!yperiodic && code[1] == yskip && yskin) continue;
                  if (!zperiodic && code[2] == zskip && zskin) continue;

                  BlockInfo &infoNei = m_refGrid->getBlockInfoAll(info.level, info.Znei_(code[0], code[1], code[2]));

                  if (m_refGrid->Tree(infoNei).CheckFiner())
                  {
                     if (info.state == Compress)
                     {
                        info.state                                             = Leave;
                        (m_refGrid->getBlockInfoAll(info.level, info.Z)).state = Leave;
                     }
                     // if (info.level == levelMax - 1) break;

                     int Bstep = 1;                                                    // face
                     if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 2)) Bstep = 3; // edge
                     else if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 3))
                        Bstep = 4; // corner

                     for (int B = 0; B <= 3; B += Bstep) // loop over blocks that make up face/edge/corner
                                                         // (respectively 4,2 or 1 blocks)
                     {
                        const int aux = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
                        int iNei = 2 * info.index[0] + max(code[0], 0) + code[0] + (B % 2) * max(0, 1 - abs(code[0]));
                        int jNei = 2 * info.index[1] + max(code[1], 0) + code[1] + aux * max(0, 1 - abs(code[1]));
#if DIMENSION == 3
                        int kNei = 2 * info.index[2] + max(code[2], 0) + code[2] + (B / 2) * max(0, 1 - abs(code[2]));
                        long long zzz = m_refGrid->getZforward(m + 1, iNei, jNei, kNei);
#else
                        long long zzz = m_refGrid->getZforward(m + 1, iNei, jNei);
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

         if (m == levelMin) break;
         // 2.
         for (size_t j = 0; j < I.size(); j++)
         {
            BlockInfo &info = I[j];
            if (info.level == m && info.state == Compress)
            {
               int aux          = 1 << info.level;
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
                  // if (k!=0) {std::cout << "k!=0\n"; abort();}
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
                     // if (k!=0) {std::cout << "k!=0\n"; abort();}
                     const long long n = m_refGrid->getZforward(m, i, j);
#endif
                     BlockInfo &infoNei = m_refGrid->getBlockInfoAll(m, n);
                     if (m_refGrid->Tree(infoNei).Exists() && infoNei.state == Compress)
                     {
                        infoNei.state = Leave;
                     }
                  }
      }

      // 4.
      for (size_t jjj = 0; jjj < I.size(); jjj++)
      {
         BlockInfo &info = I[jjj];
         if (info.state == Compress)
         {
            int m      = info.level;
            bool first = true;
            for (int i = 2 * (info.index[0] / 2); i <= 2 * (info.index[0] / 2) + 1; i++)
               for (int j = 2 * (info.index[1] / 2); j <= 2 * (info.index[1] / 2) + 1; j++)
#if DIMENSION == 3
                  for (int k = 2 * (info.index[2] / 2); k <= 2 * (info.index[2] / 2) + 1; k++)
                  {
                     const long long n = m_refGrid->getZforward(m, i, j, k);
#else
               {
                  const long long n = m_refGrid->getZforward(m, i, j);
#endif
                     BlockInfo &infoNei = m_refGrid->getBlockInfoAll(m, n);
                     if (!first)
                     {
                        infoNei.state = Leave;
                     }
                     first = false;
                  }
            if (info.index[0] % 2 == 1 || info.index[1] % 2 == 1 || info.index[2] % 2 == 1)
            {
               info.state                                             = Leave;
               (m_refGrid->getBlockInfoAll(info.level, info.Z)).state = Leave;
            }
         }
      }
   }
};

template <typename TGrid, typename TLab, typename otherTGRID = TGrid>
class MeshAdaptation : public MeshAdaptation_basic<TGrid, otherTGRID>
{

 public:
   typedef typename TGrid::BlockType BlockType;
   typedef typename TGrid::BlockType::ElementType ElementType;

   double tolerance_for_refinement;
   double tolerance_for_compression;

 protected:
   TGrid *m_refGrid;
   int s[3];
   int e[3];
   bool istensorial;
   int Is[3];
   int Ie[3];
   std::vector<int> components;
   TLab *labs;
   double time;

   bool flag;
   bool verbose;

 public:
   MeshAdaptation(TGrid &grid, double Rtol, double Ctol, bool _verbose = false)
       : MeshAdaptation_basic<TGrid, otherTGRID>(grid)
   {
      bool tensorial = true;
      verbose        = _verbose;

      const int Gx = (WENOWAVELET == 3) ? 1 : 2;
      const int Gy = (WENOWAVELET == 3) ? 1 : 2;
#if DIMENSION == 3
      const int Gz = (WENOWAVELET == 3) ? 1 : 2;
#else
      const int Gz = 0;
#endif
      components.push_back(0);
      components.push_back(1);
      components.push_back(2);
      components.push_back(3);
      components.push_back(4);
      components.push_back(5);
      components.push_back(6);
      components.push_back(7);

      StencilInfo stencil(-Gx, -Gy, -Gz, Gx + 1, Gy + 1, Gz + 1, tensorial, components);

      m_refGrid = &grid;

      s[0]        = stencil.sx;
      e[0]        = stencil.ex;
      s[1]        = stencil.sy;
      e[1]        = stencil.ey;
      s[2]        = stencil.sz;
      e[2]        = stencil.ez;
      istensorial = stencil.tensorial;

      Is[0] = stencil.sx;
      Ie[0] = stencil.ex;
      Is[1] = stencil.sy;
      Ie[1] = stencil.ey;
      Is[2] = stencil.sz;
      Ie[2] = stencil.ez;

      tolerance_for_refinement  = Rtol;
      tolerance_for_compression = Ctol;

      flag = true;
   }

   virtual ~MeshAdaptation() {}

   virtual void AdaptTheMesh(double t = 0)
   {
      time = t;

      vector<BlockInfo> &avail0 = m_refGrid->getBlocksInfo();

      const int nthreads = omp_get_max_threads();

      labs = new TLab[nthreads];

      for (int i = 0; i < nthreads; i++)
         labs[i].prepare(*m_refGrid, s[0], e[0], s[1], e[1], s[2], e[2], true, Is[0], Ie[0], Is[1], Ie[1], Is[2],
                         Ie[2]);

      bool CallValidStates = false;

      const int Ninner = avail0.size();
      #pragma omp parallel num_threads(nthreads)
      {
         int tid     = omp_get_thread_num();
         TLab &mylab = labs[tid];

         #pragma omp for
         for (int i = 0; i < Ninner; i++)
         {
            BlockInfo &ary0 = avail0[i];
            mylab.load(ary0, t);
            BlockInfo &info = m_refGrid->getBlockInfoAll(ary0.level, ary0.Z);
            ary0.state      = TagLoadedBlock(labs[tid], info);
            info.state      = ary0.state;
            #pragma omp critical
            {
               if (info.state != Leave) CallValidStates = true;
            }
         }
      }
      if (CallValidStates) MeshAdaptation_basic<TGrid, otherTGRID>::ValidStates();
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
         else if (info.state == Compress)
         {
            m_com.push_back(info.level);
            n_com.push_back(info.Z);
         }
      }
      #pragma omp parallel
      {
         #pragma omp for
         for (size_t i = 0; i < m_ref.size(); i++)
         {
            refine_1(m_ref[i], n_ref[i]);
            #pragma omp atomic
            r++;
         }
         #pragma omp for
         for (size_t i = 0; i < m_ref.size(); i++)
         {
            refine_2(m_ref[i], n_ref[i]);
         }
      }
      #pragma omp parallel for
      for (size_t i = 0; i < m_com.size(); i++)
      {
         compress(m_com[i], n_com[i]);
         #pragma omp atomic
         c++;
      }
      int result[2] = {r, c};
      if (verbose) std::cout << "Blocks refined:" << result[0] << " blocks compressed:" << result[1] << std::endl;
      m_refGrid->FillPos();
      delete[] labs;
      if (!CallValidStates)
      {
         m_refGrid->UpdateGroups         = true;
         m_refGrid->UpdateFluxCorrection = flag;
         flag                            = false;
      }
      /*************************************************/
   }

   virtual void AdaptLikeOther(otherTGRID &OtherGrid) override
   {
      const int nthreads = omp_get_max_threads();
      labs               = new TLab[nthreads];
      for (int i = 0; i < nthreads; i++)
         labs[i].prepare(*m_refGrid, s[0], e[0], s[1], e[1], s[2], e[2], true, Is[0], Ie[0], Is[1], Ie[1], Is[2],
                         Ie[2]);
      MeshAdaptation_basic<TGrid, otherTGRID>::AdaptLikeOther(OtherGrid);
      delete[] labs;
   }

 protected:
   virtual void refine_1(const int level, const long long Z) override
   {
      int tid = omp_get_thread_num();

      BlockInfo &parent = m_refGrid->getBlockInfoAll(level, Z);
      parent.state      = Leave;
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
      RefineBlocks(Blocks, parent);
   }

   virtual void refine_2(const int level, const long long Z) override
   {
      MeshAdaptation_basic<TGrid, otherTGRID>::refine_2(level, Z);
   }

   virtual void compress(const int level, const long long Z) override
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
                            0.125 * (b(i, j, k) + b(i + 1, j, k) + b(i, j + 1, k) + b(i + 1, j + 1, k) +
                                     b(i, j, k + 1) + b(i + 1, j, k + 1) + b(i, j + 1, k + 1) + b(i + 1, j + 1, k + 1));
                        (*Blocks[0])(i / 2 + offsetX[I], j / 2 + offsetY[J], k / 2 + offsetZ[K]) = average;
                     }
            }

      const long long np = m_refGrid->getZforward(level - 1, info.index[0] / 2, info.index[1] / 2, info.index[2] / 2);
      BlockInfo &parent  = m_refGrid->getBlockInfoAll(level - 1, np);
      m_refGrid->Tree(parent.level, parent.Z).setrank(m_refGrid->rank());
      parent.ptrBlock    = info.ptrBlock;
      parent.h_gridpoint = parent.h;
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
      for (int J = 0; J < 2; J++)
         for (int I = 0; I < 2; I++)
         {
            BlockType &b = *Blocks[J * 2 + I];
            for (int j = 0; j < ny; j += 2)
               for (int i = 0; i < nx; i += 2)
               {
                  ElementType average = 0.25 * (b(i, j, 0) + b(i + 1, j, 0) + b(i, j + 1, 0) + b(i + 1, j + 1, 0));
                  (*Blocks[0])(i / 2 + offsetX[I], j / 2 + offsetY[J], 0) = average;
               }
         }
      const long long np = m_refGrid->getZforward(level - 1, info.index[0] / 2, info.index[1] / 2);
      BlockInfo &parent  = m_refGrid->getBlockInfoAll(level - 1, np);
      m_refGrid->Tree(parent.level, parent.Z).setrank(m_refGrid->rank());
      parent.ptrBlock    = info.ptrBlock;
      parent.h_gridpoint = parent.h;
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
                        ElementType dudx = 0.5 * (Lab(i / 2 + offsetX[I] + 1, j / 2 + offsetY[J], k / 2 + offsetZ[K]) -
                                                  Lab(i / 2 + offsetX[I] - 1, j / 2 + offsetY[J], k / 2 + offsetZ[K]));
                        ElementType dudy = 0.5 * (Lab(i / 2 + offsetX[I], j / 2 + offsetY[J] + 1, k / 2 + offsetZ[K]) -
                                                  Lab(i / 2 + offsetX[I], j / 2 + offsetY[J] - 1, k / 2 + offsetZ[K]));
                        ElementType dudz = 0.5 * (Lab(i / 2 + offsetX[I], j / 2 + offsetY[J], k / 2 + offsetZ[K] + 1) -
                                                  Lab(i / 2 + offsetX[I], j / 2 + offsetY[J], k / 2 + offsetZ[K] - 1));

                        b(i, j, k) = Lab(i / 2 + offsetX[I], j / 2 + offsetY[J], k / 2 + offsetZ[K]) +
                                     (2 * (i % 2) - 1) * 0.25 * dudx + (2 * (j % 2) - 1) * 0.25 * dudy +
                                     (2 * (k % 2) - 1) * 0.25 * dudz;
                        b(i + 1, j, k) = Lab(i / 2 + offsetX[I], j / 2 + offsetY[J], k / 2 + offsetZ[K]) +
                                         (2 * ((i + 1) % 2) - 1) * 0.25 * dudx + (2 * (j % 2) - 1) * 0.25 * dudy +
                                         (2 * (k % 2) - 1) * 0.25 * dudz;
                        b(i, j + 1, k) = Lab(i / 2 + offsetX[I], j / 2 + offsetY[J], k / 2 + offsetZ[K]) +
                                         (2 * (i % 2) - 1) * 0.25 * dudx + (2 * ((j + 1) % 2) - 1) * 0.25 * dudy +
                                         (2 * (k % 2) - 1) * 0.25 * dudz;
                        b(i + 1, j + 1, k) = Lab(i / 2 + offsetX[I], j / 2 + offsetY[J], k / 2 + offsetZ[K]) +
                                             (2 * ((i + 1) % 2) - 1) * 0.25 * dudx +
                                             (2 * ((j + 1) % 2) - 1) * 0.25 * dudy + (2 * (k % 2) - 1) * 0.25 * dudz;
                        b(i, j, k + 1) = Lab(i / 2 + offsetX[I], j / 2 + offsetY[J], k / 2 + offsetZ[K]) +
                                         (2 * (i % 2) - 1) * 0.25 * dudx + (2 * (j % 2) - 1) * 0.25 * dudy +
                                         (2 * ((k + 1) % 2) - 1) * 0.25 * dudz;
                        b(i + 1, j, k + 1) = Lab(i / 2 + offsetX[I], j / 2 + offsetY[J], k / 2 + offsetZ[K]) +
                                             (2 * ((i + 1) % 2) - 1) * 0.25 * dudx + (2 * (j % 2) - 1) * 0.25 * dudy +
                                             (2 * ((k + 1) % 2) - 1) * 0.25 * dudz;
                        b(i, j + 1, k + 1) = Lab(i / 2 + offsetX[I], j / 2 + offsetY[J], k / 2 + offsetZ[K]) +
                                             (2 * (i % 2) - 1) * 0.25 * dudx + (2 * ((j + 1) % 2) - 1) * 0.25 * dudy +
                                             (2 * ((k + 1) % 2) - 1) * 0.25 * dudz;
                        b(i + 1, j + 1, k + 1) = Lab(i / 2 + offsetX[I], j / 2 + offsetY[J], k / 2 + offsetZ[K]) +
                                                 (2 * ((i + 1) % 2) - 1) * 0.25 * dudx +
                                                 (2 * ((j + 1) % 2) - 1) * 0.25 * dudy +
                                                 (2 * ((k + 1) % 2) - 1) * 0.25 * dudz;
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
                  b(i, j, 0) = Lab(i / 2 + offsetX[I], j / 2 + offsetY[J]) - 0.25 * dudx - 0.25 * dudy;
                  b(i + 1, j, 0) = Lab(i / 2 + offsetX[I], j / 2 + offsetY[J]) + 0.25 * dudx - 0.25 * dudy;
                  b(i, j + 1, 0) = Lab(i / 2 + offsetX[I], j / 2 + offsetY[J]) - 0.25 * dudx + 0.25 * dudy;
                  b(i + 1, j + 1, 0) = Lab(i / 2 + offsetX[I], j / 2 + offsetY[J]) + 0.25 * dudx + 0.25 * dudy;
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
