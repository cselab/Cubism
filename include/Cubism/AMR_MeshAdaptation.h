#pragma once
#include "BlockInfo.h"
#include "BlockLab.h"
#include "BlockLabMPI.h"
#include "Grid.h"
#include "GridMPI.h"
#include "LoadBalancer.h"
#include "Matrix3D.h"

#include <algorithm>
#include <cstring>
#include <omp.h>
#include <string>

namespace cubism
{

#define WENOWAVELET 3

template <typename TGrid, typename TLab>
class MeshAdaptation
{

 public:
   typedef typename TGrid::BlockType BlockType;
   typedef typename TGrid::BlockType::ElementType ElementType;
   //typedef typename TGrid::BlockType Block;

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

 public:
   MeshAdaptation(TGrid &grid, double Rtol, double Ctol)
   {
      bool tensorial = true;

      const int Gx = (WENOWAVELET == 3) ? 1 : 2;
      const int Gy = (WENOWAVELET == 3) ? 1 : 2;
      const int Gz = (WENOWAVELET == 3) ? 1 : 2;

      components.push_back(0);
      components.push_back(1);
      components.push_back(2);
      components.push_back(3);
      components.push_back(4);
      components.push_back(5);
      components.push_back(6);

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

      flag      = true;
   }

   virtual ~MeshAdaptation() {  }

   virtual void AdaptTheMesh(double t = 0)
   {
      time = t;

      vector<BlockInfo> & avail0= m_refGrid->getBlocksInfo();

      const int nthreads = omp_get_max_threads();

      labs = new TLab[nthreads];
      //for (int i = 0; i < nthreads; i++) labs[i].prepare(*m_refGrid, s[0], e[0],s[1],e[1],s[2],e[2],true,Is[0],Ie[0],Is[1],Ie[1],Is[2],Ie[2]);


      bool CallValidStates = false;

      /*------------->*/ Clock.start(1, "MeshAdaptation: inner blocks tagging");
      const int Ninner = avail0.size();
      #pragma omp parallel num_threads(nthreads)
      {
         int tid     = omp_get_thread_num();
         TLab &mylab = labs[tid];

         #pragma omp for schedule(dynamic, 1)
         for (int i = 0; i < Ninner; i++)
         {
            BlockInfo &ary0 = avail0[i];
            mylab.load(ary0, t);
            BlockInfo &info = m_refGrid->getBlockInfoAll(ary0.level, ary0.Z);
            ary0.state      = TagLoadedBlock(labs[tid],info.level);
            info.state      = ary0.state;
            #pragma omp critical
            {
                if (info.state != Leave)
                    CallValidStates = true;
            }
         }
      }
      /*------------->*/ Clock.finish(1);

      /*------------->*/ Clock.start(4, "MeshAdaptation: ValidStates");
      if (CallValidStates) ValidStates();
      /*------------->*/ Clock.finish(4);

      // Refinement/compression of blocks
      /*************************************************/
      /*------------->*/ Clock.start(6, "MeshAdaptation: refinement and compression");
      int r = 0;
      int c = 0;

      std::vector<int> mn_com;
      std::vector<int> mn_ref;

      std::vector<BlockInfo> &I = m_refGrid->getBlocksInfo();

      for (auto &i : I)
      {
         BlockInfo &info = m_refGrid->getBlockInfoAll(i.level, i.Z);
         if (info.state == Refine)
         {
            mn_ref.push_back(info.level);
            mn_ref.push_back(info.Z);
         }
         else if (info.state == Compress)
         {
            mn_com.push_back(info.level);
            mn_com.push_back(info.Z);
         }
      }

      #pragma omp parallel
      {
         #pragma omp for schedule(runtime)
         for (size_t i = 0; i < mn_ref.size() / 2; i++)
         {
            int m = mn_ref[2 * i];
            int n = mn_ref[2 * i + 1];
            refine_1(m, n);
            #pragma omp atomic
            r++;
         }
         #pragma omp for schedule(runtime)
         for (size_t i = 0; i < mn_ref.size() / 2; i++)
         {
            int m = mn_ref[2 * i];
            int n = mn_ref[2 * i + 1];
            refine_2(m, n);
         }
     }

      #pragma omp parallel
      {
         #pragma omp for schedule(runtime)
         for (size_t i = 0; i < mn_com.size() / 2; i++)
         {
            int m = mn_com[2 * i];
            int n = mn_com[2 * i + 1];
            compress(m, n);
            #pragma omp atomic
            c++;
         }
      }
      int result[2] = {r,c};
      std::cout << "==============================================================\n";
      std::cout << " refined:" << result[0] << "   compressed:" << result[1] << std::endl;
      std::cout << "==============================================================\n";
      m_refGrid->FillPos();     
      delete[] labs;
      if (!CallValidStates)
      {
        m_refGrid->UpdateFluxCorrection = flag;
        flag                            = false;
      }
      /*************************************************/
      /*------------->*/ Clock.finish(6);
   }

 protected:
   void refine_1(int level, int Z)
   {
      int tid = omp_get_thread_num();

      BlockInfo &parent = m_refGrid->getBlockInfoAll(level, Z);
      parent.state      = Leave;
      labs[tid].load(parent, time, true);

      int p[3] = {parent.index[0], parent.index[1], parent.index[2]};

      assert(parent.ptrBlock != NULL);
      assert(level <= m_refGrid->getlevelMax() - 1);

      BlockType *Blocks[8];

      for (int k = 0; k < 2; k++)
         for (int j = 0; j < 2; j++)
            for (int i = 0; i < 2; i++)
            {
               int nc = m_refGrid->getZforward(level + 1, 2 * p[0] + i, 2 * p[1] + j, 2 * p[2] + k);
               BlockInfo &Child = m_refGrid->getBlockInfoAll(level + 1, nc);

               Child.state = Leave;
               #pragma omp critical
               {
                  m_refGrid->_alloc(level + 1, nc);
               }
               Blocks[k * 4 + j * 2 + i] = (BlockType *)Child.ptrBlock;
            }
      RefineBlocks(Blocks, parent);
   }

   void refine_2(int level, int Z)
   {
      #pragma omp critical
      {
         m_refGrid->_dealloc(level, Z);
         m_refGrid->getBlockInfoAll(level, Z).state = Leave;
      }

      BlockInfo &parent = m_refGrid->getBlockInfoAll(level, Z);
      parent.TreePos    = CheckFiner;

      int p[3] = {parent.index[0], parent.index[1], parent.index[2]};
      for (int k = 0; k < 2; k++)
         for (int j = 0; j < 2; j++)
            for (int i = 0; i < 2; i++)
            {
               int nc = m_refGrid->getZforward(level + 1, 2 * p[0] + i, 2 * p[1] + j, 2 * p[2] + k);
               BlockInfo &Child = m_refGrid->getBlockInfoAll(level + 1, nc);
               Child.TreePos    = Exists;
            }
   }

   virtual void compress(int level, int Z)
   {
        int rank=0;
        assert(level > 0);

        BlockInfo &info = m_refGrid->getBlockInfoAll(level, Z);

        assert(info.TreePos == Exists);
        assert(info.state == Compress);
        assert(info.myrank == rank);


        BlockType *Blocks[8];
        for (int K = 0; K < 2; K++)
        for (int J = 0; J < 2; J++)
        for (int I = 0; I < 2; I++)
        {
            int blk = K * 4 + J * 2 + I;
            int n   = m_refGrid->getZforward(level, info.index[0] + I, info.index[1] + J,info.index[2] + K);
            Blocks[blk] = (BlockType *)(m_refGrid->getBlockInfoAll(level, n)).ptrBlock;
        }

        const int nx   = BlockType::sizeX;
        const int ny   = BlockType::sizeY;
        const int nz   = BlockType::sizeZ;
        int offsetX[2] = {0, nx / 2};
        int offsetY[2] = {0, ny / 2};
        int offsetZ[2] = {0, nz / 2};
        for (int K = 0; K < 2; K++)
        for (int J = 0; J < 2; J++)
        for (int I = 0; I < 2; I++)
        {
            BlockType &b = *Blocks[K * 4 + J * 2 + I];
            for (int k = 0; k < nz; k += 2)
            for (int j = 0; j < ny; j += 2)
            for (int i = 0; i < nx; i += 2)
            {
                ElementType average =  0.125 * (b(i, j    , k    ) + b(i + 1, j    , k    ) + 
                                                b(i, j + 1, k    ) + b(i + 1, j + 1, k    ) + 
                                                b(i, j    , k + 1) + b(i + 1, j    , k + 1) +
                                                b(i, j + 1, k + 1) + b(i + 1, j + 1, k + 1));
                (*Blocks[0])(i / 2 + offsetX[I], j / 2 + offsetY[J], k / 2 + offsetZ[K]) = average;
            }
        }

        int np             = m_refGrid->getZforward(level - 1, info.index[0] / 2, info.index[1] / 2, info.index[2] / 2);
        BlockInfo &parent  = m_refGrid->getBlockInfoAll(level - 1, np);
        parent.myrank      = m_refGrid->rank();
        parent.ptrBlock    = info.ptrBlock;
        parent.TreePos     = Exists;
        parent.h_gridpoint = parent.h;
        parent.state       = Leave;

        #pragma omp critical
        {
            for (int K = 0; K < 2; K++)
            for (int J = 0; J < 2; J++)
            for (int I = 0; I < 2; I++)
            {
                int n = m_refGrid->getZforward(level, info.index[0] + I, info.index[1] + J,info.index[2] + K);
                if (I + J + K == 0)
                {
                   m_refGrid->FindBlockInfo(level, n, level - 1, np);
                }
                else
                {
                   m_refGrid->_dealloc(level, n);
                }
                m_refGrid->getBlockInfoAll(level, n).TreePos = CheckCoarser;
                m_refGrid->getBlockInfoAll(level, n).state   = Leave;
            }
        }
   }

   virtual void ValidStates()
   {
      static std::array<int, 3> blocksPerDim = m_refGrid->getMaxBlocks();
      static const int levelMin              = 0;
      static const int levelMax              = m_refGrid->getlevelMax();
      static const bool xperiodic            = labs[0].is_xperiodic();
      static const bool yperiodic            = labs[0].is_yperiodic();
      static const bool zperiodic            = labs[0].is_zperiodic();

      std::vector<BlockInfo> &I = m_refGrid->getBlocksInfo();

      for (size_t j = 0; j < I.size(); j++)
      {
         BlockInfo &info = I[j];
         if ((info.state == Refine && info.level == levelMax - 1) ||
             (info.state == Compress && info.level == levelMin))
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
         /*------------->*/ Clock.start(20, "MeshAdaptation: step 1");
         for (size_t j = 0; j < I.size(); j++)
         {
            BlockInfo &info = I[j];
            if (info.level == m && info.state != Refine && info.level != levelMax - 1)
            {
               assert(info.TreePos == Exists);
               int TwoPower = 1 << info.level;
               const bool xskin =
                   info.index[0] == 0 || info.index[0] == blocksPerDim[0] * TwoPower - 1;
               const bool yskin =
                   info.index[1] == 0 || info.index[1] == blocksPerDim[1] * TwoPower - 1;
               const bool zskin =
                   info.index[2] == 0 || info.index[2] == blocksPerDim[2] * TwoPower - 1;
               const int xskip = info.index[0] == 0 ? -1 : 1;
               const int yskip = info.index[1] == 0 ? -1 : 1;
               const int zskip = info.index[2] == 0 ? -1 : 1;

               for (int icode = 0; icode < 27; icode++)
               {
                  if (icode == 1 * 1 + 3 * 1 + 9 * 1) continue;
                  const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, (icode / 9) % 3 - 1};
                  if (!xperiodic && code[0] == xskip && xskin) continue;
                  if (!yperiodic && code[1] == yskip && yskin) continue;
                  if (!zperiodic && code[2] == zskip && zskin) continue;

                  BlockInfo &infoNei =
                      m_refGrid->getBlockInfoAll(info.level, info.Znei_(code[0], code[1], code[2]));
                  if (infoNei.TreePos == CheckFiner)
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

                     for (int B = 0; B <= 3;
                          B += Bstep) // loop over blocks that make up face/edge/corner
                                      // (respectively 4,2 or 1 blocks)
                     {
                        const int aux = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
                        int iNei      = 2 * info.index[0] + max(code[0], 0) + code[0] +
                                   (B % 2) * max(0, 1 - abs(code[0]));
                        int jNei = 2 * info.index[1] + max(code[1], 0) + code[1] +
                                   aux * max(0, 1 - abs(code[1]));
                        int kNei = 2 * info.index[2] + max(code[2], 0) + code[2] +
                                   (B / 2) * max(0, 1 - abs(code[2]));
                        int zzz             = m_refGrid->getZforward(m + 1, iNei, jNei, kNei);
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
         /*------------->*/ Clock.finish(20);

         if (m == levelMin) break;
         // 2.
         /*------------->*/ Clock.start(21, "MeshAdaptation: step 2");
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

                  BlockInfo &infoNei =
                      m_refGrid->getBlockInfoAll(info.level, info.Znei_(code[0], code[1], code[2]));
                  if (infoNei.TreePos == Exists && infoNei.state == Refine)
                  {
                     info.state                                             = Leave;
                     (m_refGrid->getBlockInfoAll(info.level, info.Z)).state = Leave;
                     break;
                  }
               }
            }
         }
         /*------------->*/ Clock.finish(21);
      } // m

      /*------------->*/ Clock.start(22, "MeshAdaptation: step 3");
      // 3.
      for (size_t jjj = 0; jjj < I.size(); jjj++)
      {
         BlockInfo &info = I[jjj];
         int m = info.level;
         bool found = false;
         for (int i = 2 * (info.index[0] / 2); i <= 2 * (info.index[0] / 2) + 1; i++)
            for (int j = 2 * (info.index[1] / 2); j <= 2 * (info.index[1] / 2) + 1; j++)
               for (int k = 2 * (info.index[2] / 2); k <= 2 * (info.index[2] / 2) + 1; k++)
               {
                  int n              = m_refGrid->getZforward(m, i, j, k);
                  BlockInfo &infoNei = m_refGrid->getBlockInfoAll(m, n);
                  if (infoNei.TreePos != Exists || infoNei.state != Compress)
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
                  int n              = m_refGrid->getZforward(m, i, j, k);
                  BlockInfo &infoNei = m_refGrid->getBlockInfoAll(m, n);
                  if (infoNei.TreePos == Exists && infoNei.state == Compress)
                  {
                     infoNei.state = Leave;
                  }
               }
      }
       /*------------->*/ Clock.finish(22);

       /*------------->*/ Clock.start(23, "MeshAdaptation: step 4");
      // 4.
      for (size_t jjj = 0; jjj < I.size(); jjj++)
      {
         BlockInfo &info = I[jjj];
         if (info.state == Compress)
         {
            int m = info.level;
            bool first = true;
            for (int i = 2 * (info.index[0] / 2); i <= 2 * (info.index[0] / 2) + 1; i++)
               for (int j = 2 * (info.index[1] / 2); j <= 2 * (info.index[1] / 2) + 1; j++)
                  for (int k = 2 * (info.index[2] / 2); k <= 2 * (info.index[2] / 2) + 1; k++)
                  {
                     int n              = m_refGrid->getZforward(m, i, j, k);
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
      /*------------->*/ Clock.finish(23);
   }

   ////////////////////////////////////////////////////////////////////////////////////////////////
   // Virtual functions that can be overwritten by user
   ////////////////////////////////////////////////////////////////////////////////////////////////
   virtual void RefineBlocks(BlockType *B[8], BlockInfo parent)
   {
      int tid = omp_get_thread_num();

      const int nx = BlockType::sizeX;
      const int ny = BlockType::sizeY;
      const int nz = BlockType::sizeZ;

      int offsetX[2] = {0, nx / 2};
      int offsetY[2] = {0, ny / 2};
      int offsetZ[2] = {0, nz / 2};

      TLab &Lab = labs[tid];

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

                #if 0 // simple linear 
      
                  ElementType dudx = 0.5*( Lab(i/2+offsetX[I]+1,j/2+offsetY[J]  ,k/2+offsetZ[K]  )-Lab(i/2+offsetX[I]-1,j/2+offsetY[J]  ,k/2+offsetZ[K]  ));
                  ElementType dudy = 0.5*( Lab(i/2+offsetX[I]  ,j/2+offsetY[J]+1,k/2+offsetZ[K]  )-Lab(i/2+offsetX[I]  ,j/2+offsetY[J]-1,k/2+offsetZ[K]  ));
                  ElementType dudz = 0.5*( Lab(i/2+offsetX[I]  ,j/2+offsetY[J]  ,k/2+offsetZ[K]+1)-Lab(i/2+offsetX[I]  ,j/2+offsetY[J]  ,k/2+offsetZ[K]-1));
      
                  b(i  ,j  ,k  ) = Lab( i   /2+offsetX[I], j   /2+offsetY[J]  ,k    /2+offsetZ[K] )+ (2*( i   %2)-1)*0.25*dudx + (2*( j   %2)-1)*0.25*dudy + (2*(k    %2)-1)*0.25*dudz; 
                  b(i+1,j  ,k  ) = Lab((i+1)/2+offsetX[I], j   /2+offsetY[J]  ,k    /2+offsetZ[K] )+ (2*((i+1)%2)-1)*0.25*dudx + (2*( j   %2)-1)*0.25*dudy + (2*(k    %2)-1)*0.25*dudz; 
                  b(i  ,j+1,k  ) = Lab( i   /2+offsetX[I],(j+1)/2+offsetY[J]  ,k    /2+offsetZ[K] )+ (2*( i   %2)-1)*0.25*dudx + (2*((j+1)%2)-1)*0.25*dudy + (2*(k    %2)-1)*0.25*dudz; 
                  b(i+1,j+1,k  ) = Lab((i+1)/2+offsetX[I],(j+1)/2+offsetY[J]  ,k    /2+offsetZ[K] )+ (2*((i+1)%2)-1)*0.25*dudx + (2*((j+1)%2)-1)*0.25*dudy + (2*(k    %2)-1)*0.25*dudz; 
                  b(i  ,j  ,k+1) = Lab( i   /2+offsetX[I], j   /2+offsetY[J]  ,(k+1)/2+offsetZ[K] )+ (2*( i   %2)-1)*0.25*dudx + (2*( j   %2)-1)*0.25*dudy + (2*((k+1)%2)-1)*0.25*dudz; 
                  b(i+1,j  ,k+1) = Lab((i+1)/2+offsetX[I], j   /2+offsetY[J]  ,(k+1)/2+offsetZ[K] )+ (2*((i+1)%2)-1)*0.25*dudx + (2*( j   %2)-1)*0.25*dudy + (2*((k+1)%2)-1)*0.25*dudz; 
                  b(i  ,j+1,k+1) = Lab( i   /2+offsetX[I],(j+1)/2+offsetY[J]  ,(k+1)/2+offsetZ[K] )+ (2*( i   %2)-1)*0.25*dudx + (2*((j+1)%2)-1)*0.25*dudy + (2*((k+1)%2)-1)*0.25*dudz; 
                  b(i+1,j+1,k+1) = Lab((i+1)/2+offsetX[I],(j+1)/2+offsetY[J]  ,(k+1)/2+offsetZ[K] )+ (2*((i+1)%2)-1)*0.25*dudx + (2*((j+1)%2)-1)*0.25*dudy + (2*((k+1)%2)-1)*0.25*dudz;
                #else // WENO

                        const int Nweno = WENOWAVELET;
                        ElementType El[Nweno][Nweno][Nweno];
                        for (int i0 = -Nweno / 2; i0 <= Nweno / 2; i0++)
                           for (int i1 = -Nweno / 2; i1 <= Nweno / 2; i1++)
                              for (int i2 = -Nweno / 2; i2 <= Nweno / 2; i2++)
                                 El[i0 + Nweno / 2][i1 + Nweno / 2][i2 + Nweno / 2] =
                                     Lab(i / 2 + offsetX[I] + i0, j / 2 + offsetY[J] + i1,
                                         k / 2 + offsetZ[K] + i2);

                        ElementType Lines[Nweno][Nweno][2];
                        ElementType Planes[Nweno][4];
                        ElementType Ref[8];

                #if WENOWAVELET == 3

                        for (int i2 = -Nweno / 2; i2 <= Nweno / 2; i2++)
                           for (int i1 = -Nweno / 2; i1 <= Nweno / 2; i1++)
                              Kernel_1D(El[0][i1 + Nweno / 2][i2 + Nweno / 2],
                                        El[1][i1 + Nweno / 2][i2 + Nweno / 2],
                                        El[2][i1 + Nweno / 2][i2 + Nweno / 2],
                                        Lines[i1 + Nweno / 2][i2 + Nweno / 2][0],
                                        Lines[i1 + Nweno / 2][i2 + Nweno / 2][1]);
                        for (int i2 = -Nweno / 2; i2 <= Nweno / 2; i2++)
                        {
                           Kernel_1D(Lines[0][i2 + Nweno / 2][0], Lines[1][i2 + Nweno / 2][0],
                                     Lines[2][i2 + Nweno / 2][0], Planes[i2 + Nweno / 2][0],
                                     Planes[i2 + Nweno / 2][1]);

                           Kernel_1D(Lines[0][i2 + Nweno / 2][1], Lines[1][i2 + Nweno / 2][1],
                                     Lines[2][i2 + Nweno / 2][1], Planes[i2 + Nweno / 2][2],
                                     Planes[i2 + Nweno / 2][3]);
                        }
                        Kernel_1D(Planes[0][0], Planes[1][0], Planes[2][0], Ref[0], Ref[1]);
                        Kernel_1D(Planes[0][1], Planes[1][1], Planes[2][1], Ref[2], Ref[3]);
                        Kernel_1D(Planes[0][2], Planes[1][2], Planes[2][2], Ref[4], Ref[5]);
                        Kernel_1D(Planes[0][3], Planes[1][3], Planes[2][3], Ref[6], Ref[7]);

                #else

                        for (int i2 = -Nweno / 2; i2 <= Nweno / 2; i2++)
                           for (int i1 = -Nweno / 2; i1 <= Nweno / 2; i1++)
                              Kernel_1D(El[0][i1 + Nweno / 2][i2 + Nweno / 2],
                                        El[1][i1 + Nweno / 2][i2 + Nweno / 2],
                                        El[2][i1 + Nweno / 2][i2 + Nweno / 2],
                                        El[3][i1 + Nweno / 2][i2 + Nweno / 2],
                                        El[4][i1 + Nweno / 2][i2 + Nweno / 2],
                                        Lines[i1 + Nweno / 2][i2 + Nweno / 2][0],
                                        Lines[i1 + Nweno / 2][i2 + Nweno / 2][1]);

                        for (int i2 = -Nweno / 2; i2 <= Nweno / 2; i2++)
                        {
                           Kernel_1D(Lines[0][i2 + Nweno / 2][0], Lines[1][i2 + Nweno / 2][0],
                                     Lines[2][i2 + Nweno / 2][0], Lines[3][i2 + Nweno / 2][0],
                                     Lines[4][i2 + Nweno / 2][0], Planes[i2 + Nweno / 2][0],
                                     Planes[i2 + Nweno / 2][1]);

                           Kernel_1D(Lines[0][i2 + Nweno / 2][1], Lines[1][i2 + Nweno / 2][1],
                                     Lines[2][i2 + Nweno / 2][1], Lines[3][i2 + Nweno / 2][1],
                                     Lines[4][i2 + Nweno / 2][1], Planes[i2 + Nweno / 2][2],
                                     Planes[i2 + Nweno / 2][3]);
                        }
                        Kernel_1D(Planes[0][0], Planes[1][0], Planes[2][0], Planes[3][0],
                                  Planes[4][0], Ref[0], Ref[1]);
                        Kernel_1D(Planes[0][1], Planes[1][1], Planes[2][1], Planes[3][1],
                                  Planes[4][1], Ref[2], Ref[3]);
                        Kernel_1D(Planes[0][2], Planes[1][2], Planes[2][2], Planes[3][2],
                                  Planes[4][2], Ref[4], Ref[5]);
                        Kernel_1D(Planes[0][3], Planes[1][3], Planes[2][3], Planes[3][3],
                                  Planes[4][3], Ref[6], Ref[7]);
                #endif

                        b(i, j, k)             = Ref[0];
                        b(i, j, k + 1)         = Ref[1];
                        b(i, j + 1, k)         = Ref[2];
                        b(i, j + 1, k + 1)     = Ref[3];
                        b(i + 1, j, k)         = Ref[4];
                        b(i + 1, j, k + 1)     = Ref[5];
                        b(i + 1, j + 1, k)     = Ref[6];
                        b(i + 1, j + 1, k + 1) = Ref[7];
                #endif
                     }
            }
   }

   virtual void WENOWavelets3(double cm, double c, double cp, double &left, double &right)
   {
      double b1  = (c - cm) * (c - cm);
      double b2  = (c - cp) * (c - cp);
      double w1  = (1e-6 + b2) * (1e-6 + b2); // yes, 2 goes to 1 and 1 goes to 2
      double w2  = (1e-6 + b1) * (1e-6 + b1);
      double aux = 1.0 / (w1 + w2);
      w1 *= aux;
      w2 *= aux;
      double g1, g2;
      g1    = 0.75 * c + 0.25 * cm;
      g2    = 1.25 * c - 0.25 * cp;
      left  = g1 * w1 + g2 * w2;
      g1    = 1.25 * c - 0.25 * cm;
      g2    = 0.75 * c + 0.25 * cp;
      right = g1 * w1 + g2 * w2;
   }

   virtual void WENOWavelets5(double cm2, double cm, double c, double cp, double cp2, double &left,
                              double &right)
   {
      static const double k  = 13.0 / 12.0;
      static const double d1 = 0.625;
      static const double d2 = 0.1875;
      static const double d3 = 0.1875;

      double b1 = k * pow(cm - 2.0 * c + cp, 2) + 0.25 * pow(cm - cp, 2);
      double b2 = k * pow(c - 2.0 * cp + cp2, 2) + 0.25 * pow(3.0 * c - 4.0 * cp + cp2, 2);
      double b3 = k * pow(c - 2.0 * cm + cm2, 2) + 0.25 * pow(3.0 * c - 4.0 * cm + cm2, 2);

      double w1  = d1 / pow(1e-6 + b1, 2);
      double w2  = d2 / pow(1e-6 + b2, 2);
      double w3  = d3 / pow(1e-6 + b3, 2);
      double aux = 1.0 / (w1 + w2 + w3);

      double g1, g2, g3;

      g1   = 0.125 * cm + c - 0.125 * cp;
      g2   = 1.375 * c - 0.5 * cp + 0.125 * cp2;
      g3   = 0.625 * c + 0.5 * cm - 0.125 * cm2;
      left = aux * (g1 * w1 + g2 * w2 + g3 * w3);

      g1    = -0.125 * cm + c + 0.125 * cp;
      g2    = 0.625 * c + 0.5 * cp - 0.125 * cp2;
      g3    = 1.375 * c - 0.5 * cm + 0.125 * cm2;
      right = aux * (g1 * w1 + g2 * w2 + g3 * w3);
   }

#if WENOWAVELET == 3
   virtual void Kernel_1D(ElementType E0, ElementType E1, ElementType E2, ElementType &left,
                          ElementType &right)
   {
      left.dummy  = E1.dummy; // - 0.125*(E2.dummy-E0.dummy);
      right.dummy = E1.dummy; // + 0.125*(E2.dummy-E0.dummy);
      WENOWavelets3(E0.alpha1rho1, E1.alpha1rho1, E2.alpha1rho1, left.alpha1rho1, right.alpha1rho1);
      WENOWavelets3(E0.alpha2rho2, E1.alpha2rho2, E2.alpha2rho2, left.alpha2rho2, right.alpha2rho2);
      WENOWavelets3(E0.ru, E1.ru, E2.ru, left.ru, right.ru);
      WENOWavelets3(E0.rv, E1.rv, E2.rv, left.rv, right.rv);
      WENOWavelets3(E0.rw, E1.rw, E2.rw, left.rw, right.rw);
      WENOWavelets3(E0.alpha2, E1.alpha2, E2.alpha2, left.alpha2, right.alpha2);
      WENOWavelets3(E0.energy, E1.energy, E2.energy, left.energy, right.energy);

      // clipping
      if (left.alpha2 < 0.0 || right.alpha2 < 0.0 || left.alpha2 > 1.0 || right.alpha2 > 1.0) {
        left.alpha2  = E1.alpha2;
        right.alpha2 = E1.alpha2;
      }
  
      if (left.alpha1rho1 < 0.0 || right.alpha1rho1 < 0.0) {
        left.alpha1rho1  = E1.alpha1rho1;
        right.alpha1rho1 = E1.alpha1rho1;
      }
  
      if (left.alpha2rho2 < 0.0 || right.alpha2rho2 < 0.0) {
        left.alpha2rho2  = E1.alpha2rho2;
        right.alpha2rho2 = E1.alpha2rho2;
      }
  
      if (left.energy < 0.0 || right.energy < 0.0) {
        left.energy  = E1.energy;
        right.energy = E1.energy;
      }
   }
#else
   virtual void Kernel_1D(ElementType E0, ElementType E1, ElementType E2, ElementType E3,
                          ElementType E4, ElementType &left, ElementType &right)
   {
      left.dummy = E2.dummy;  // - 0.125*(E2.dummy-E0.dummy);
      right.dummy = E2.dummy; // + 0.125*(E2.dummy-E0.dummy);

      WENOWavelets5(E0.alpha1rho1, E1.alpha1rho1, E2.alpha1rho1, E3.alpha1rho1, E4.alpha1rho1,
                    left.alpha1rho1, right.alpha1rho1);
      WENOWavelets5(E0.alpha2rho2, E1.alpha2rho2, E2.alpha2rho2, E3.alpha2rho2, E4.alpha2rho2,
                    left.alpha2rho2, right.alpha2rho2);
      WENOWavelets5(E0.ru, E1.ru, E2.ru, E3.ru, E4.ru, left.ru, right.ru);
      WENOWavelets5(E0.rv, E1.rv, E2.rv, E3.rv, E4.rv, left.rv, right.rv);
      WENOWavelets5(E0.rw, E1.rw, E2.rw, E3.rw, E4.rw, left.rw, right.rw);
      WENOWavelets5(E0.alpha2, E1.alpha2, E2.alpha2, E3.alpha2, E4.alpha2, left.alpha2,
                    right.alpha2);
      WENOWavelets5(E0.energy, E1.energy, E2.energy, E3.energy, E4.energy, left.energy,
                    right.energy);

      // clipping
      if (left.alpha2 < 0.0 || right.alpha2 < 0.0 || left.alpha2 > 1.0 || right.alpha2 > 1.0) {
        left.alpha2  = E1.alpha2;
        right.alpha2 = E1.alpha2;
      }
  
      if (left.alpha1rho1 < 0.0 || right.alpha1rho1 < 0.0) {
        left.alpha1rho1  = E1.alpha1rho1;
        right.alpha1rho1 = E1.alpha1rho1;
      }
  
      if (left.alpha2rho2 < 0.0 || right.alpha2rho2 < 0.0) {
        left.alpha2rho2  = E1.alpha2rho2;
        right.alpha2rho2 = E1.alpha2rho2;
      }
  
      if (left.energy < 0.0 || right.energy < 0.0) {
        left.energy  = E1.energy;
        right.energy = E1.energy;
      }
   }
#endif
   // Tag block loaded in Lab for refinement/compression; user can write own function

   double get_pressure(double alpha1rho1, double alpha2rho2, double ru, double rv, double rw,
                       double alpha2, double energy)
   {
      static const double g1      = Simulation_Environment::GAMMA1;
      static const double g2      = Simulation_Environment::GAMMA2;
      static const double pc1     = Simulation_Environment::PC1;
      static const double pc2     = Simulation_Environment::PC2;
      static const double g1m1Inv = 1.0 / (g1 - 1.0);
      static const double g2m1Inv = 1.0 / (g2 - 1.0);

      const double alpha1     = 1.0 - alpha2;
      const double gmix_m1Inv = alpha1 * g1m1Inv + alpha2 * g2m1Inv;
      const double pcmix      = alpha1 * g1 * pc1 * g1m1Inv + alpha2 * g2 * pc2 * g2m1Inv;

      const double rho            = alpha1rho1 + alpha2rho2;
      const double kinetic_energy = 0.5 * (ru * ru + rv * rv + rw * rw) / rho;

      const double pressure = (energy - kinetic_energy - pcmix) / gmix_m1Inv;
      return pressure;
   }

   double get_energy(double alpha1rho1, double alpha2rho2, double ru, double rv, double rw,
                     double alpha2, double pressure)
   {
      // assuming zero velocity

      static const double g1      = Simulation_Environment::GAMMA1;
      static const double g2      = Simulation_Environment::GAMMA2;
      static const double pc1     = Simulation_Environment::PC1;
      static const double pc2     = Simulation_Environment::PC2;
      static const double g1m1Inv = 1.0 / (g1 - 1.0);
      static const double g2m1Inv = 1.0 / (g2 - 1.0);

      const double alpha1     = 1.0 - alpha2;
      const double gmix_m1Inv = alpha1 * g1m1Inv + alpha2 * g2m1Inv;
      const double pcmix      = alpha1 * g1 * pc1 * g1m1Inv + alpha2 * g2 * pc2 * g2m1Inv;

      const double rho            = alpha1rho1 + alpha2rho2;
      const double kinetic_energy = 0.5 * (ru * ru + rv * rv + rw * rw) / rho;

      const double energy = gmix_m1Inv * pressure + pcmix + kinetic_energy;

      return energy;
   }

   virtual State TagLoadedBlock(TLab &Lab_, int level)
   {
        static const int nx = BlockType::sizeX;
        static const int ny = BlockType::sizeY;
        static const int nz = BlockType::sizeZ;
    
        // double L1     = 0.0;
        double Linf   = 0.0;
        double Linf_2 = 0.0;
    
        for (int k = 0 - 1; k < nz + 1; k++)
          for (int j = 0 - 1; j < ny + 1; j++)
            for (int i = 0 - 1; i < nx + 1; i++) {
              Lab_(i, j, k).energy = get_pressure(Lab_(i, j, k).alpha1rho1, Lab_(i, j, k).alpha2rho2,
                                                  Lab_(i, j, k).ru, Lab_(i, j, k).rv, Lab_(i, j, k).rw,
                                                  Lab_(i, j, k).alpha2, Lab_(i, j, k).energy);
            }
    
        for (int k = 0; k < nz; k++)
          for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++) {
            #if 0
                      double dudx = 0.5*( Lab_(i+1,j  ,k  ).energy-Lab_(i-1,j  ,k  ).energy);
                      double dudy = 0.5*( Lab_(i  ,j+1,k  ).energy-Lab_(i  ,j-1,k  ).energy);
                      double dudz = 0.5*( Lab_(i  ,j  ,k+1).energy-Lab_(i  ,j  ,k-1).energy);  
                      double gradMag = sqrt(dudx*dudx+dudy*dudy+dudz*dudz);
                      gradMag /= ( 1e-6 + Lab_(i,j,k).energy);
            
            #if 1
                        dudx = 0.5*( Lab_(i+1,j  ,k  ).alpha2-Lab_(i-1,j  ,k  ).alpha2);
                        dudy = 0.5*( Lab_(i  ,j+1,k  ).alpha2-Lab_(i  ,j-1,k  ).alpha2);
                        dudz = 0.5*( Lab_(i  ,j  ,k+1).alpha2-Lab_(i  ,j  ,k-1).alpha2);         
                        double gradMag1 = sqrt(dudx*dudx+dudy*dudy+dudz*dudz);
                        gradMag1 /= ( 1e-6 + Lab_(i,j,k).alpha2); 
                        gradMag = max(gradMag,gradMag1);
            #endif
                
                      L1 += gradMag;
                      Linf = max(Linf,gradMag);
            #else
            
              double s0 = Lab_(i, j, k).energy;
              double s0i = 1.0 / s0;
              double ax = std::abs((Lab_(i + 1, j, k).energy - s0) * s0i);
              double ay = std::abs((Lab_(i, j + 1, k).energy - s0) * s0i);
              double az = std::abs((Lab_(i, j, k + 1).energy - s0) * s0i);
              ax = max(ax, std::abs((s0 - Lab_(i - 1, j, k).energy) * s0i));
              ay = max(ay, std::abs((s0 - Lab_(i, j - 1, k).energy) * s0i));
              az = max(az, std::abs((s0 - Lab_(i, j, k - 1).energy) * s0i));
            
            //       double s0 = Lab_(i, j, k).alpha2rho2;
            //       double s0i = 1.0 / s0;
            //       double ax = std::abs((Lab_(i + 1, j, k).alpha2rho2 - s0) * s0i);
            //       double ay = std::abs((Lab_(i, j + 1, k).alpha2rho2 - s0) * s0i);
            //       double az = std::abs((Lab_(i, j, k + 1).alpha2rho2 - s0) * s0i);
            //       ax = max(ax, std::abs((s0 - Lab_(i - 1, j, k).alpha2rho2) * s0i));
            //       ay = max(ay, std::abs((s0 - Lab_(i, j - 1, k).alpha2rho2) * s0i));
            //       az = max(az, std::abs((s0 - Lab_(i, j, k - 1).alpha2rho2) * s0i));
    
              Linf = max(Linf, ax);
              Linf = max(Linf, ay);
              Linf = max(Linf, az);
    
              s0 = Lab_(i, j, k).alpha2;
              ax = std::abs(Lab_(i + 1, j, k).alpha2 - s0);
              ay = std::abs(Lab_(i, j + 1, k).alpha2 - s0);
              az = std::abs(Lab_(i, j, k + 1).alpha2 - s0);
              ax = max(ax, std::abs(s0 - Lab_(i - 1, j, k).alpha2));
              ay = max(ay, std::abs(s0 - Lab_(i, j - 1, k).alpha2));
              az = max(az, std::abs(s0 - Lab_(i, j, k - 1).alpha2));
              Linf_2 = max(Linf_2, ax);
              Linf_2 = max(Linf_2, ay);
              Linf_2 = max(Linf_2, az);
            #endif
            }
    
        for (int k = 0 - 1; k < nz + 1; k++)
          for (int j = 0 - 1; j < ny + 1; j++)
            for (int i = 0 - 1; i < nx + 1; i++) {
              Lab_(i, j, k).energy = get_energy(Lab_(i, j, k).alpha1rho1, Lab_(i, j, k).alpha2rho2,
                                                Lab_(i, j, k).ru, Lab_(i, j, k).rv, Lab_(i, j, k).rw,
                                                Lab_(i, j, k).alpha2, Lab_(i, j, k).energy);
            }
    
        //if (Linf > tolerance_for_refinement)
        //  return Refine;
        //if (Linf_2 > 0.05)
        //  return Refine;
        //if (Linf < tolerance_for_compression && Linf_2 < 0.01)
        //  return Compress;

        Linf   /= (level+1);
        Linf_2 /= (level+1);

        if (Linf > tolerance_for_refinement)
          return Refine;
        if (Linf_2 > tolerance_for_refinement)
          return Refine;
        if (Linf < tolerance_for_compression && Linf_2 < tolerance_for_compression)
          return Compress;



    
        return Leave;
   }
};

} // namespace cubism
