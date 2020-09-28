#pragma once

#include <algorithm>
#include <cmath>
#include <map>
#include <mpi.h>
#include <omp.h>
#include <vector>

#include "BlockInfo.h"
#include "GrowingVector.h"
#include "PUPkernelsMPI.h"
#include "StencilInfo.h"

CUBISM_NAMESPACE_BEGIN

struct Interface
{
   BlockInfo *infos[2];
   int icode[2];
   bool CoarseStencil;

   Interface(BlockInfo &i0, BlockInfo &i1, int a_icode0, int a_icode1)
   {
      infos[0] = &i0;
      infos[1] = &i1;
      icode[0] = a_icode0;
      icode[1] = a_icode1;
      CoarseStencil = false;
   }
};

struct InterfaceInfo
{
   bool ToBeKept;
   int dis;
   InterfaceInfo()
   {
      ToBeKept = true;
      dis      = 0;
   }
};

struct MyRange
{
   int index;
   int sx, sy, sz, ex, ey, ez;
   bool needed;
   std::vector<int> removedIndices;
   bool avg_down;

   bool contains(MyRange r) const
   {
      if (avg_down != r.avg_down) return false;

      int V  = (ez - sz) * (ey - sy) * (ex - sx);
      int Vr = (r.ez - r.sz) * (r.ey - r.sy) * (r.ex - r.sx);

      return (sx <= r.sx && r.ex <= ex) && (sy <= r.sy && r.ey <= ey) &&
             (sz <= r.sz && r.ez <= ez) && (Vr < V);
   }
   bool operator<(const MyRange &other) const { return index < other.index; }
   void copy(const MyRange &other)
   {
      sx = other.sx;
      sy = other.sy;
      sz = other.sz;
      ex = other.ex;
      ey = other.ey;
      ez = other.ez;
   }
   void Remove(const MyRange &other)
   {
      size_t s = removedIndices.size();
      removedIndices.resize(s + other.removedIndices.size());
      for (size_t i = 0; i < other.removedIndices.size(); i++)
      {
         removedIndices[s + i] = other.removedIndices[i];
      }
   }
};

struct UnPackInfo
{
   int offset;
   int lx;
   int ly;
   int lz;
   int srcxstart;
   int srcystart;
   int srczstart;
   int LX;
   int LY;
   int CoarseVersionOffset;
   int CoarseVersionLX;
   int CoarseVersionLY;
   int CoarseVersionsrcxstart;
   int CoarseVersionsrcystart;
   int CoarseVersionsrczstart;
   int level;
   int Z;
   int icode;
   long long IDreceiver;
};

struct StencilManager
{
   const StencilInfo stencil;
   const StencilInfo Cstencil;
   int blocksize[3];
   int blocksPerDim[3];
   int sLength[3 * 27 * 3];
   std::array<MyRange, 3 * 27> AllStencils;
   MyRange Coarse_Range;

   StencilManager(StencilInfo a_stencil, StencilInfo a_Cstencil, int nX, int nY, int nZ, int bX,int bY, int bZ)
       : stencil(a_stencil), Cstencil(a_Cstencil)
   {
      blocksize[0] = nX;
      blocksize[1] = nY;
      blocksize[2] = nZ;

      blocksPerDim[0] = bX;
      blocksPerDim[1] = bY;
      blocksPerDim[2] = bZ;

      for (int icode = 0; icode < 27; icode++)
      {
         const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, (icode / 9) % 3 - 1};
         sLength[3 * icode + 0] =
             code[0] < 1 ? (code[0] < 0 ? -stencil.sx : blocksize[0]) : stencil.ex - 1;
         sLength[3 * icode + 1] =
             code[1] < 1 ? (code[1] < 0 ? -stencil.sy : blocksize[1]) : stencil.ey - 1;
         sLength[3 * icode + 2] =
             code[2] < 1 ? (code[2] < 0 ? -stencil.sz : blocksize[2]) : stencil.ez - 1;

         sLength[3 * (icode + 27) + 0] =
             code[0] < 1 ? (code[0] < 0 ? -stencil.sx : blocksize[0] / 2) : stencil.ex - 1;
         sLength[3 * (icode + 27) + 1] =
             code[1] < 1 ? (code[1] < 0 ? -stencil.sy : blocksize[1] / 2) : stencil.ey - 1;
         sLength[3 * (icode + 27) + 2] =
             code[2] < 1 ? (code[2] < 0 ? -stencil.sz : blocksize[2] / 2) : stencil.ez - 1;

         sLength[3 * (icode + 2 * 27) + 0] =
             code[0] < 1 ? (code[0] < 0 ? -((stencil.sx - 1) / 2 + Cstencil.sx) : blocksize[0] / 2)
                         : (stencil.ex) / 2 + Cstencil.ex - 1;
         sLength[3 * (icode + 2 * 27) + 1] =
             code[1] < 1 ? (code[1] < 0 ? -((stencil.sy - 1) / 2 + Cstencil.sy) : blocksize[1] / 2)
                         : (stencil.ey) / 2 + Cstencil.ey - 1;
         sLength[3 * (icode + 2 * 27) + 2] =
             code[2] < 1 ? (code[2] < 0 ? -((stencil.sz - 1) / 2 + Cstencil.sz) : blocksize[2] / 2)
                         : (stencil.ez) / 2 + Cstencil.ez - 1;
      }

      for (int icode = 0; icode < 27; icode++)
      {
         const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, (icode / 9) % 3 - 1};
         MyRange &range    = AllStencils[icode];
         range.sx          = code[0] < 1 ? (code[0] < 0 ? nX + stencil.sx : 0) : 0;
         range.sy          = code[1] < 1 ? (code[1] < 0 ? nY + stencil.sy : 0) : 0;
         range.sz          = code[2] < 1 ? (code[2] < 0 ? nZ + stencil.sz : 0) : 0;
         range.ex          = code[0] < 1 ? (code[0] < 0 ? nX : nX) : stencil.ex - 1;
         range.ey          = code[1] < 1 ? (code[1] < 0 ? nY : nY) : stencil.ey - 1;
         range.ez          = code[2] < 1 ? (code[2] < 0 ? nZ : nZ) : stencil.ez - 1;
      }
      for (int icode = 0; icode < 27; icode++)
      {
         const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, (icode / 9) % 3 - 1};
         MyRange &range    = AllStencils[icode + 27];
         range.sx          = code[0] < 1 ? (code[0] < 0 ? nX + 2 * stencil.sx : 0) : 0;
         range.sy          = code[1] < 1 ? (code[1] < 0 ? nY + 2 * stencil.sy : 0) : 0;
         range.sz          = code[2] < 1 ? (code[2] < 0 ? nZ + 2 * stencil.sz : 0) : 0;
         range.ex          = code[0] < 1 ? (code[0] < 0 ? nX : nX) : 2 * stencil.ex - 1;
         range.ey          = code[1] < 1 ? (code[1] < 0 ? nY : nY) : 2 * stencil.ey - 1;
         range.ez          = code[2] < 1 ? (code[2] < 0 ? nZ : nZ) : 2 * stencil.ez - 1;
      }
      for (int icode = 0; icode < 27; icode++)
      {
         const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, (icode / 9) % 3 - 1};
         MyRange &range    = AllStencils[icode + 2 * 27];
         const int eC[3]   = {stencil.ex / 2 + Cstencil.ex + 1 - 1,
                            stencil.ey / 2 + Cstencil.ey + 1 - 1,
                            stencil.ez / 2 + Cstencil.ez + 1 - 1};

         const int sC[3] = {(stencil.sx - 1) / 2 + Cstencil.sx, (stencil.sy - 1) / 2 + Cstencil.sy,
                            (stencil.sz - 1) / 2 + Cstencil.sz};

         range.sx = code[0] < 1 ? (code[0] < 0 ? nX / 2 + sC[0] : 0) : 0;
         range.sy = code[1] < 1 ? (code[1] < 0 ? nY / 2 + sC[1] : 0) : 0;
         range.sz = code[2] < 1 ? (code[2] < 0 ? nZ / 2 + sC[2] : 0) : 0;
         range.ex = code[0] < 1 ? (code[0] < 0 ? nX / 2 : nX / 2) : eC[0];
         range.ey = code[1] < 1 ? (code[1] < 0 ? nY / 2 : nY / 2) : eC[1];
         range.ez = code[2] < 1 ? (code[2] < 0 ? nZ / 2 : nZ / 2) : eC[2];
      }
   }
   void CoarseStencilLength(const int *code, int *L) const
   {
      L[0] = code[0] < 1 ? (code[0] < 0 ? -((stencil.sx - 1) / 2 + Cstencil.sx) : blocksize[0] / 2)
                         : (stencil.ex / 2 + Cstencil.ex - 1);
      L[1] = code[1] < 1 ? (code[1] < 0 ? -((stencil.sy - 1) / 2 + Cstencil.sy) : blocksize[1] / 2)
                         : (stencil.ey / 2 + Cstencil.ey - 1);
      L[2] = code[2] < 1 ? (code[2] < 0 ? -((stencil.sz - 1) / 2 + Cstencil.sz) : blocksize[2] / 2)
                         : (stencil.ez / 2 + Cstencil.ez - 1);

   }

   void DetermineStencilLength(const int level_sender, const int level_receiver, const int icode, int *L)
   {
      if (level_sender == level_receiver)
      {
         L[0] = sLength[3 * icode +0];
         L[1] = sLength[3 * icode +1];
         L[2] = sLength[3 * icode +2];
      }
      else if (level_sender > level_receiver)
      {
         L[0] = sLength[3 * (icode + 27) +0];
         L[1] = sLength[3 * (icode + 27) +1];
         L[2] = sLength[3 * (icode + 27) +2];
      }
      else
      {
         L[0] = sLength[3 * (icode + 2 * 27) +0];
         L[1] = sLength[3 * (icode + 2 * 27) +1];
         L[2] = sLength[3 * (icode + 2 * 27) +2];
      }
   }

   MyRange &DetermineStencil(const Interface &f, bool CoarseVersion = false)
   {
      if (CoarseVersion)
      {
         AllStencils[f.icode[1] + 2 * 27].needed = true;
         return AllStencils[f.icode[1] + 2 * 27];
      }
      else
      {
         if (f.infos[0]->level == f.infos[1]->level)
         {
            AllStencils[f.icode[1]].needed = true;
            return AllStencils[f.icode[1]];
         }
         else if (f.infos[0]->level > f.infos[1]->level)
         {
            AllStencils[f.icode[1] + 27].needed = true;
            return AllStencils[f.icode[1] + 27];
         }
         else
         {
            Coarse_Range.needed = true;

            const int code[3] = {f.icode[1] % 3 - 1, (f.icode[1] / 3) % 3 - 1,
                                 (f.icode[1] / 9) % 3 - 1};

            const int nX = blocksize[0];
            const int nY = blocksize[1];
            const int nZ = blocksize[2];

            const int s[3] = {
                code[0] < 1 ? (code[0] < 0 ? ((stencil.sx - 1) / 2 + Cstencil.sx) : 0) : nX / 2,
                code[1] < 1 ? (code[1] < 0 ? ((stencil.sy - 1) / 2 + Cstencil.sy) : 0) : nY / 2,
                code[2] < 1 ? (code[2] < 0 ? ((stencil.sz - 1) / 2 + Cstencil.sz) : 0) : nZ / 2};

            const int e[3] = {code[0] < 1 ? (code[0] < 0 ? 0 : nX / 2)
                                          : nX / 2 + (stencil.ex) / 2 + Cstencil.ex - 1,
                              code[1] < 1 ? (code[1] < 0 ? 0 : nY / 2)
                                          : nY / 2 + (stencil.ey) / 2 + Cstencil.ey - 1,
                              code[2] < 1 ? (code[2] < 0 ? 0 : nZ / 2)
                                          : nZ / 2 + (stencil.ez) / 2 + Cstencil.ez - 1};

            const int base[3] = {(f.infos[1]->index[0] + code[0]) % 2,
                                 (f.infos[1]->index[1] + code[1]) % 2,
                                 (f.infos[1]->index[2] + code[2]) % 2};

            int aux = 1 << f.infos[1]->level;
            int Cindex[3];
            for (int d = 0; d < 3; d++)
               Cindex[d] = (f.infos[1]->index[d] + code[d] + aux * blocksPerDim[d]) %
                           (aux * blocksPerDim[d]);

            int CoarseEdge[3];

            CoarseEdge[0] =
                (code[0] == 0)
                    ? 0
                    : (((f.infos[1]->index[0] % 2 == 0) && (Cindex[0] > f.infos[1]->index[0])) ||
                       ((f.infos[1]->index[0] % 2 == 1) && (Cindex[0] < f.infos[1]->index[0])))
                          ? 1
                          : 0;
            CoarseEdge[1] =
                (code[1] == 0)
                    ? 0
                    : (((f.infos[1]->index[1] % 2 == 0) && (Cindex[1] > f.infos[1]->index[1])) ||
                       ((f.infos[1]->index[1] % 2 == 1) && (Cindex[1] < f.infos[1]->index[1])))
                          ? 1
                          : 0;
            CoarseEdge[2] =
                (code[2] == 0)
                    ? 0
                    : (((f.infos[1]->index[2] % 2 == 0) && (Cindex[2] > f.infos[1]->index[2])) ||
                       ((f.infos[1]->index[2] % 2 == 1) && (Cindex[2] < f.infos[1]->index[2])))
                          ? 1
                          : 0;

            Coarse_Range.sx = s[0] + max(code[0], 0) * nX / 2 +
                              (1 - abs(code[0])) * base[0] * nX / 2 - code[0] * nX +
                              CoarseEdge[0] * code[0] * nX / 2;
            Coarse_Range.sy = s[1] + max(code[1], 0) * nY / 2 +
                              (1 - abs(code[1])) * base[1] * nY / 2 - code[1] * nY +
                              CoarseEdge[1] * code[1] * nY / 2;
            Coarse_Range.sz = s[2] + max(code[2], 0) * nZ / 2 +
                              (1 - abs(code[2])) * base[2] * nZ / 2 - code[2] * nZ +
                              CoarseEdge[2] * code[2] * nZ / 2;

            Coarse_Range.ex = e[0] + max(code[0], 0) * nX / 2 +
                              (1 - abs(code[0])) * base[0] * nX / 2 - code[0] * nX +
                              CoarseEdge[0] * code[0] * nX / 2;
            Coarse_Range.ey = e[1] + max(code[1], 0) * nY / 2 +
                              (1 - abs(code[1])) * base[1] * nY / 2 - code[1] * nY +
                              CoarseEdge[1] * code[1] * nY / 2;
            Coarse_Range.ez = e[2] + max(code[2], 0) * nZ / 2 +
                              (1 - abs(code[2])) * base[2] * nZ / 2 - code[2] * nZ +
                              CoarseEdge[2] * code[2] * nZ / 2;

            return Coarse_Range;
         }
      }
   }

   void __FixDuplicates(const Interface &f, const Interface &f_dup, int lx, int ly, int lz,
                        int lx_dup, int ly_dup, int lz_dup, int &sx, int &sy, int &sz)
   {
      BlockInfo &receiver     = *f.infos[1];
      BlockInfo &receiver_dup = *f_dup.infos[1];

      if (receiver.level >= receiver_dup.level)
      {
         int icode_dup         = f_dup.icode[1];
         const int code_dup[3] = {icode_dup % 3 - 1, (icode_dup / 3) % 3 - 1,
                                  (icode_dup / 9) % 3 - 1};
         sx                    = (lx == lx_dup || code_dup[0] != -1) ? 0 : lx - lx_dup;
         sy                    = (ly == ly_dup || code_dup[1] != -1) ? 0 : ly - ly_dup;
         sz                    = (lz == lz_dup || code_dup[2] != -1) ? 0 : lz - lz_dup;
      }
      else
      {
         MyRange range     = DetermineStencil(f);
         MyRange range_dup = DetermineStencil(f_dup);

         sx = range_dup.sx - range.sx;
         sy = range_dup.sy - range.sy;
         sz = range_dup.sz - range.sz;
      }
   }
   void __FixDuplicates2(const Interface &f, const Interface &f_dup, int &sx, int &sy, int &sz)
   {
      if (f.infos[0]->level != f.infos[1]->level || f_dup.infos[0]->level != f_dup.infos[1]->level)
         return;

      MyRange range     = DetermineStencil(f, true);
      MyRange range_dup = DetermineStencil(f_dup, true);
      sx                = range_dup.sx - range.sx;
      sy                = range_dup.sy - range.sy;
      sz                = range_dup.sz - range.sz;
   }
};

template <typename Real>
class SynchronizerMPI_AMR
{
   StencilInfo stencil;  // stencil associated with kernel (advection,diffusion etc.)
   StencilInfo Cstencil; // stencil required to do coarse->fine interpolation

   MPI_Comm comm;
   int rank, size;
   const bool xperiodic, yperiodic, zperiodic;

   struct PackInfo
   {
      Real *block, *pack;
      int sx, sy, sz, ex, ey, ez;
   };

   std::vector<GrowingVector<Real>> send_buffer;
   std::vector<GrowingVector<Real>> recv_buffer;

   static std::vector<GrowingVector<Interface>> send_interfaces;
   std::vector<std::vector<InterfaceInfo>> send_interfaces_infos;
   std::vector<GrowingVector<PackInfo>> send_packinfos;

   std::vector<int> send_buffer_size;
   std::vector<int> recv_buffer_size;

   std::vector<std::vector<int>> ToBeAveragedDown;

   std::vector<MPI_Request> send_requests;
   std::vector<MPI_Request> recv_requests;

   // grid parameters
   const int levelMax;
   int blocksPerDim[3];
   int blocksize[3];
   //SpaceFillingCurve * Zcurve;
   size_t myInfos_size;
   BlockInfo *myInfos;
   std::vector<std::vector<BlockInfo> *> BlockInfoAll;

   static std::vector<BlockInfo *> inner_blocks;
   static std::vector<BlockInfo *> halo_blocks;

   struct UnpacksManagerStruct
   {
      size_t blocks;
      size_t *sizes;
      int size;

      std::vector<GrowingVector<UnPackInfo>> manyUnpacks;
      std::vector<GrowingVector<UnPackInfo>> manyUnpacks_recv; 

      GrowingVector<GrowingVector<UnPackInfo *>> unpacks;

      std::vector<MPI_Request> pack_requests;
      std::vector<std::array<long long int, 2>> MapOfInfos;

      MPI_Datatype MPI_PACK;

      MPI_Comm comm;

      UnpacksManagerStruct()
      {
         comm = MPI_COMM_WORLD;
         sizes = nullptr;
         MPI_Comm_size(comm, &size);
         manyUnpacks_recv.resize(size);
         manyUnpacks.resize(size);

         int array_of_blocklengths[19];
         MPI_Datatype array_of_types[19];
         for (int i=0;i<18;i++)
         {
             array_of_blocklengths[i] = 1;
             array_of_types[i] = MPI_INT;
         }
         array_of_blocklengths[18] = 1;
         array_of_types[18] = MPI_LONG_LONG_INT;

         MPI_Aint array_of_displacements[19];
         UnPackInfo p;    
         MPI_Aint base; MPI_Get_address(&p,&base);
         MPI_Aint offset; MPI_Get_address(&p.offset,&offset); array_of_displacements[0] = offset - base;
         MPI_Aint lx; MPI_Get_address(&p.lx,&lx); array_of_displacements[1] = lx - base;
         MPI_Aint ly; MPI_Get_address(&p.ly,&ly); array_of_displacements[2] = ly - base;
         MPI_Aint lz; MPI_Get_address(&p.lz,&lz); array_of_displacements[3] = lz - base;
         MPI_Aint srcxstart; MPI_Get_address(&p.srcxstart,&srcxstart); array_of_displacements[4] = srcxstart - base;
         MPI_Aint srcystart; MPI_Get_address(&p.srcystart,&srcystart); array_of_displacements[5] = srcystart - base;
         MPI_Aint srczstart; MPI_Get_address(&p.srczstart,&srczstart); array_of_displacements[6] = srczstart - base;
         MPI_Aint LX; MPI_Get_address(&p.LX,&LX); array_of_displacements[7] = LX - base;
         MPI_Aint LY; MPI_Get_address(&p.LY,&LY); array_of_displacements[8] = LY - base;
         MPI_Aint CoarseVersionOffset; MPI_Get_address(&p.CoarseVersionOffset,&CoarseVersionOffset); array_of_displacements[9] = CoarseVersionOffset - base;
         MPI_Aint CoarseVersionLX; MPI_Get_address(&p.CoarseVersionLX,&CoarseVersionLX); array_of_displacements[10] = CoarseVersionLX - base;
         MPI_Aint CoarseVersionLY; MPI_Get_address(&p.CoarseVersionLY,&CoarseVersionLY); array_of_displacements[11] = CoarseVersionLY - base;
         MPI_Aint CoarseVersionsrcxstart; MPI_Get_address(&p.CoarseVersionsrcxstart,&CoarseVersionsrcxstart); array_of_displacements[12] = CoarseVersionsrcxstart - base;
         MPI_Aint CoarseVersionsrcystart; MPI_Get_address(&p.CoarseVersionsrcystart,&CoarseVersionsrcystart); array_of_displacements[13] = CoarseVersionsrcystart - base;
         MPI_Aint CoarseVersionsrczstart; MPI_Get_address(&p.CoarseVersionsrczstart,&CoarseVersionsrczstart); array_of_displacements[14] = CoarseVersionsrczstart - base;
         MPI_Aint level; MPI_Get_address(&p.level,&level); array_of_displacements[15] = level - base;
         MPI_Aint Z; MPI_Get_address(&p.Z,&Z); array_of_displacements[16] = Z - base;
         MPI_Aint icode; MPI_Get_address(&p.icode,&icode); array_of_displacements[17] = icode - base;
         MPI_Aint IDreceiver; MPI_Get_address(&p.IDreceiver,&IDreceiver); array_of_displacements[18] = IDreceiver - base;
         MPI_Type_create_struct(19, array_of_blocklengths, array_of_displacements, array_of_types,&MPI_PACK);
         MPI_Type_commit(&MPI_PACK);
      }

      void clear()
      {
         if (sizes != nullptr)
         {
            delete[] sizes;
            for (size_t i = 0; i < blocks; i++)
            {
               unpacks[i].clear();
            }
            unpacks.clear();
         }
         for (int i = 0; i < size; i++)
         {
            manyUnpacks[i].clear();
            manyUnpacks_recv[i].clear();
         }
         MapOfInfos.clear();
      }

      ~UnpacksManagerStruct() { clear(); MPI_Type_free(&MPI_PACK);}

      void _allocate(size_t a_blocks, size_t *L)
      {
        blocks = a_blocks;
        unpacks.resize(blocks);
        sizes = new size_t[blocks];
        for (size_t i = 0; i < blocks; i++)
        {
           sizes[i] = 0;
           unpacks[i].resize(L[i]);
        }
      }

      void add(UnPackInfo &info, size_t block_id)
      {
        assert(block_id < blocks);
        assert(sizes[block_id] < unpacks[block_id].size());
        unpacks[block_id][sizes[block_id]] = &info;
        sizes[block_id]++;
      }

      void SendPacks(std::set<int> Neighbors, int timestamp)
      {
         pack_requests.clear();
         assert(pack_requests.size() == 0);

         for (auto &r : Neighbors)
         {
            pack_requests.resize(pack_requests.size() + 1);
            MPI_Isend(manyUnpacks[r].data(), manyUnpacks[r].size(), MPI_PACK, r, timestamp, comm, &pack_requests.back());         
         }
         for (auto &r : Neighbors)
         {
            int number_amount;
            MPI_Status status;
            MPI_Probe(r, timestamp, comm, &status);
            MPI_Get_count(&status, MPI_PACK, &number_amount);
            manyUnpacks_recv[r].resize(number_amount);
            pack_requests.resize(pack_requests.size() + 1);
            MPI_Irecv(manyUnpacks_recv[r].data(), manyUnpacks_recv[r].size(),MPI_PACK, r, timestamp, comm, &pack_requests.back());           
         }       
      }

      void MapIDs()
      {
         if (pack_requests.size() == 0) return;

         MPI_Waitall(pack_requests.size(), pack_requests.data(), MPI_STATUSES_IGNORE);
         pack_requests.clear();        

         std::sort(MapOfInfos.begin(), MapOfInfos.end());

         int rank;
         MPI_Comm_rank(comm,&rank);

         for (int r = 0; r < size; r++) if (r!=rank)
            for (size_t i = 0; i < manyUnpacks_recv[r].size(); i++)
            {
               UnPackInfo &info                     = manyUnpacks_recv[r][i];
               std::array<long long int, 2> element = {info.IDreceiver, -1};
               auto low   = std::lower_bound(MapOfInfos.begin(), MapOfInfos.end(), element);
               int Target = (*low)[1];
               assert((*low)[0] == info.IDreceiver);
               add(info, Target);
            }
      }
   };

   UnpacksManagerStruct UnpacksManager;
   StencilManager SM;

#if 1 //new
   struct DuplicatesManager
   {
     #if 0
      struct cube // could be more efficient, fix later
      {
         GrowingVector<MyRange> compass[27];

         void clear()
         {
            for (int i = 0; i < 27; i++) compass[i].clear();
         }

         void EraseAll()
         {
            for (int i = 0; i < 27; i++) compass[i].EraseAll();
         }

         cube() {}

         std::vector<MyRange *> keepEl()
         {
            std::vector<MyRange *> retval;

            for (int i = 0; i < 27; i++)
            {
               for (size_t j = 0; j < compass[i].size(); j++)
               {
                  if (compass[i][j].needed) retval.push_back(&compass[i][j]);
               }
            }
            return retval;
         }

         void __needed(std::vector<int> &v)
         {
            static constexpr std::array<int, 3> faces_and_edges[18] = {
                {0, 1, 1}, {2, 1, 1}, {1, 0, 1}, {1, 2, 1}, {1, 1, 0}, {1, 1, 2},

                {0, 0, 1}, {0, 2, 1}, {2, 0, 1}, {2, 2, 1}, {1, 0, 0}, {1, 0, 2},
                {1, 2, 0}, {1, 2, 2}, {0, 1, 0}, {0, 1, 2}, {2, 1, 0}, {2, 1, 2}};

            for (auto &f : faces_and_edges)
               if (compass[f[0] + f[1] * 3 + f[2] * 9].size() != 0)
               {
                  auto &me = compass[f[0] + f[1] * 3 + f[2] * 9];

                  bool needme  = false;
                  auto &other1 = compass[f[0] + f[1] * 3 + f[2] * 9];
                  for (size_t j1 = 0; j1 < other1.size(); j1++)
                  {
                     auto &o = other1[j1];
                     if (o.needed)
                     {
                        needme = true;

                        for (size_t j = 0; j < me.size(); j++)
                        {
                           auto &m = me[j];
                           if (m.needed && m.contains(o))
                           {
                              o.needed = false;
                              m.removedIndices.push_back(o.index);
                              m.Remove(o);
                              v.push_back(o.index);
                              break;
                           }
                        }
                     }
                  }
                  if (!needme) continue;

                  int imax = (f[0] == 1) ? 2 : f[0];
                  int imin = (f[0] == 1) ? 0 : f[0];

                  int jmax = (f[1] == 1) ? 2 : f[1];
                  int jmin = (f[1] == 1) ? 0 : f[1];

                  int kmax = (f[2] == 1) ? 2 : f[2];
                  int kmin = (f[2] == 1) ? 0 : f[2];

                  for (int k = kmin; k <= kmax; k++)
                     for (int j = jmin; j <= jmax; j++)
                        for (int i = imin; i <= imax; i++)
                        {
                           if (i == f[0] && j == f[1] && k == f[2]) continue;
                           auto &other = compass[i + j * 3 + k * 9];

                           for (size_t j1 = 0; j1 < other.size(); j1++)
                           {
                              auto &o = other[j1];
                              if (o.needed)
                              {
                                 for (size_t k1 = 0; k1 < me.size(); k1++)
                                 {
                                    auto &m = me[k1];
                                    if (m.needed && m.contains(o))
                                    {
                                       o.needed = false;
                                       m.removedIndices.push_back(o.index);
                                       m.Remove(o);
                                       v.push_back(o.index);
                                       break;
                                    }
                                 }
                              }
                           }
                        }
               }
         }
      };
      cube C;
      std::vector<bool> skip_needed;
     #endif
      int size;
      SynchronizerMPI_AMR *Synch_ptr;
      std::vector<GrowingVector<int>> positions;

      DuplicatesManager(int a_size, SynchronizerMPI_AMR &Synch)
      {
         size = a_size;
         //skip_needed.resize(size, false);
         positions.resize(size);

         Synch_ptr = &Synch;
      }

      void Add(int r, int index) { positions[r].push_back(index); }

      void RemoveDuplicates(int r, std::vector<Interface> &f, int &offset, int &total_size, int NC,
                            std::vector<InterfaceInfo> &fInfo)
      {
        #if 0
         std::vector<int> remEl;

         /*------------->*/ Clock.start(22, "SynchronizerMPI_AMR : RemoveDuplicates fill cube");
         C.clear();
         for (size_t i = 0; i < positions[r].size(); i++)
         {
            C.compass[f[positions[r][i]].icode[0]].push_back(Synch_ptr->SM.DetermineStencil(f[positions[r][i]]));
            C.compass[f[positions[r][i]].icode[0]].back().index = positions[r][i];
            C.compass[f[positions[r][i]].icode[0]].back().avg_down = (f[positions[r][i]].infos[0]->level > f[positions[r][i]].infos[1]->level);

            if (!skip_needed[r]) skip_needed[r] = f[positions[r][i]].CoarseStencil;
         }
         /*------------->*/ Clock.finish(22);

         /*------------->*/ Clock.start(23, "SynchronizerMPI_AMR : RemoveDuplicates __needed");
         if (!skip_needed[r])
         {
            C.__needed(remEl);
            for (size_t k = 0; k < remEl.size(); k++)
            {
               fInfo[remEl[k]].ToBeKept = false;
            }
         }
         /*------------->*/ Clock.finish(23);

         /*------------->*/ Clock.start(24, "SynchronizerMPI_AMR : Gather UnPackInfos");
         for (auto &i : C.keepEl())
        #else

         std::vector<bool> KEEP (positions[r].size(),true);
         std::vector< std::vector<int> > removed(positions[r].size());

         for (size_t i = 0; i < positions[r].size(); i++)
         {
            int k     = positions[r][i];
            bool checkDuplicates = KEEP[i] && ( f[k].infos[0]->level == f[k].infos[1]->level ) && (f[k].CoarseStencil == false);
            if (!checkDuplicates) continue;

            const int code[3] = {f[k].icode[1] % 3      - 1, 
                                (f[k].icode[1] / 3) % 3 - 1,
                                (f[k].icode[1] / 9) % 3 - 1};

            if (abs(code[0]) + abs(code[1]) + abs(code[2]) == 1)
            {
                for (size_t i1 = 0; i1 < positions[r].size(); i1++)
                {
                    int k1     = positions[r][i1];
                    bool checkDuplicates1 = KEEP[i1] && ( f[k1].infos[0]->level == f[k1].infos[1]->level ) && (f[k1].CoarseStencil == false);
                    if (!checkDuplicates1) continue;
                    if (i1 == i)           continue;
                    const int code1[3] = {f[k1].icode[1] % 3      - 1, 
                                         (f[k1].icode[1] / 3) % 3 - 1,
                                         (f[k1].icode[1] / 9) % 3 - 1};
                    if (abs(code1[0]) + abs(code1[1]) + abs(code1[2]) > 1 )
                    {
                    //    if (code1[0] == code[0] || code1[1] == code[1] || code1[2] == code[2])
                    //    {
                    //        KEEP[i1] = false;
                    //        removed[i].push_back(k1);
                    //        fInfo[k1].ToBeKept = false;
                    //    }
                    }
                }
            }
         }

         for (size_t i = 0; i < positions[r].size(); i++)
         #endif
         {
            int k     = positions[r][i];
           
            if (!KEEP[i]) continue;

            int L[3]  = {0, 0, 0};
            int Lc[3] = {0, 0, 0};

            const int code[3] = {f[k].icode[1] % 3      - 1, 
                                (f[k].icode[1] / 3) % 3 - 1,
                                (f[k].icode[1] / 9) % 3 - 1};

            Synch_ptr->SM.DetermineStencilLength(f[k].infos[0]->level,
                                                 f[k].infos[1]->level,
                                                 f[k].icode[1], 
                                                 &L[0]);
            int V  = L[0] * L[1] * L[2];
            int Vc = 0;

            total_size += V;
            if (f[k].CoarseStencil)
            {
               Synch_ptr->SM.CoarseStencilLength(&code[0], &Lc[0]);
               Vc = Lc[0] * Lc[1] * Lc[2];
               total_size += Vc;
            }

            UnPackInfo info = {offset,
                               L[0],
                               L[1],
                               L[2],
                               0,
                               0,
                               0,
                               L[0],
                               L[1],
                               -1,
                               0,
                               0,
                               0,
                               0,
                               0,
                               f[k].infos[0]->level,
                               f[k].infos[0]->Z,
                               f[k].icode[1],
                               f[k].infos[1]->blockID};

            fInfo[k].dis = offset;
            offset += V * NC;

            if (f[k].CoarseStencil)
            {
               offset += Vc * NC;
               info.CoarseVersionOffset = V * NC;
               info.CoarseVersionLX     = Lc[0];
               info.CoarseVersionLY     = Lc[1];
            }

            Synch_ptr->UnpacksManager.manyUnpacks[r].push_back(info);

            for (size_t kk = 0; kk < removed[i].size(); kk++)
            {
               int remEl1 = removed[i][kk];

               Synch_ptr->SM.DetermineStencilLength(f[remEl1].infos[0]->level, f[remEl1].infos[1]->level, f[remEl1].icode[1], &L[0]);

               int srcx, srcy, srcz;

               Synch_ptr->SM.__FixDuplicates(f[k], f[remEl1], info.lx, info.ly, info.lz, L[0], L[1],
                                             L[2], srcx, srcy, srcz);

               int Csrcx = 0;
               int Csrcy = 0;
               int Csrcz = 0;

               Synch_ptr->UnpacksManager.manyUnpacks[r].push_back(
                   {info.offset, L[0], L[1], L[2], srcx, srcy, srcz, info.LX, info.LY,
                    info.CoarseVersionOffset, info.CoarseVersionLX, info.CoarseVersionLY, Csrcx,
                    Csrcy, Csrcz, f[remEl1].infos[0]->level, f[remEl1].infos[0]->Z,
                    f[remEl1].icode[1], f[remEl1].infos[1]->blockID});

               fInfo[remEl1].dis = info.offset;
            }
        #if 0
            for (size_t kk = 0; kk < (*i).removedIndices.size(); kk++)
            {
               int remEl1 = i->removedIndices[kk];

               Synch_ptr->SM.DetermineStencilLength(
                   f[remEl1].infos[0]->level, f[remEl1].infos[1]->level, f[remEl1].icode[1], &L[0]);

               int srcx, srcy, srcz;

               Synch_ptr->SM.__FixDuplicates(f[k], f[remEl1], info.lx, info.ly, info.lz, L[0], L[1],
                                             L[2], srcx, srcy, srcz);

               int Csrcx = 0;
               int Csrcy = 0;
               int Csrcz = 0;

               if (f[k].CoarseStencil)
               {
                  Synch_ptr->SM.__FixDuplicates2(f[k], f[remEl1], Csrcx, Csrcy, Csrcz);
               }

               Synch_ptr->UnpacksManager.manyUnpacks[r].push_back(
                   {info.offset, L[0], L[1], L[2], srcx, srcy, srcz, info.LX, info.LY,
                    info.CoarseVersionOffset, info.CoarseVersionLX, info.CoarseVersionLY, Csrcx,
                    Csrcy, Csrcz, f[remEl1].infos[0]->level, f[remEl1].infos[0]->Z,
                    f[remEl1].icode[1], f[remEl1].infos[1]->blockID});

               fInfo[remEl1].dis = info.offset;
            }
       #endif
         }
         ///*------------->*/ Clock.finish(24);
      }

   };
#else
    struct DuplicatesManager
    {
        struct cube //could be more efficient, fix later
        {
            GrowingVector <MyRange> compass [27];

            void clear()
            {
                for (int i=0;i<27;i++) compass[i].clear();
            }

            void EraseAll()
            {
                for (int i=0;i<27;i++) compass[i].EraseAll();
            }
     
            cube(){}
        
            std::vector<MyRange *> keepEl()
            {
                std::vector<MyRange *> retval;
            
                for (int i=0; i<27; i++)
                {
                    for (size_t j=0; j< compass[i].size() ; j++)
                    {
                        if (compass[i][j].needed)
                            retval.push_back(&compass[i][j]);
                    }
                }    
                return retval;
            }
        
            void __needed(std::vector<int> & v)
            {
                static constexpr std::array <int,3> faces_and_edges [18] = {
                    {0,1,1},{2,1,1},{1,0,1},{1,2,1},{1,1,0},{1,1,2},
                    
                    {0,0,1},{0,2,1},{2,0,1},{2,2,1},{1,0,0},{1,0,2},
                    {1,2,0},{1,2,2},{0,1,0},{0,1,2},{2,1,0},{2,1,2}};
    
                for (auto & f:faces_and_edges)
                if ( compass[f[0] + f[1]*3 + f[2]*9].size() != 0 )
                {
                    auto & me = compass[f[0] + f[1]*3 + f[2]*9];

                    bool needme = false;
                    auto & other1 = compass[f[0] + f[1]*3 + f[2]*9];       
                    for (size_t j1=0; j1<other1.size(); j1++)
                    { 
                        auto & o = other1[j1];
                        if (o.needed)
                        {
                            needme = true;
    
                            for (size_t j=0; j<me.size(); j++)
                            {
                             auto & m = me[j];
                             if (m.needed && m.contains(o) )
                             {
                                o.needed = false;
                                m.removedIndices.push_back(o.index);
                                m.Remove(o);
                                v.push_back(o.index);
                                break;
                             }
                            }
                        }
                    }             
                    if (!needme) continue;
     
                    int imax = (f[0] == 1) ? 2:f[0];
                    int imin = (f[0] == 1) ? 0:f[0]; 
                    
                    int jmax = (f[1] == 1) ? 2:f[1];
                    int jmin = (f[1] == 1) ? 0:f[1]; 
                    
                    int kmax = (f[2] == 1) ? 2:f[2];
                    int kmin = (f[2] == 1) ? 0:f[2];  
    
                    for (int k=kmin;k<=kmax;k++)
                    for (int j=jmin;j<=jmax;j++)
                    for (int i=imin;i<=imax;i++)
                    {
                        if (i==f[0] && j==f[1] && k==f[2]) continue;
                        auto & other = compass[i + j*3 + k*9];       
    
                        for (size_t j1=0; j1<other.size(); j1++)
                        { 
                            auto & o = other[j1];
                            if (o.needed)
                            {
                               for (size_t k1=0; k1<me.size(); k1++)
                               {  
                                    auto & m = me[k1];
                                    if (m.needed && m.contains(o) )
                                    {
                                        o.needed = false;
                                        m.removedIndices.push_back(o.index);
                                        m.Remove(o);
                                        v.push_back(o.index);
                                        break;
                                    }
                               }
                            }
                        }
                    }
                } 
            }
        };
      
        cube C;

        std::vector<bool> skip_needed;
        int size;

        SynchronizerMPI_AMR * Synch_ptr;

        //std::vector<std::vector <int>> positions;
        std::vector< GrowingVector <int>> positions;


        DuplicatesManager(int a_size, SynchronizerMPI_AMR & Synch)
        {
            size = a_size;
            skip_needed.resize(size,false);
            positions.resize(size);

            Synch_ptr = & Synch;
        }

        void Add(int r,int index)
        {
            positions[r].push_back(index);
        }


        void RemoveDuplicates(int r, std::vector<Interface> & f, int & offset, int & total_size, int NC)
        {
            std::vector<int> remEl;

            
            /*------------->*/Clock.start(22,"SynchronizerMPI_AMR : RemoveDuplicates fill cube");
            C.clear();
            for (size_t i=0; i<positions[r].size();i++)
            {              
                C.compass[f[positions[r][i]].icode[0]].push_back(Synch_ptr->SM.DetermineStencil(f[positions[r][i]]));
                C.compass[f[positions[r][i]].icode[0]].back().index = positions[r][i];
                C.compass[f[positions[r][i]].icode[0]].back().avg_down = (f[positions[r][i]].infos[0]->level > f[positions[r][i]].infos[1]->level);

                if (!skip_needed[r]) skip_needed[r] = f[positions[r][i]].CoarseStencil;
            }
            /*------------->*/Clock.finish(22);


            /*------------->*/Clock.start(23,"SynchronizerMPI_AMR : RemoveDuplicates __needed");
            if (!skip_needed[r])
            {
                C.__needed(remEl);
                for (size_t k=0; k< remEl.size();k++)
                    f[remEl[k]].ToBeKept = false;
            }
            /*------------->*/Clock.finish(23); 
  


            /*------------->*/Clock.start(24,"SynchronizerMPI_AMR : Gather UnPackInfos");
            for (auto & i:C.keepEl())
            {
                int k = i->index;            
                int L [3] ={0,0,0};
                int Lc[3] ={0,0,0};
                 
                int code[3] = { f[k].icode[1]%3-1, (f[k].icode[1]/3)%3-1, (f[k].icode[1]/9)%3-1};

                Synch_ptr->SM.DetermineStencilLength(f[k].infos[0]->level,f[k].infos[1]->level,f[k].icode[1],&L[0]);

                int V = L[0]*L[1]*L[2];
                int Vc = 0;

                total_size+= V;
                if (f[k].CoarseStencil)
                {
                    Synch_ptr->SM.CoarseStencilLength(&code[0],&Lc[0]);
                    Vc = Lc[0]*Lc[1]*Lc[2];
                    total_size += Vc;
                }                    

                UnPackInfo info = {offset,L[0],L[1],L[2],0,0,0,L[0],L[1],-1, 0,0,0,0,0,f[k].infos[0]->level,f[k].infos[0]->Z,f[k].icode[1],f[k].infos[1]->blockID};
                
                f[k].dis = offset;
                offset += V*NC;

                if (f[k].CoarseStencil)
                {
                    offset += Vc*NC; 
                    info.CoarseVersionOffset = V*NC;                                       
                    info.CoarseVersionLX = Lc[0];
                    info.CoarseVersionLY = Lc[1];
                }                   
                    
                Synch_ptr->UnpacksManager.manyUnpacks[r].push_back(info);
                
                for (size_t kk=0; kk< (*i).removedIndices.size();kk++)
                {
                    int remEl1 = i->removedIndices[kk];

                    Synch_ptr->SM.DetermineStencilLength(f[remEl1].infos[0]->level,f[remEl1].infos[1]->level,f[remEl1].icode[1],&L[0]);
                        
                    int srcx, srcy, srcz;

                    Synch_ptr->SM.__FixDuplicates(f[k],f[remEl1], info.lx,info.ly,info.lz,L[0],L[1],L[2], srcx,srcy,srcz);

                    int Csrcx=0;
                    int Csrcy=0;
                    int Csrcz=0;

                    if (f[k].CoarseStencil)
                    {
                        Synch_ptr->SM.__FixDuplicates2(f[k],f[remEl1],Csrcx,Csrcy,Csrcz);
                    }

                    Synch_ptr->UnpacksManager.manyUnpacks[r].push_back({info.offset,L[0],L[1],L[2],srcx, srcy, srcz,info.LX,info.LY,
                    info.CoarseVersionOffset, info.CoarseVersionLX, info.CoarseVersionLY,
                    Csrcx, Csrcy, Csrcz,
                    f[remEl1].infos[0]->level,f[remEl1].infos[0]->Z,f[remEl1].icode[1],f[remEl1].infos[1]->blockID});
                    
                    f[remEl1].dis = info.offset;
                } 
            }
            /*------------->*/Clock.finish(24);
        }
    };
#endif

   inline BlockInfo &getBlockInfoAll(int m, int n) { return (*BlockInfoAll[m])[n]; }

   bool UseCoarseStencil(Interface &f)
   {
    //return true;
      BlockInfo &a = *f.infos[0];
      BlockInfo &b = *f.infos[1];

      if (a.level != b.level) return false;

      int imin[3]= {-1,-1,-1};
      int imax[3]= {+1,+1,+1};
      //for (int d = 0; d < 3; d++)
      //{
      //   imin[d] = (a.index[d] < b.index[d]) ? 0 : -1;
      //   imax[d] = (a.index[d] > b.index[d]) ? 0 : +1;
      //}

      bool retval = false;

      for (int i2 = imin[2]; i2 <= imax[2]; i2++)
         for (int i1 = imin[1]; i1 <= imax[1]; i1++)
            for (int i0 = imin[0]; i0 <= imax[0]; i0++)
            {
               int n = a.Znei_(i0, i1, i2);
               if ((getBlockInfoAll(a.level, n)).TreePos == CheckCoarser)
               {
                  retval = true;
                  break;
               }
            }
      return retval;
   }

   void AverageDownAndFill(Real *dst, const BlockInfo *const info, const int s[3], const int e[3],
                           const int code[3], const int *const selcomponents, const int NC,
                           const int gptfloats)
   {
      static const int nX = blocksize[0];
      static const int nY = blocksize[1];
      static const int nZ = blocksize[2];

      Real *src = (Real *)(*info).ptrBlock;

      const int xStep = (code[0] == 0) ? 2 : 1;
      const int yStep = (code[1] == 0) ? 2 : 1;
      const int zStep = (code[2] == 0) ? 2 : 1;

      int pos = 0;

      for (int iz = s[2]; iz < e[2]; iz += zStep)
      {
         const int ZZ = (abs(code[2]) == 1) ? 2 * (iz - code[2] * nZ) + min(0, code[2]) * nZ : iz;
         for (int iy = s[1]; iy < e[1]; iy += yStep)
         {
            const int YY =
                (abs(code[1]) == 1) ? 2 * (iy - code[1] * nY) + min(0, code[1]) * nY : iy;
            for (int ix = s[0]; ix < e[0]; ix += xStep)
            {
               const int XX =
                   (abs(code[0]) == 1) ? 2 * (ix - code[0] * nX) + min(0, code[0]) * nX : ix;

               for (int c = 0; c < NC; c++)
               {
                  int comp = selcomponents[c];

                  dst[pos] =
                      0.125 *
                      ((*(src + gptfloats * ((XX) + ((YY) + (ZZ)*nY) * nX) + comp)) +
                       (*(src + gptfloats * ((XX) + ((YY) + (ZZ + 1) * nY) * nX) + comp)) +
                       (*(src + gptfloats * ((XX) + ((YY + 1) + (ZZ)*nY) * nX) + comp)) +
                       (*(src + gptfloats * ((XX) + ((YY + 1) + (ZZ + 1) * nY) * nX) + comp)) +
                       (*(src + gptfloats * ((XX + 1) + ((YY) + (ZZ)*nY) * nX) + comp)) +
                       (*(src + gptfloats * ((XX + 1) + ((YY) + (ZZ + 1) * nY) * nX) + comp)) +
                       (*(src + gptfloats * ((XX + 1) + ((YY + 1) + (ZZ)*nY) * nX) + comp)) +
                       (*(src + gptfloats * ((XX + 1) + ((YY + 1) + (ZZ + 1) * nY) * nX) + comp)));
                  pos++;
               }
            }
         }
      }
   }

   void AverageDownAndFill2(Real *dst, const BlockInfo *const info, const int code[3],
                            const int *const selcomponents, const int NC, const int gptfloats)
   {
      static const int nX = blocksize[0];
      static const int nY = blocksize[1];
      static const int nZ = blocksize[2];

      const int eC[3] = {(stencil.ex) / 2 + Cstencil.ex, (stencil.ey) / 2 + Cstencil.ey,
                         (stencil.ez) / 2 + Cstencil.ez};
      const int sC[3] = {(stencil.sx - 1) / 2 + Cstencil.sx, (stencil.sy - 1) / 2 + Cstencil.sy,
                         (stencil.sz - 1) / 2 + Cstencil.sz};

      const int s[3] = {code[0] < 1 ? (code[0] < 0 ? sC[0] : 0) : nX / 2,
                        code[1] < 1 ? (code[1] < 0 ? sC[1] : 0) : nY / 2,
                        code[2] < 1 ? (code[2] < 0 ? sC[2] : 0) : nZ / 2};

      const int e[3] = {code[0] < 1 ? (code[0] < 0 ? 0 : nX / 2) : nX / 2 + eC[0] - 1,
                        code[1] < 1 ? (code[1] < 0 ? 0 : nY / 2) : nY / 2 + eC[1] - 1,
                        code[2] < 1 ? (code[2] < 0 ? 0 : nZ / 2) : nZ / 2 + eC[2] - 1};

      Real *src = (Real *)(*info).ptrBlock;

      int pos = 0;

      for (int iz = s[2]; iz < e[2]; iz++)
      {
         const int ZZ = 2 * (iz - s[2]) + s[2] + max(code[2], 0) * nZ / 2 - code[2] * nZ +
                        min(0, code[2]) * (e[2] - s[2]);
         for (int iy = s[1]; iy < e[1]; iy++)
         {
            const int YY = 2 * (iy - s[1]) + s[1] + max(code[1], 0) * nY / 2 - code[1] * nY +
                           min(0, code[1]) * (e[1] - s[1]);
            for (int ix = s[0]; ix < e[0]; ix++)
            {
               const int XX = 2 * (ix - s[0]) + s[0] + max(code[0], 0) * nX / 2 - code[0] * nX +
                              min(0, code[0]) * (e[0] - s[0]);

               for (int c = 0; c < NC; c++)
               {
                  int comp = selcomponents[c];

                  dst[pos] =
                      0.125 *
                      ((*(src + gptfloats * ((XX) + ((YY) + (ZZ)*nY) * nX) + comp)) +
                       (*(src + gptfloats * ((XX) + ((YY) + (ZZ + 1) * nY) * nX) + comp)) +
                       (*(src + gptfloats * ((XX) + ((YY + 1) + (ZZ)*nY) * nX) + comp)) +
                       (*(src + gptfloats * ((XX) + ((YY + 1) + (ZZ + 1) * nY) * nX) + comp)) +
                       (*(src + gptfloats * ((XX + 1) + ((YY) + (ZZ)*nY) * nX) + comp)) +
                       (*(src + gptfloats * ((XX + 1) + ((YY) + (ZZ + 1) * nY) * nX) + comp)) +
                       (*(src + gptfloats * ((XX + 1) + ((YY + 1) + (ZZ)*nY) * nX) + comp)) +
                       (*(src + gptfloats * ((XX + 1) + ((YY + 1) + (ZZ + 1) * nY) * nX) + comp)));

                  pos++;
               }
            }
         }
      }
   }

   void DefineInterfaces()
   {
      if (define_neighbors)
      {
         Neighbors.clear();
         inner_blocks.clear();
         halo_blocks.clear();
         interface_ranks_and_positions.clear();
         lengths.clear();
         for (int r = 0; r < size; r++) send_interfaces[r].clear();
      }

      for (int r = 0; r < size; r++) send_interfaces_infos[r].clear();
      std::vector<int> offsets(size, 0);
      for (int r = 0; r < size; r++)
      {
         send_buffer_size[r] = 0;
      }
      UnpacksManager.clear();

      if (define_neighbors)
         for (int i = 0; i < (int)myInfos_size; i++)
         {
            BlockInfo &info    = myInfos[i];
            info.halo_block_id = -1;

            int aux          = 1 << info.level;
            const bool xskin = info.index[0] == 0 || info.index[0] == blocksPerDim[0] * aux - 1;
            const bool yskin = info.index[1] == 0 || info.index[1] == blocksPerDim[1] * aux - 1;
            const bool zskin = info.index[2] == 0 || info.index[2] == blocksPerDim[2] * aux - 1;
            const int xskip  = info.index[0] == 0 ? -1 : 1;
            const int yskip  = info.index[1] == 0 ? -1 : 1;
            const int zskip  = info.index[2] == 0 ? -1 : 1;

            bool isInner = true;

            std::vector<int> ToBeChecked;
            bool Coarsened = false;

            int l = 0;
            /*------------->*/ Clock.start(15, "DefineInterfaces: icode loop");
            for (int icode = 0; icode < 27; icode++)
            {
               if (icode == 1 * 1 + 3 * 1 + 9 * 1) continue;
               const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, (icode / 9) % 3 - 1};
               if (!xperiodic && code[0] == xskip && xskin) continue;
               if (!yperiodic && code[1] == yskip && yskin) continue;
               if (!zperiodic && code[2] == zskip && zskin) continue;

                //if (!stencil.tensorial && !Cstencil.tensorial && abs(code[0])+abs(code[1])+abs(code[2])>1) continue;

               BlockInfo &infoNei =
                   getBlockInfoAll(info.level, info.Znei_(code[0], code[1], code[2]));

               if (infoNei.TreePos == Exists && infoNei.myrank != rank)
               {
                  if (isInner)
                     interface_ranks_and_positions.resize(interface_ranks_and_positions.size() + 1);
                  isInner    = false;
                  int icode2 = (-code[0] + 1) + (-code[1] + 1) * 3 + (-code[2] + 1) * 9;
                  send_interfaces[infoNei.myrank].push_back(Interface(info, infoNei, icode, icode2));
                  ToBeChecked.push_back(infoNei.myrank);
                  ToBeChecked.push_back((int)send_interfaces[infoNei.myrank].size() - 1);
                  Neighbors.insert(infoNei.myrank);
                  interface_ranks_and_positions.back().push_back({infoNei.myrank, (int)send_interfaces[infoNei.myrank].size() - 1});
                  l++;
               }

               else if (infoNei.TreePos == CheckCoarser)
               {
                  Coarsened = true;
                  // int nCoarse = infoNei.Z /8; // this does not work for Hilbert
                  int nCoarse = infoNei.Zparent;

                  BlockInfo &infoNeiCoarser = getBlockInfoAll(infoNei.level - 1, nCoarse);
                  if (infoNeiCoarser.myrank != rank)
                  {
                     if (isInner)
                        interface_ranks_and_positions.resize(interface_ranks_and_positions.size() + 1);
                     isInner         = false;
                     int code2[3]    = {-code[0], -code[1], -code[2]};
                     int icode2      = (code2[0] + 1) + (code2[1] + 1) * 3 + (code2[2] + 1) * 9;
                     BlockInfo &test = getBlockInfoAll(
                         infoNeiCoarser.level, infoNeiCoarser.Znei_(code2[0], code2[1], code2[2]));
                     if (info.index[0] / 2 == test.index[0] && info.index[1] / 2 == test.index[1] &&
                         info.index[2] / 2 == test.index[2])
                     {
                        send_interfaces[infoNeiCoarser.myrank].push_back(
                            Interface(info, infoNeiCoarser, icode, icode2));
                        interface_ranks_and_positions.back().push_back(
                            {infoNeiCoarser.myrank,
                             (int)send_interfaces[infoNeiCoarser.myrank].size() - 1});
                     }
                     Neighbors.insert(infoNeiCoarser.myrank);
                     l++;
                  }
               }
               else if (infoNei.TreePos == CheckFiner)
               {
                  int Bstep = 1;
                  if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 2)) Bstep = 3; // edge
                  else if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 3))
                     Bstep = 4; // corner

                  for (int B = 0; B <= 3;B += Bstep) // loop over blocks that make up face/edge/corner (4/2/1 blocks)
                  {
                     const int temp = (abs(code[0]) == 1) ? (B % 2) : (B / 2);

                     int nFine1 =
                         infoNei.Zchild[max(code[0], 0) + (B % 2) * max(0, 1 - abs(code[0]))]
                                       [max(code[1], 0) + temp * max(0, 1 - abs(code[1]))]
                                       [max(code[2], 0) + (B / 2) * max(0, 1 - abs(code[2]))];
                     int nFine = getBlockInfoAll(infoNei.level + 1, nFine1)
                                     .Znei_(-code[0], -code[1], -code[2]);

                     BlockInfo &infoNeiFiner = getBlockInfoAll(infoNei.level + 1, nFine);
                     if (infoNeiFiner.myrank != rank)
                     {
                        if (isInner)
                           interface_ranks_and_positions.resize(interface_ranks_and_positions.size() + 1);
                        isInner    = false;
                        int icode2 = (-code[0] + 1) + (-code[1] + 1) * 3 + (-code[2] + 1) * 9;
                        send_interfaces[infoNeiFiner.myrank].push_back(Interface(info, infoNeiFiner, icode, icode2));
                        interface_ranks_and_positions.back().push_back({infoNeiFiner.myrank,(int)send_interfaces[infoNeiFiner.myrank].size() - 1});
                        Neighbors.insert(infoNeiFiner.myrank);
                        l++;
                        if (Bstep == 3) // if I'm filling an edge then I'm also filling a corner
                        {
                           int code3[3];
                           code3[0]   = (code[0] == 0) ? (B == 0 ? 1 : -1) : -code[0];
                           code3[1]   = (code[1] == 0) ? (B == 0 ? 1 : -1) : -code[1];
                           code3[2]   = (code[2] == 0) ? (B == 0 ? 1 : -1) : -code[2];
                           int icode3 = (code3[0] + 1) + (code3[1] + 1) * 3 + (code3[2] + 1) * 9;
                           send_interfaces[infoNeiFiner.myrank].push_back(
                               Interface(info, infoNeiFiner, icode, icode3));
                           interface_ranks_and_positions.back().push_back(
                               {infoNeiFiner.myrank,
                                (int)send_interfaces[infoNeiFiner.myrank].size() - 1});
                        }
                        else if (Bstep == 1) // if I'm filling a face then I'm also filling two
                                             // edges and a corner
                        {
                           int code3[3];
                           int code4[3];
                           int code5[3];
                           int d0, d1, d2;
                           assert(code[0] != 0 || code[1] != 0 || code[2] != 0);
                           assert(abs(code[0]) + abs(code[1]) + abs(code[2]) == 1);
                           if (code[0] != 0)
                           {
                              d0 = 0;
                              d1 = 1;
                              d2 = 2;
                           }
                           else if (code[1] != 0)
                           {
                              d0 = 1;
                              d1 = 0;
                              d2 = 2;
                           }
                           else /*if (code[2]!=0)*/
                           {
                              d0 = 2;
                              d1 = 0;
                              d2 = 1;
                           }
                           code3[d0] = -code[d0];
                           code4[d0] = -code[d0];
                           code5[d0] = -code[d0];
                           if (B == 0)
                           {
                              code3[d1] = 1;
                              code3[d2] = 1;
                              code4[d1] = 1;
                              code4[d2] = 0;
                              code5[d1] = 0;
                              code5[d2] = 1;
                           }
                           else if (B == 1)
                           {
                              code3[d1] = -1;
                              code3[d2] = 1;
                              code4[d1] = -1;
                              code4[d2] = 0;
                              code5[d1] = 0;
                              code5[d2] = 1;
                           }
                           else if (B == 2)
                           {
                              code3[d1] = 1;
                              code3[d2] = -1;
                              code4[d1] = 1;
                              code4[d2] = 0;
                              code5[d1] = 0;
                              code5[d2] = -1;
                           }
                           else // if (B==3)
                           {
                              code3[d1] = -1;
                              code3[d2] = -1;
                              code4[d1] = -1;
                              code4[d2] = 0;
                              code5[d1] = 0;
                              code5[d2] = -1;
                           }
                           int icode3 = (code3[0] + 1) + (code3[1] + 1) * 3 + (code3[2] + 1) * 9;
                           int icode4 = (code4[0] + 1) + (code4[1] + 1) * 3 + (code4[2] + 1) * 9;
                           int icode5 = (code5[0] + 1) + (code5[1] + 1) * 3 + (code5[2] + 1) * 9;
                           send_interfaces[infoNeiFiner.myrank].push_back(
                               Interface(info, infoNeiFiner, icode, icode3));
                           interface_ranks_and_positions.back().push_back(
                               {infoNeiFiner.myrank,
                                (int)send_interfaces[infoNeiFiner.myrank].size() - 1});
                           send_interfaces[infoNeiFiner.myrank].push_back(
                               Interface(info, infoNeiFiner, icode, icode4));
                           interface_ranks_and_positions.back().push_back(
                               {infoNeiFiner.myrank,
                                (int)send_interfaces[infoNeiFiner.myrank].size() - 1});
                           send_interfaces[infoNeiFiner.myrank].push_back(
                               Interface(info, infoNeiFiner, icode, icode5));
                           interface_ranks_and_positions.back().push_back(
                               {infoNeiFiner.myrank,
                                (int)send_interfaces[infoNeiFiner.myrank].size() - 1});
                        }
                     }
                  }
               }
            } // icode = 0,...,26

            /*------------->*/ Clock.finish(15);

            if (isInner)
            {
               info.halo_block_id = -1;
               inner_blocks.push_back(&info);
            }
            else
            {
               info.halo_block_id = halo_blocks.size();
               halo_blocks.push_back(&info);
               lengths.push_back(l);
               if (Coarsened)
                  for (size_t j = 0; j < ToBeChecked.size(); j += 2)
                     send_interfaces[ToBeChecked[j]][ToBeChecked[j + 1]].CoarseStencil =
                         UseCoarseStencil(send_interfaces[ToBeChecked[j]][ToBeChecked[j + 1]]);
            }

            getBlockInfoAll(info.level, info.Z).halo_block_id = info.halo_block_id;
         } // i-loop

      for (int r = 0; r < size; r++) send_interfaces_infos[r].resize(send_interfaces[r].size());

      DuplicatesManager DM(size, *(this));
      for (size_t i = 0; i < interface_ranks_and_positions.size(); i++)
      {
         BlockInfo &info = *halo_blocks[i];

         std::vector<std::pair<int, int>> &rp = interface_ranks_and_positions[i];

         UnpacksManager.MapOfInfos.push_back({info.blockID, info.halo_block_id});

         for (int r = 0; r < size; r++) DM.positions[r].clear();
         for (auto &_rp : rp)
         {
            DM.Add(_rp.first, _rp.second);
         }
         /*------------->*/ Clock.start(18, "DefineInterfaces: Duplicates handling");
         for (int r = 0; r < size; r++)
            DM.RemoveDuplicates(r, send_interfaces[r].v, offsets[r], send_buffer_size[r],
                                stencil.selcomponents.size(), send_interfaces_infos[r]);
         /*------------->*/ Clock.finish(18);
      }
      
      UnpacksManager._allocate(halo_blocks.size(), lengths.data());
   }

 public:
   static 
   std::set<int> Neighbors;
   bool define_neighbors;
   static 
   std::vector<std::vector<std::pair<int, int>>> interface_ranks_and_positions;
   static 
   std::vector<size_t> lengths;

   SynchronizerMPI_AMR(StencilInfo a_stencil, StencilInfo a_Cstencil, MPI_Comm a_comm,
                       const bool a_periodic[3], const int a_levelMax, const int a_nx,
                       const int a_ny, const int a_nz, const int a_bx, const int a_by,
                       const int a_bz)//, SpaceFillingCurve & sfc)
       : stencil(a_stencil), Cstencil(a_Cstencil), comm(a_comm), xperiodic(a_periodic[0]),
         yperiodic(a_periodic[1]), zperiodic(a_periodic[2]), levelMax(a_levelMax),
         SM(a_stencil, a_Cstencil, a_nx, a_ny, a_nz, a_bx, a_by, a_bz)
   {
      MPI_Comm_rank(comm, &rank);
      MPI_Comm_size(comm, &size);
      blocksize[0]    = a_nx;
      blocksize[1]    = a_ny;
      blocksize[2]    = a_nz;
      blocksPerDim[0] = a_bx;
      blocksPerDim[1] = a_by;
      blocksPerDim[2] = a_bz;
      send_interfaces.resize(size);
      #if 1 //new
      send_interfaces_infos.resize(size);
      #endif
      send_packinfos.resize(size);
      send_buffer_size.resize(size);
      recv_buffer_size.resize(size);
      send_buffer.resize(size);
      recv_buffer.resize(size);


      UnpacksManager.comm = comm;
   }

   std::vector<BlockInfo *> avail_inner() { return inner_blocks; }

   std::vector<BlockInfo *> avail_halo()
   {
      MPI_Waitall(recv_requests.size(), recv_requests.data(), MPI_STATUSES_IGNORE);
      return halo_blocks;
   }



   std::vector<MPI_Request> size_requests;
   std::vector<int> ss1;
   std::vector<int> ss;
   void _Setup(BlockInfo *a_myInfos, 
               size_t a_myInfos_size,
               std::vector<std::vector<BlockInfo>> &a_BlockInfoAll, 
               const int timestamp,
               const bool a_define_neighbors = false)
   {
      define_neighbors = a_define_neighbors;

      myInfos_size = a_myInfos_size;
      myInfos      = a_myInfos;
      BlockInfoAll.resize(levelMax);
      for (int m = 0; m < levelMax; m++) BlockInfoAll[m] = &a_BlockInfoAll[m];
      const int NC = stencil.selcomponents.size();

      if (rank == 0) 
      {
        std::cout << " Calling _Setup with define_neighbors=" << define_neighbors << "\n";
      }
        

      /*------------->*/ Clock.start(12, "DefineInterfaces total time");
      DefineInterfaces();
      /*------------->*/ Clock.finish(12);


      /*------------->*/ Clock.start(13, "Synchronizer: send/recv sizes");
      for (int r = 0; r < size; r++) 
        send_buffer[r].resize(send_buffer_size[r] * NC, 666.0);
      recv_buffer_size.resize(size, 0);
      ss1.resize(size, 0);
      ss.resize(size, 0);
      size_requests.clear();
      size_requests.resize(2 * Neighbors.size());
      int k = 0;
      for (auto r : Neighbors)
      {
         ss1[r] = send_buffer[r].size();
         MPI_Irecv(&ss [r], 1, MPI_INT, r, timestamp, comm, &size_requests[k]    );
         MPI_Isend(&ss1[r], 1, MPI_INT, r, timestamp, comm, &size_requests[k + 1]);
         k += 2;
      }
      /*------------->*/ Clock.finish(13);

      UnpacksManager.SendPacks(Neighbors, timestamp);

      ToBeAveragedDown.resize(size);

      for (int r = 0; r < size; r++)
      {
         send_packinfos[r].clear();
         ToBeAveragedDown[r].clear();

         MPI_Waitall(send_requests.size(), send_requests.data(), MPI_STATUSES_IGNORE);

         for (int i = 0; i < (int)send_interfaces[r].size(); i++)
         {
            Interface &f         = send_interfaces[r][i];
            
            InterfaceInfo &fInfo = send_interfaces_infos[r][i];
            if (!fInfo.ToBeKept) continue;

            if (f.infos[0]->level <= f.infos[1]->level)
            {
               MyRange range = SM.DetermineStencil(f);
               int V = (range.ex - range.sx) * (range.ey - range.sy) * (range.ez - range.sz);
               send_packinfos[r].push_back({(Real *)f.infos[0]->ptrBlock,
                                            &send_buffer[r][fInfo.dis], range.sx, range.sy,
                                            range.sz, range.ex, range.ey, range.ez});

               if (f.CoarseStencil)
               {
                  ToBeAveragedDown[r].push_back(i);
                  ToBeAveragedDown[r].push_back(fInfo.dis + V * NC);
               }
            }
            else // receiver is coarser, so sender averages down data first
            {
               ToBeAveragedDown[r].push_back(i);
               ToBeAveragedDown[r].push_back(fInfo.dis);
            }
         }
      }
   }

   void sync(unsigned int gptfloats, MPI_Datatype MPIREAL, const int timestamp)
   {
      static const int nX = blocksize[0];
      static const int nY = blocksize[1];
      static const int nZ = blocksize[2];

      std::vector<int> &selcomponents = stencil.selcomponents;
      std::sort(selcomponents.begin(), selcomponents.end());
      const int NC = selcomponents.size();

      // Pack data
      /*------------->*/ Clock.start(25, "Synchronizer: data packing");
      for (int r = 0; r < size; r++)
      {
         if (ToBeAveragedDown[r].size() == 0 && send_buffer_size[r] == 0) continue;

         #pragma omp parallel
         {
            #pragma omp for schedule(runtime)
            for (size_t j = 0; j < ToBeAveragedDown[r].size(); j += 2)
            {
               int i              = ToBeAveragedDown[r][j];
               int d              = ToBeAveragedDown[r][j + 1];
               Interface &f       = send_interfaces[r][i];
               const int code[3]  = {f.icode[0] % 3 - 1, (f.icode[0] / 3) % 3 - 1,
                                    (f.icode[0] / 9) % 3 - 1};
               const int code0[3] = {-code[0], -code[1], -code[2]};
               if (f.CoarseStencil)
               {
                  AverageDownAndFill2(send_buffer[r].data() + d, f.infos[0], code0,
                                      &selcomponents[0], NC, gptfloats);
               }
               else
               {
                  const int s[3] = {code0[0] < 1 ? (code0[0] < 0 ? stencil.sx : 0) : nX,
                                    code0[1] < 1 ? (code0[1] < 0 ? stencil.sy : 0) : nY,
                                    code0[2] < 1 ? (code0[2] < 0 ? stencil.sz : 0) : nZ};
                  const int e[3] = {code0[0] < 1 ? (code0[0] < 0 ? 0 : nX) : nX + stencil.ex - 1,
                                    code0[1] < 1 ? (code0[1] < 0 ? 0 : nY) : nY + stencil.ey - 1,
                                    code0[2] < 1 ? (code0[2] < 0 ? 0 : nZ) : nZ + stencil.ez - 1};
                  AverageDownAndFill(send_buffer[r].data() + d, f.infos[0], s, e, code0,
                                     &selcomponents[0], NC, gptfloats);
               }
            }

            if (send_buffer_size[r] != 0)
            {
               const int N = send_packinfos[r].size();
               #pragma omp for schedule(runtime)
               for (int i = 0; i < N; i++)
               {
                  PackInfo &info = send_packinfos[r][i];
                  pack(info.block, info.pack, gptfloats, &selcomponents.front(), NC, info.sx,
                       info.sy, info.sz, info.ex, info.ey, info.ez, blocksize[0], blocksize[1]);
               }
            }
         }
      }
      /*------------->*/ Clock.finish(25);

      /*------------->*/ Clock.start(13, "Synchronizer: send/recv sizes");
      MPI_Waitall(size_requests.size(), size_requests.data(), MPI_STATUSES_IGNORE);
      for (auto r : Neighbors)
      {
         recv_buffer_size[r] = ss[r] / NC;
         recv_buffer[r].resize(recv_buffer_size[r] * NC, 777.0);
      }
      /*------------->*/ Clock.finish(13);

      send_requests.clear();
      recv_requests.clear();
      for (auto r : Neighbors)
      {
         if (recv_buffer_size[r] > 0)
         {
            recv_requests.resize(recv_requests.size() + 1);
            MPI_Irecv(&recv_buffer[r][0], recv_buffer_size[r] * NC, MPIREAL, r, timestamp, comm,
                      &recv_requests.back());
         }
      }
      for (auto r : Neighbors)
      {
         if (send_buffer_size[r] > 0)
         {
            send_requests.resize(send_requests.size() + 1);
            MPI_Isend(&send_buffer[r][0], send_buffer_size[r] * NC, MPIREAL, r, timestamp, comm,
                      &send_requests.back());
         }
      }

      /*------------->*/ Clock.start(27, "Synchronizer: MapIDs");
      UnpacksManager.MapIDs();
      /*------------->*/ Clock.finish(27);
   }

   StencilInfo getstencil() const { return stencil; }

   StencilInfo getCstencil() const { return Cstencil; }

   void fetch(const BlockInfo &info, const int gptfloats, const size_t Length[3],
              const size_t CLength[3], const size_t ElemsPerSlice[2], Real *cacheBlock,
              Real *coarseBlock)
   {
      static const int nX = blocksize[0];
      static const int nY = blocksize[1];
      static const int nZ = blocksize[2];

      int id = info.halo_block_id;
      if (id < 0) return;

      UnPackInfo **unpacks = UnpacksManager.unpacks[id].data();

      for (size_t jj = 0; jj < UnpacksManager.sizes[id]; jj++)
      {
         UnPackInfo &unpack = *unpacks[jj];

         const int code[3] = {unpack.icode % 3 - 1, (unpack.icode / 3) % 3 - 1,
                              (unpack.icode / 9) % 3 - 1};

         BlockInfo &other = getBlockInfoAll(unpack.level, unpack.Z);

         const int s[3] = {code[0] < 1 ? (code[0] < 0 ? stencil.sx : 0) : nX,
                           code[1] < 1 ? (code[1] < 0 ? stencil.sy : 0) : nY,
                           code[2] < 1 ? (code[2] < 0 ? stencil.sz : 0) : nZ};
         const int e[3] = {code[0] < 1 ? (code[0] < 0 ? 0 : nX) : nX + stencil.ex - 1,
                           code[1] < 1 ? (code[1] < 0 ? 0 : nY) : nY + stencil.ey - 1,
                           code[2] < 1 ? (code[2] < 0 ? 0 : nZ) : nZ + stencil.ez - 1};

         if (other.level == info.level)
         {
            Real *dst = cacheBlock + ((s[2] - stencil.sz) * ElemsPerSlice[0] +
                                      (s[1] - stencil.sy) * Length[0] + s[0] - stencil.sx) *
                                         gptfloats;

            unpack_subregion<Real>(&recv_buffer[other.myrank][unpack.offset], &dst[0], gptfloats,
                                   &stencil.selcomponents[0], stencil.selcomponents.size(),
                                   unpack.srcxstart, unpack.srcystart, unpack.srczstart, unpack.LX,
                                   unpack.LY, 0, 0, 0, unpack.lx, unpack.ly, unpack.lz, Length[0],
                                   Length[1], Length[2]);

            if (unpack.CoarseVersionOffset >= 0)
            {
               const int sC[3] = {(stencil.sx - 1) / 2 + Cstencil.sx,
                                  (stencil.sy - 1) / 2 + Cstencil.sy,
                                  (stencil.sz - 1) / 2 + Cstencil.sz};

               const int s1[3] = {code[0] < 1 ? (code[0] < 0 ? sC[0] : 0) : nX / 2,
                                  code[1] < 1 ? (code[1] < 0 ? sC[1] : 0) : nY / 2,
                                  code[2] < 1 ? (code[2] < 0 ? sC[2] : 0) : nZ / 2};

               Real *dst1 = coarseBlock + ((s1[2] - sC[2]) * ElemsPerSlice[1] +
                                           (s1[1] - sC[1]) * CLength[0] + s1[0] - sC[0]) *
                                              gptfloats;

               int L[3];
               int code__[3] = {-code[0], -code[1], -code[2]};
               SM.CoarseStencilLength(&code__[0], &L[0]);

               unpack_subregion<Real>(
                   &recv_buffer[other.myrank][unpack.offset + unpack.CoarseVersionOffset], &dst1[0],
                   gptfloats, &stencil.selcomponents[0], stencil.selcomponents.size(),
                   unpack.CoarseVersionsrcxstart, unpack.CoarseVersionsrcystart,
                   unpack.CoarseVersionsrczstart, unpack.CoarseVersionLX, unpack.CoarseVersionLY, 0,
                   0, 0, L[0], L[1], L[2], CLength[0], CLength[1], CLength[2]);
            }
         }
         else if (other.level < info.level)
         {
            const int sC[3] = {
                code[0] < 1 ? (code[0] < 0 ? ((stencil.sx - 1) / 2 + Cstencil.sx) : 0) : nX / 2,
                code[1] < 1 ? (code[1] < 0 ? ((stencil.sy - 1) / 2 + Cstencil.sy) : 0) : nY / 2,
                code[2] < 1 ? (code[2] < 0 ? ((stencil.sz - 1) / 2 + Cstencil.sz) : 0) : nZ / 2};

            const int offset[3] = {(stencil.sx - 1) / 2 + Cstencil.sx,
                                   (stencil.sy - 1) / 2 + Cstencil.sy,
                                   (stencil.sz - 1) / 2 + Cstencil.sz};

            Real *dst = coarseBlock + ((sC[2] - offset[2]) * ElemsPerSlice[1] + sC[0] - offset[0] +
                                       (sC[1] - offset[1]) * CLength[0]) *
                                          gptfloats;

            unpack_subregion<Real>(&recv_buffer[other.myrank][unpack.offset], &dst[0], gptfloats,
                                   &stencil.selcomponents[0], stencil.selcomponents.size(),
                                   unpack.srcxstart, unpack.srcystart, unpack.srczstart, unpack.LX,
                                   unpack.LY, 0, 0, 0, unpack.lx, unpack.ly, unpack.lz, CLength[0],
                                   CLength[1], CLength[2]);
         }
         else
         {
            int B;
            if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 3)) B = 0; // corner
            else if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 2))   // edge
            {
               int t;
               if (code[0] == 0)
               {
                  t = other.index[0] - (2 * info.index[0] + max(code[0], 0) + code[0]);
               }
               else if (code[1] == 0)
               {
                  t = other.index[1] - (2 * info.index[1] + max(code[1], 0) + code[1]);
               }
               else // if (code[2] ==0)
               {
                  t = other.index[2] - (2 * info.index[2] + max(code[2], 0) + code[2]);
               }

               assert(t == 0 || t == 1);

               if (t == 1) B = 3;
               else
                  B = 0;
            }
            else
            {
               int Bmod, Bdiv;
               if (abs(code[0]) == 1)
               {
                  Bmod = other.index[1] - (2 * info.index[1] + max(code[1], 0) + code[1]);
                  Bdiv = other.index[2] - (2 * info.index[2] + max(code[2], 0) + code[2]);
               }
               else if (abs(code[1]) == 1)
               {
                  Bmod = other.index[0] - (2 * info.index[0] + max(code[0], 0) + code[0]);
                  Bdiv = other.index[2] - (2 * info.index[2] + max(code[2], 0) + code[2]);
               }
               else
               {
                  Bmod = other.index[0] - (2 * info.index[0] + max(code[0], 0) + code[0]);
                  Bdiv = other.index[1] - (2 * info.index[1] + max(code[1], 0) + code[1]);
               }

               B = 2 * Bdiv + Bmod;
            }
            const int aux1 = (abs(code[0]) == 1) ? (B % 2) : (B / 2);

            int iz = s[2];
            int iy = s[1];

            const int my_ix =
                abs(code[0]) * (s[0] - stencil.sx) +
                (1 - abs(code[0])) * (s[0] - stencil.sx + (B % 2) * (e[0] - s[0]) / 2);
            const int m_vSize0         = Length[0];
            const int m_nElemsPerSlice = ElemsPerSlice[0];
            const int my_izx =
                (abs(code[2]) * (iz - stencil.sz) +
                 (1 - abs(code[2])) * (iz / 2 - stencil.sz + (B / 2) * (e[2] - s[2]) / 2)) *
                    m_nElemsPerSlice +
                my_ix;

            Real *dst =
                cacheBlock +
                (my_izx + (abs(code[1]) * (iy - stencil.sy) +
                           (1 - abs(code[1])) * (iy / 2 - stencil.sy + aux1 * (e[1] - s[1]) / 2)) *
                              m_vSize0) *
                    gptfloats;

            unpack_subregion<Real>(&recv_buffer[other.myrank][unpack.offset], &dst[0], gptfloats,
                                   &stencil.selcomponents[0], stencil.selcomponents.size(),
                                   unpack.srcxstart, unpack.srcystart, unpack.srczstart, unpack.LX,
                                   unpack.LY, 0, 0, 0, unpack.lx, unpack.ly, unpack.lz, Length[0],
                                   Length[1], Length[2]);
         }
      }
   }
};

template <typename Real> std::set<int> SynchronizerMPI_AMR<Real>::Neighbors;
template <typename Real> std::vector<BlockInfo *> SynchronizerMPI_AMR<Real>::halo_blocks;
template <typename Real> std::vector<BlockInfo *> SynchronizerMPI_AMR<Real>::inner_blocks;
template <typename Real> std::vector<std::vector<std::pair<int, int>>> SynchronizerMPI_AMR<Real>::interface_ranks_and_positions;
template <typename Real> std::vector<GrowingVector<Interface>> SynchronizerMPI_AMR<Real>::send_interfaces;
template <typename Real> std::vector<size_t> SynchronizerMPI_AMR<Real>::lengths;

CUBISM_NAMESPACE_END