#pragma once

#include <mpi.h>
#include <omp.h>
#include <vector>
#include <set>
#include <algorithm>

#include "BlockInfo.h"
#include "PUPkernelsMPI.h"
#include "StencilInfo.h"
#include "ConsistentOperations.h"

namespace cubism
{

/** \brief Auxiliary class for SynchronizerMPI_AMR; similar to std::vector however, the stored data 
 * does not decrease in size, it can only increase (the use of this class instead of std::vector 
 * in AMR_Synchronizer resulted in faster performance). */
template <typename T>
class GrowingVector
{
   size_t pos;
   size_t s;

 public:
   std::vector<T> v;
   GrowingVector() { pos = 0; }
   GrowingVector(size_t size) { resize(size); }
   GrowingVector(size_t size, T value) { resize(size, value); }

   void resize(size_t new_size, T value)
   {
      v.resize(new_size, value);
      s = new_size;
   }
   void resize(size_t new_size)
   {
      v.resize(new_size);
      s = new_size;
   }

   size_t size() { return s; }

   void clear()
   {
      pos = 0;
      s   = 0;
   }

   void push_back(T value)
   {
      if (pos < v.size()) v[pos] = value;
      else
         v.push_back(value);
      pos++;
      s++;
   }

   T *data() { return v.data(); }

   T &operator[](size_t i) { return v[i]; }

   T &back() { return v[pos - 1]; }

   void EraseAll()
   {
      v.clear();
      pos = 0;
      s   = 0;
   }

   ~GrowingVector() { v.clear(); }
};

/** \brief Auxiliary struct for SynchronizerMPI_AMR; describes how two adjacent blocks touch.*/
struct Interface
{
  BlockInfo *infos[2]; ///< the two blocks of the interface
  int icode[2]; ///< Two integers from 0 to 26. Each integer can be decoded to a 3-digit number ABC. icode[0] = 1-10 (A=1,B=-1,C=0) means Block 1 is at the +x,-y side of Block 0.
  bool CoarseStencil; ///< =true if the blocks need to exchange cells of their parent blocks
  bool ToBeKept; ///< false if this inteface is a subset of another inteface that will be sent anyway
  int dis; ///< auxiliary variable

  ///Class constructor
  Interface(BlockInfo &i0, BlockInfo &i1, const int a_icode0, const int a_icode1)
  {
    infos[0] = &i0;
    infos[1] = &i1;
    icode[0] = a_icode0;
    icode[1] = a_icode1;
    CoarseStencil = false;
    ToBeKept = true;
    dis      = 0;
  }
};

/** Auxiliary struct for SynchronizerMPI_AMR; similar to StencilInfo.
 * It is possible that the halo cells needed by two or more blocks overlap. To avoid sending the
 * same data twice, this struct has the option to keep track of other MyRanges that are contained 
 * in it and do not need to be sent/received.
 */
struct MyRange
{
  std::vector<int> removedIndices; ///< keep track of all 'index' from other MyRange instances that are contained in this one
  int index; ///< index of this instance of MyRange
  int sx; ///< stencil start in x-direction
  int sy; ///< stencil start in y-direction
  int sz; ///< stencil start in z-direction
  int ex; ///< stencil end in x-direction
  int ey; ///< stencil end in y-direction
  int ez; ///< stencil end in z-direction
  bool needed{true}; ///< set to false if this MyRange is contained in another
  bool avg_down{true}; ///< set to true if gridpoints of this MyRange will be averaged down for coarse stencil interpolation

  /// check if another MyRange is contained here
  bool contains(MyRange & r) const
  {
    if (avg_down != r.avg_down) return false;
    int V  = (ez - sz) * (ey - sy) * (ex - sx);
    int Vr = (r.ez - r.sz) * (r.ey - r.sy) * (r.ex - r.sx);
    return (sx <= r.sx && r.ex <= ex) && (sy <= r.sy && r.ey <= ey) && (sz <= r.sz && r.ez <= ez) && (Vr < V);
  }

  /// keep track of indices of other MyRanges that are contained here
  void Remove(const MyRange &other)
  {
    size_t s = removedIndices.size();
    removedIndices.resize(s + other.removedIndices.size());
    for (size_t i = 0; i < other.removedIndices.size(); i++) removedIndices[s + i] = other.removedIndices[i];
  }
};

/** Auxiliary struct for SynchronizerMPI_AMR; Meta-data of buffers sent among processes.
 *  Data is received in one contiguous buffer. This struct helps unpack the buffer and put data 
 *  in the correct locations.
 */
struct UnPackInfo
{
  int offset; ///< Offset in the buffer where the data related to this UnPackInfo starts.
  int lx; ///< Total size of data in x-direction 
  int ly; ///< Total size of data in y-direction 
  int lz; ///< Total size of data in z-direction 
  int srcxstart; ///< Where in x-direction to start receiving data
  int srcystart; ///< Where in y-direction to start receiving data
  int srczstart; ///< Where in z-direction to start receiving data
  int LX; 
  int LY; 
  int CoarseVersionOffset; ///< Offset in the buffer where the coarsened data related to this UnPackInfo starts.
  int CoarseVersionLX;
  int CoarseVersionLY;
  int CoarseVersionsrcxstart; ///< Where in x-direction to start receiving coarsened data
  int CoarseVersionsrcystart; ///< Where in y-direction to start receiving coarsened data
  int CoarseVersionsrczstart; ///< Where in z-direction to start receiving coarsened data
  int level; ///< refinement level of data
  int icode; ///< Integer from 0 to 26, can be decoded to a 3-digit number ABC. icode = 1-10 (A=1,B=-1,C=0) means Block 1 is at the +x,-y side of Block 0.
  int rank; ///< rank from which this data is received
  int index_0; ///< index of Block in x-direction that sent this data
  int index_1; ///< index of Block in y-direction that sent this data
  int index_2; ///< index of Block in z-direction that sent this data
  long long IDreceiver; ///< unique blockID2 of receiver
};

/** Auxiliary struct for SynchronizerMPI_AMR; keeps track of stencil and range sizes that need to be sent/received.
 *  For a block in 3D, there are a total of 26 possible directions that might require halo cells.
 *  There are also four types of halo cells to exchange, based on the refinement level of the two
 *  neighboring blocks: 1)same level 2)coarse-fine 3)fine-coarse 4)same level that also need to 
 *  exchange averaged down data, in order to perform coarse-fine interpolation for other blocks.
 *  This class creates 4 x 26 (actually 4 x 27) MyRange instances, based on a given StencilInfo.
 */
struct StencilManager
{
  const StencilInfo stencil; ///< stencil to send/receive
  const StencilInfo Cstencil; ///< stencil used by BlockLab for coarse-fine interpolation
  int nX; ///< Block size if x-direction
  int nY; ///< Block size if y-direction
  int nZ; ///< Block size if z-direction
  int sLength[3 * 27 * 3]; ///< Length of all possible stencils to send/receive
  std::array<MyRange, 3 * 27> AllStencils; ///< All possible stencils to send/receive
  MyRange Coarse_Range; ///< range for Cstencil

  /// Class constructor
  StencilManager(StencilInfo a_stencil, StencilInfo a_Cstencil, int a_nX, int a_nY, int a_nZ): stencil(a_stencil), Cstencil(a_Cstencil), nX(a_nX), nY(a_nY), nZ(a_nZ)
  {
    const int sC[3]   = {(stencil.sx - 1) / 2 + Cstencil.sx  , (stencil.sy - 1) / 2 + Cstencil.sy  , (stencil.sz - 1) / 2 + Cstencil.sz};
    const int eC[3]   = { stencil.ex      / 2 + Cstencil.ex  ,  stencil.ey      / 2 + Cstencil.ey  ,  stencil.ez      / 2 + Cstencil.ez};

    for (int icode = 0; icode < 27; icode++)
    {
      const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, (icode / 9) % 3 - 1};

      //This also works for DIMENSION=2 and code[2]=0
      //Same level sender and receiver
      MyRange &range0 = AllStencils[icode];
      range0.sx                         = code[0] < 1 ? (code[0] < 0 ? nX + stencil.sx : 0 ) : 0;
      range0.sy                         = code[1] < 1 ? (code[1] < 0 ? nY + stencil.sy : 0 ) : 0;
      range0.sz                         = code[2] < 1 ? (code[2] < 0 ? nZ + stencil.sz : 0 ) : 0;
      range0.ex                         = code[0] < 1 ? nX                                   : stencil.ex - 1;
      range0.ey                         = code[1] < 1 ? nY                                   : stencil.ey - 1;
      range0.ez                         = code[2] < 1 ? nZ                                   : stencil.ez - 1;
      sLength[3 *  icode +           0] = range0.ex - range0.sx;
      sLength[3 *  icode +           1] = range0.ey - range0.sy;
      sLength[3 *  icode +           2] = range0.ez - range0.sz;

      //Fine sender, coarse receiver
      //Fine sender just needs to send "double" the stencil, so that what it sends gets averaged down
      MyRange &range1 = AllStencils[icode + 27];
      range1.sx                         = code[0] < 1 ? (code[0] < 0 ? nX + 2 * stencil.sx : 0     ) : 0                   ;
      range1.sy                         = code[1] < 1 ? (code[1] < 0 ? nY + 2 * stencil.sy : 0     ) : 0                   ;
      range1.sz                         = code[2] < 1 ? (code[2] < 0 ? nZ + 2 * stencil.sz : 0     ) : 0                   ;
      range1.ex                         = code[0] < 1 ? nX                                           : 2 * (stencil.ex - 1);
      range1.ey                         = code[1] < 1 ? nY                                           : 2 * (stencil.ey - 1);
      range1.ez                         = code[2] < 1 ? nZ                                           : 2 * (stencil.ez - 1);
      sLength[3 * (icode +     27) + 0] = (range1.ex - range1.sx)/2;
      sLength[3 * (icode +     27) + 1] = (range1.ey - range1.sy)/2;
      #if DIMENSION == 3
      sLength[3 * (icode +     27) + 2] = (range1.ez - range1.sz)/2;
      #else
      sLength[3 * (icode +     27) + 2] = 1;
      #endif

      //Coarse sender, fine receiver
      //Coarse sender just needs to send "half" the stencil plus extra cells for coarse-fine interpolation
      MyRange &range2 = AllStencils[icode + 2 * 27];
      range2.sx                         = code[0] < 1 ? (code[0] < 0 ? nX / 2 + sC[0]      : 0) : 0;
      range2.sy                         = code[1] < 1 ? (code[1] < 0 ? nY / 2 + sC[1]      : 0) : 0;
      range2.ex                         = code[0] < 1 ? nX / 2 : eC[0] - 1;
      range2.ey                         = code[1] < 1 ? nY / 2 : eC[1] - 1;
      #if DIMENSION == 3
      range2.sz                         = code[2] < 1 ? (code[2] < 0 ? nZ / 2 + sC[2]      : 0) : 0;
      range2.ez                         = code[2] < 1 ? nZ / 2 : eC[2] - 1;
      #else
      range2.sz                         = 0;
      range2.ez                         = 1;
      #endif
      sLength[3 * (icode + 2 * 27) + 0] = range2.ex-range2.sx;
      sLength[3 * (icode + 2 * 27) + 1] = range2.ey-range2.sy;
      sLength[3 * (icode + 2 * 27) + 2] = range2.ez-range2.sz;
    }
  }

  /// Return stencil XxYxZ dimensions for Cstencil, based on integer icode
  void CoarseStencilLength(const int icode, int *L) const
  {
    L[0] = sLength[3 * (icode + 2 * 27) + 0];
    L[1] = sLength[3 * (icode + 2 * 27) + 1];
    L[2] = sLength[3 * (icode + 2 * 27) + 2];
  }

  /// Return stencil XxYxZ dimensions for Cstencil, based on integer icode and refinement level of sender/receiver
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

  /// Determine which stencil to send, based on interface type of two blocks
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
        const int code[3] = {f.icode[1] % 3 - 1, (f.icode[1] / 3) % 3 - 1, (f.icode[1] / 9) % 3 - 1};

        const int s[3] = {
                code[0] < 1 ? (code[0] < 0 ? ((stencil.sx - 1) / 2 + Cstencil.sx) : 0) : nX / 2,
                code[1] < 1 ? (code[1] < 0 ? ((stencil.sy - 1) / 2 + Cstencil.sy) : 0) : nY / 2,
                code[2] < 1 ? (code[2] < 0 ? ((stencil.sz - 1) / 2 + Cstencil.sz) : 0) : nZ / 2};

        const int e[3] = {code[0] < 1 ? (code[0] < 0 ? 0 : nX / 2) : nX / 2 + stencil.ex / 2 + Cstencil.ex - 1,
                          code[1] < 1 ? (code[1] < 0 ? 0 : nY / 2) : nY / 2 + stencil.ey / 2 + Cstencil.ey - 1,
                          code[2] < 1 ? (code[2] < 0 ? 0 : nZ / 2) : nZ / 2 + stencil.ez / 2 + Cstencil.ez - 1};

        const int base[3] = {(f.infos[1]->index[0] + code[0]) % 2, 
                             (f.infos[1]->index[1] + code[1]) % 2,
                             (f.infos[1]->index[2] + code[2]) % 2};

        int Cindex_true[3];
        for (int d = 0; d < 3; d++)
          Cindex_true[d] = f.infos[1]->index[d] + code[d];

        int CoarseEdge[3];

        CoarseEdge[0] = (code[0] == 0) ? 0
                    : (((f.infos[1]->index[0] % 2 == 0) && (Cindex_true[0] > f.infos[1]->index[0])) ||
                       ((f.infos[1]->index[0] % 2 == 1) && (Cindex_true[0] < f.infos[1]->index[0])))
                          ? 1 : 0;
        CoarseEdge[1] = (code[1] == 0) ? 0
                    : (((f.infos[1]->index[1] % 2 == 0) && (Cindex_true[1] > f.infos[1]->index[1])) ||
                       ((f.infos[1]->index[1] % 2 == 1) && (Cindex_true[1] < f.infos[1]->index[1])))
                          ? 1 : 0;
        CoarseEdge[2] = (code[2] == 0) ? 0
                    : (((f.infos[1]->index[2] % 2 == 0) && (Cindex_true[2] > f.infos[1]->index[2])) ||
                       ((f.infos[1]->index[2] % 2 == 1) && (Cindex_true[2] < f.infos[1]->index[2])))
                          ? 1 : 0;

        Coarse_Range.sx = s[0] + std::max(code[0], 0) * nX / 2 + (1 - abs(code[0])) * base[0] * nX / 2 - code[0] * nX + CoarseEdge[0] * code[0] * nX / 2;
        Coarse_Range.sy = s[1] + std::max(code[1], 0) * nY / 2 + (1 - abs(code[1])) * base[1] * nY / 2 - code[1] * nY + CoarseEdge[1] * code[1] * nY / 2;
        #if DIMENSION == 3
        Coarse_Range.sz = s[2] + std::max(code[2], 0) * nZ / 2 + (1 - abs(code[2])) * base[2] * nZ / 2 - code[2] * nZ + CoarseEdge[2] * code[2] * nZ / 2;
        #else
        Coarse_Range.sz = 0;
        #endif

        Coarse_Range.ex = e[0] + std::max(code[0], 0) * nX / 2 + (1 - abs(code[0])) * base[0] * nX / 2 - code[0] * nX + CoarseEdge[0] * code[0] * nX / 2;
        Coarse_Range.ey = e[1] + std::max(code[1], 0) * nY / 2 + (1 - abs(code[1])) * base[1] * nY / 2 - code[1] * nY + CoarseEdge[1] * code[1] * nY / 2;
        #if DIMENSION == 3
        Coarse_Range.ez = e[2] + std::max(code[2], 0) * nZ / 2 + (1 - abs(code[2])) * base[2] * nZ / 2 - code[2] * nZ + CoarseEdge[2] * code[2] * nZ / 2;
        #else
        Coarse_Range.ez = 1;
        #endif

        return Coarse_Range;
      }
    }
  }

  /// Fix MyRange classes that contain other MyRange classes, in order to avoid sending the same data twice
  void __FixDuplicates(const Interface &f, const Interface &f_dup, int lx, int ly, int lz, int lx_dup, int ly_dup, int lz_dup, int &sx, int &sy, int &sz)
  {
    const BlockInfo &receiver     = *f.infos[1];
    const BlockInfo &receiver_dup = *f_dup.infos[1];
    if (receiver.level >= receiver_dup.level)
    {
      int icode_dup         = f_dup.icode[1];
      const int code_dup[3] = {icode_dup % 3 - 1, (icode_dup / 3) % 3 - 1, (icode_dup / 9) % 3 - 1};
      sx                    = (lx == lx_dup || code_dup[0] != -1) ? 0 : lx - lx_dup;
      sy                    = (ly == ly_dup || code_dup[1] != -1) ? 0 : ly - ly_dup;
      sz                    = (lz == lz_dup || code_dup[2] != -1) ? 0 : lz - lz_dup;
    }
    else
    {
      MyRange & range     = DetermineStencil(f);
      MyRange & range_dup = DetermineStencil(f_dup);
      sx = range_dup.sx - range.sx;
      sy = range_dup.sy - range.sy;
      sz = range_dup.sz - range.sz;
    }
  }

  /// Fix MyRange classes that contain other MyRange classes, in order to avoid sending the same data twice
  void __FixDuplicates2(const Interface &f, const Interface &f_dup, int &sx, int &sy, int &sz)
  {
    if (f.infos[0]->level != f.infos[1]->level || f_dup.infos[0]->level != f_dup.infos[1]->level) return;
    MyRange & range     = DetermineStencil(f, true);
    MyRange & range_dup = DetermineStencil(f_dup, true);
    sx                = range_dup.sx - range.sx;
    sy                = range_dup.sy - range.sy;
    sz                = range_dup.sz - range.sz;
   }
};

/** \brief Auxiliary struct for SynchronizerMPI_AMR; manages how UnpackInfos are sent
 * 
 */
struct UnpacksManagerStruct
{
  size_t blocks; ///< number of boundary blocks that will receive UnPackInfos
  size_t *sizes; ///< sizes[i] is the number of UnPackInfos for block with blockID=i
  int size; ///< number of MPI processes
  std::vector<GrowingVector<UnPackInfo>> manyUnpacks; ///< UnPackInfos this rank will send; manyUnpacks_recv[i] are the UnPackInfos sent to rank i
  std::vector<GrowingVector<UnPackInfo>> manyUnpacks_recv; ///< UnPackInfos this rank will receive; manyUnpacks_recv[i] are the UnPackInfos received from rank i
  GrowingVector<GrowingVector<UnPackInfo *>> unpacks; ///< vector of vectors of UnPackInfos; unpacks[i] contains all UnPackInfos needed for a block with blockID=i
  std::vector<MPI_Request> pack_requests; ///< MPI requests for non-blocking communication of UnPackInfos
  std::vector<std::array<long long int, 2>> MapOfInfos; ///< Pairs of halo block IDs (0,1,...,number of halo blocks-1) and unique blockIDs
  MPI_Datatype MPI_PACK; ///< Custom MPI datatype to send an UnPackInfo
  MPI_Comm comm; ///< MPI communicator

  ///Constructor.
  UnpacksManagerStruct(MPI_Comm a_comm)
  {
    comm = a_comm;
    sizes = nullptr;
    MPI_Comm_size(comm, &size);
    manyUnpacks_recv.resize(size);
    manyUnpacks.resize(size);
    int array_of_blocklengths[22];
    MPI_Datatype array_of_types[22];
    for (int i=0;i<21;i++)
    {
      array_of_blocklengths[i] = 1;
      array_of_types[i] = MPI_INT;
    }
    array_of_blocklengths[21] = 1;
    array_of_types[21] = MPI_LONG_LONG_INT;
    MPI_Aint array_of_displacements[23];
    UnPackInfo p;    
    MPI_Aint base                  ; MPI_Get_address(&p                       ,&base);
    MPI_Aint offset                ; MPI_Get_address(&p.offset                ,&offset                ); array_of_displacements[ 0] = offset                 - base;
    MPI_Aint lx                    ; MPI_Get_address(&p.lx                    ,&lx                    ); array_of_displacements[ 1] = lx                     - base;
    MPI_Aint ly                    ; MPI_Get_address(&p.ly                    ,&ly                    ); array_of_displacements[ 2] = ly                     - base;
    MPI_Aint lz                    ; MPI_Get_address(&p.lz                    ,&lz                    ); array_of_displacements[ 3] = lz                     - base;
    MPI_Aint srcxstart             ; MPI_Get_address(&p.srcxstart             ,&srcxstart             ); array_of_displacements[ 4] = srcxstart              - base;
    MPI_Aint srcystart             ; MPI_Get_address(&p.srcystart             ,&srcystart             ); array_of_displacements[ 5] = srcystart              - base;
    MPI_Aint srczstart             ; MPI_Get_address(&p.srczstart             ,&srczstart             ); array_of_displacements[ 6] = srczstart              - base;
    MPI_Aint LX                    ; MPI_Get_address(&p.LX                    ,&LX                    ); array_of_displacements[ 7] = LX                     - base;
    MPI_Aint LY                    ; MPI_Get_address(&p.LY                    ,&LY                    ); array_of_displacements[ 8] = LY                     - base;
    MPI_Aint CoarseVersionOffset   ; MPI_Get_address(&p.CoarseVersionOffset   ,&CoarseVersionOffset   ); array_of_displacements[ 9] = CoarseVersionOffset    - base;
    MPI_Aint CoarseVersionLX       ; MPI_Get_address(&p.CoarseVersionLX       ,&CoarseVersionLX       ); array_of_displacements[10] = CoarseVersionLX        - base;
    MPI_Aint CoarseVersionLY       ; MPI_Get_address(&p.CoarseVersionLY       ,&CoarseVersionLY       ); array_of_displacements[11] = CoarseVersionLY        - base;
    MPI_Aint CoarseVersionsrcxstart; MPI_Get_address(&p.CoarseVersionsrcxstart,&CoarseVersionsrcxstart); array_of_displacements[12] = CoarseVersionsrcxstart - base;
    MPI_Aint CoarseVersionsrcystart; MPI_Get_address(&p.CoarseVersionsrcystart,&CoarseVersionsrcystart); array_of_displacements[13] = CoarseVersionsrcystart - base;
    MPI_Aint CoarseVersionsrczstart; MPI_Get_address(&p.CoarseVersionsrczstart,&CoarseVersionsrczstart); array_of_displacements[14] = CoarseVersionsrczstart - base;
    MPI_Aint level                 ; MPI_Get_address(&p.level                 ,&level                 ); array_of_displacements[15] = level                  - base;
    MPI_Aint icode                 ; MPI_Get_address(&p.icode                 ,&icode                 ); array_of_displacements[16] = icode                  - base;
    MPI_Aint rank                  ; MPI_Get_address(&p.rank                  ,&rank                  ); array_of_displacements[17] = rank                   - base;
    MPI_Aint index_0               ; MPI_Get_address(&p.index_0               ,&index_0               ); array_of_displacements[18] = index_0                - base;
    MPI_Aint index_1               ; MPI_Get_address(&p.index_1               ,&index_1               ); array_of_displacements[19] = index_1                - base;
    MPI_Aint index_2               ; MPI_Get_address(&p.index_2               ,&index_2               ); array_of_displacements[20] = index_2                - base;
    MPI_Aint IDreceiver            ; MPI_Get_address(&p.IDreceiver            ,&IDreceiver            ); array_of_displacements[21] = IDreceiver             - base;
    MPI_Type_create_struct(22, array_of_blocklengths, array_of_displacements, array_of_types,&MPI_PACK);
    MPI_Type_commit(&MPI_PACK);
  }

  ///Reset the arrays of this struct.
  void clear()
  {
    if (sizes != nullptr)
    {
      delete[] sizes;
      sizes = nullptr;
      for (size_t i = 0; i < blocks; i++) unpacks[i].clear();
      unpacks.clear();
    }
    for (int i = 0; i < size; i++)
    {
      manyUnpacks[i].clear();
      manyUnpacks_recv[i].clear();
    }
    MapOfInfos.clear();
  }

  ///Destructor.
  ~UnpacksManagerStruct() { clear(); MPI_Type_free(&MPI_PACK);}

  ///Allocate a number of UnPackInfos for each halo block
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

  ///Add a received UnPackInfo to the list of UnPackInfos for block with blockID=block_id
  void add(UnPackInfo &info, const size_t block_id)
  {
    assert(block_id < blocks);
    assert(sizes[block_id] < unpacks[block_id].size());
    unpacks[block_id][sizes[block_id]] = &info;
    sizes[block_id]++;
  }

  /// Send UnPackInfos to neighboring ranks
  void SendPacks(std::set<int> Neighbor, const int timestamp)
  {
    pack_requests.clear();
    for (auto &r : Neighbor)
    {
      pack_requests.resize(pack_requests.size() + 1);
      MPI_Isend(manyUnpacks[r].data(), manyUnpacks[r].size(), MPI_PACK, r, timestamp, comm, &pack_requests.back());         
    }
    for (auto &r : Neighbor)
    {
      int number_amount;
      MPI_Status status;
      MPI_Probe(r, timestamp, comm, &status);
      MPI_Get_count(&status, MPI_PACK, &number_amount);
      manyUnpacks_recv[r].resize(number_amount);
      pack_requests.resize(pack_requests.size() + 1);
      MPI_Irecv(manyUnpacks_recv[r].data(), manyUnpacks_recv[r].size(), MPI_PACK, r, timestamp, comm, &pack_requests.back());           
    }
  }

  /// Take received UnPackInfos and reorganise them based on the block they correspond to
  void MapIDs()
  {
    if (pack_requests.size() == 0) return;
    pack_requests.clear();        
    std::sort(MapOfInfos.begin(), MapOfInfos.end());
    int myrank;
    MPI_Comm_rank(comm,&myrank);
    for (int r = 0; r < size; r++) if (r!=myrank)
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

/**
 *  @brief Class responsible for halo cell exchange between different MPI processes.
 *  This class works together with BlockLabMPI to fill the halo cells needed for each GridBlock. To
 *  overlap communication and computation, it distinguishes between 'inner' blocks and 'halo' 
 *  blocks. Inner blocks do not need halo cells from other MPI processes, so they can be immediately
 *  filled; halo blocks are at the boundary of a rank and require cells owned by other ranks. This
 *  class will initiate communication for halo blocks and will provide an array with pointers to
 *  inner blocks. While waiting for communication to complete, the user can operate on inner blocks,
 *  which allows for communication-computation overlap.
 * 
 *  An instance of this class is constructed by providing the StencilInfo (aka the stencil) for a
 *  particular computation, in the class constructor. Then, for a fixed mesh configuration, one call
 *  to '_Setup()' is required. This identifies the boundaries of each rank, its neighbors and the
 *  types of interfaces (faces/edges/corners) shared by two blocks that belong to two different 
 *  ranks. '_Setup()' will then have to be called again only when the mesh changes (this call is
 *  done by the MeshAdaptation class).
 * 
 *  To use this class and send/receive halo cells, the 'sync()' function needs to be called. This 
 *  initiates communication and MPI 'sends' and 'receives'. Once called, the inner and halo blocks
 *  (with their halo cells) can be accessed through 'avail_inner' and 'avail_halo'. Note that 
 *  calling 'avail_halo' will result in waiting time, for the communication of halo cells to 
 *  complete. Therefore, 'avail_inner' should be called first, while communication is performed in
 *  the background. Once the inner blocks are processed, 'avail_halo' should be used to process the
 *  outer/halo blocks.
 * 
 *  @tparam Real: type of data to be sent/received (double/float etc.)
 *  @tparam TGrid: type of grid to operate on (should be GridMPI)
 */ 
template <typename Real, typename TGrid>
class SynchronizerMPI_AMR
{
  MPI_Comm comm;        ///< MPI communicator, same as the communicator from 'grid'
  int rank;             ///< MPI process ID, same as the ID from 'grid'
  int size;             ///< total number of processes, same as number from 'grid'
  StencilInfo stencil;  ///< stencil associated with kernel (advection,diffusion etc.)
  StencilInfo Cstencil; ///< stencil required to do coarse-fine interpolation
  TGrid * grid;         ///< grid which owns blocks that need ghost cells 
  int nX;               ///< size of each block in x-direction
  int nY;               ///< size of each block in y-direction
  int nZ;               ///< size of each block in z-direction
  MPI_Datatype MPIREAL; ///< MPI datatype matching template parameter 'Real'

  std::vector<BlockInfo *> inner_blocks; ///< will contain inner blocks with loaded ghost cells
  std::vector<BlockInfo *>  halo_blocks; ///< will contain outer blocks with loaded ghost cells

  std::vector<GrowingVector<Real>> send_buffer; ///< send_buffer[i] contains data to send to rank i
  std::vector<GrowingVector<Real>> recv_buffer; ///< recv_buffer[i] will receive data from rank i

  std::vector<MPI_Request> requests; ///< requests for non-blocking sends/receives

  std::vector<int> send_buffer_size; ///< sizes of send_buffer (communicated before actual data)
  std::vector<int> recv_buffer_size; ///< sizes of recv_buffer (communicated before actual data)

  std::set<int> Neighbors; ///< IDs of neighboring MPI processes 

  UnpacksManagerStruct UnpacksManager;
  StencilManager SM;

  const unsigned int gptfloats; ///< number of Reals (doubles/float) each Element from Grid has
  const int NC;                 ///< number of components from each Element to send/receive

  /// meta-data for the parts of a particular block that will be sent to another rank
  struct PackInfo
  {
    Real *block; ///< Pointer to the first element of the block whose data will be sent
    Real *pack;  ///< Pointer to the buffer where the block's elements will be copied
    int sx; ///< Start of the block's subset that will be sent (in x-direction)
    int sy; ///< Start of the block's subset that will be sent (in y-direction)
    int sz; ///< Start of the block's subset that will be sent (in z-direction)
    int ex; ///< End of the block's subset that will be sent (in x-direction)
    int ey; ///< End of the block's subset that will be sent (in y-direction)
    int ez; ///< End of the block's subset that will be sent (in z-direction)
  };
  std::vector<GrowingVector<PackInfo>> send_packinfos; ///< vector of vectors of PackInfos; send_packinfos[i] contains all the PackInfos to send to rank i

  std::vector<GrowingVector<Interface>> send_interfaces;  ///< vector of vectors of Interfaces; send_interfaces[i] contains all the Interfaces between this rank and rank i

  std::vector<std::vector<int>> ToBeAveragedDown; ///< vector of vectors of Interfaces that need to be averaged down when sent

  bool use_averages;///< if true, fine blocks average down their cells to provide halo cells for coarse blocks (2nd order accurate). If false, they perform a 3rd-order accurate interpolation instead (which is the accuracy needed to compute 2nd derivatives).

  ///Auxiliary struct used to avoid sending the same data twice
  struct DuplicatesManager
  {
    ///Auxiliary struct to detect and remove duplicate Interfaces
    struct cube //could be more efficient
    {
      GrowingVector <MyRange> compass [27]; ///< All possible MyRange stencil that will be exchanged

      void clear() { for (int i=0;i<27;i++) compass[i].clear(); }

      cube(){}
        
      ///Returns the MyRange objects that will be kept
      std::vector<MyRange *> keepEl()
      {
        std::vector<MyRange *> retval;
        for (int i=0; i<27; i++)
          for (size_t j=0; j< compass[i].size() ; j++)
            if (compass[i][j].needed) retval.push_back(&compass[i][j]);

        return retval;
      }

      ///Will return the indices of the removed MyRange objects (in v)
      void __needed(std::vector<int> & v)
      {
        static constexpr std::array <int,3> faces_and_edges [18] = {
          {0,1,1},{2,1,1},{1,0,1},{1,2,1},{1,1,0},{1,1,2},

          {0,0,1},{0,2,1},{2,0,1},{2,2,1},{1,0,0},{1,0,2},
          {1,2,0},{1,2,2},{0,1,0},{0,1,2},{2,1,0},{2,1,2}};
    
        for (auto & f:faces_and_edges) if ( compass[f[0] + f[1]*3 + f[2]*9].size() != 0 )
        {
          bool needme = false;
          auto & me = compass[f[0] + f[1]*3 + f[2]*9];

          for (size_t j1=0; j1<me.size(); j1++)
            if (me[j1].needed)
            {
              needme = true;
              for (size_t j2=0; j2<me.size(); j2++)
                if (me[j2].needed && me[j2].contains(me[j1]) )
                {
                  me[j1].needed = false;
                  me[j2].removedIndices.push_back(me[j1].index);
                  me[j2].Remove(me[j1]);
                  v.push_back(me[j1].index);
                  break;
                }
            }

          if (!needme) continue;
     
          const int imax = (f[0] == 1) ? 2:f[0];
          const int imin = (f[0] == 1) ? 0:f[0]; 
          const int jmax = (f[1] == 1) ? 2:f[1];
          const int jmin = (f[1] == 1) ? 0:f[1]; 
          const int kmax = (f[2] == 1) ? 2:f[2];
          const int kmin = (f[2] == 1) ? 0:f[2];  
    
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
    };
    cube C;

    std::vector<int> offsets; ///< As the send buffer for each rank is being filled, offset[i] is the current offset where sent data is located in the send buffer.
    SynchronizerMPI_AMR * Synch_ptr; ///< pointer to the SynchronizerMPI_AMR for which to remove duplicate data
    std::vector< GrowingVector <int>> positions;///< positions[i][j ]contains the offset of Interface j in the send buffer to rank i

    DuplicatesManager(SynchronizerMPI_AMR & Synch)
    {
      positions.resize(Synch.size);
      offsets.resize(Synch.size,0);
      Synch_ptr = & Synch;
    }

    ///Adds an element to 'positions[r]'
    void Add(const int r,const int index)
    {
      positions[r].push_back(index);
    }

    /**Remove duplicate data that will be sent to one rank.
     * @param r: the rank where the data will be sent
     * @param f: all the Interfaces between rank r and this rank
     * @param total_size: eventual size of the send buffer to rank r, after duplicate Interfaces are removed.
     */
    void RemoveDuplicates(const int r, std::vector<Interface> & f, int & total_size)
    {
      bool skip_needed = false;
      const int nc = Synch_ptr->getstencil().selcomponents.size();

      C.clear();
      for (size_t i=0; i<positions[r].size();i++)
      {              
        C.compass[f[positions[r][i]].icode[0]].push_back(Synch_ptr->SM.DetermineStencil(f[positions[r][i]]));
        C.compass[f[positions[r][i]].icode[0]].back().index = positions[r][i];
        C.compass[f[positions[r][i]].icode[0]].back().avg_down = (f[positions[r][i]].infos[0]->level > f[positions[r][i]].infos[1]->level);
        if (skip_needed == false) skip_needed = f[positions[r][i]].CoarseStencil;
      }

      if (skip_needed == false)
      {
        std::vector<int> remEl;
        C.__needed(remEl);
        for (size_t k=0; k< remEl.size();k++)
          f[remEl[k]].ToBeKept = false;
      }

      for (auto & i:C.keepEl())
      {
        const int k = i->index;            
        int L [3] ={0,0,0};
        int Lc[3] ={0,0,0};
        Synch_ptr->SM.DetermineStencilLength(f[k].infos[0]->level,f[k].infos[1]->level,f[k].icode[1],L);
        const int V = L[0]*L[1]*L[2];
        int Vc = 0;
        total_size+= V;
        if (f[k].CoarseStencil)
        {
          Synch_ptr->SM.CoarseStencilLength(f[k].icode[1],Lc);
          Vc = Lc[0]*Lc[1]*Lc[2];
          total_size += Vc;
        }                    

        UnPackInfo info = {offsets[r],L[0],L[1],L[2],0,0,0,L[0],L[1],-1, 0,0,0,0,0,f[k].infos[0]->level,
            f[k].icode[1], Synch_ptr->rank,
            f[k].infos[0]->index[0],
            f[k].infos[0]->index[1],
            f[k].infos[0]->index[2], f[k].infos[1]->blockID_2};
          
        f[k].dis = offsets[r];
        offsets[r] += V*nc;
        if (f[k].CoarseStencil)
        {
          offsets[r] += Vc*nc;
          info.CoarseVersionOffset = V*nc;
          info.CoarseVersionLX = Lc[0];
          info.CoarseVersionLY = Lc[1];
        }                   
              
        Synch_ptr->UnpacksManager.manyUnpacks[r].push_back(info);
          
        for (size_t kk=0; kk< (*i).removedIndices.size();kk++)
        {
          const int remEl1 = i->removedIndices[kk];
          Synch_ptr->SM.DetermineStencilLength(f[remEl1].infos[0]->level,f[remEl1].infos[1]->level,f[remEl1].icode[1],&L[0]);
                  
          int srcx, srcy, srcz;
          Synch_ptr->SM.__FixDuplicates(f[k],f[remEl1], info.lx,info.ly,info.lz,L[0],L[1],L[2],srcx,srcy,srcz);
          int Csrcx=0;
          int Csrcy=0;
          int Csrcz=0;
          if (f[k].CoarseStencil) Synch_ptr->SM.__FixDuplicates2(f[k],f[remEl1],Csrcx,Csrcy,Csrcz);
          Synch_ptr->UnpacksManager.manyUnpacks[r].push_back({info.offset,L[0],L[1],L[2],srcx, srcy, srcz,info.LX,info.LY,
          info.CoarseVersionOffset, info.CoarseVersionLX, info.CoarseVersionLY,
              Csrcx, Csrcy, Csrcz,
              f[remEl1].infos[0]->level, f[remEl1].icode[1], Synch_ptr->rank,
              f[remEl1].infos[0]->index[0],
              f[remEl1].infos[0]->index[1],
              f[remEl1].infos[0]->index[2], f[remEl1].infos[1]->blockID_2});

          f[remEl1].dis = info.offset;
        } 
      }
    }
  };

  /// Check if blocks on the same refinement level need to exchange averaged down cells that will be used for coarse-fine interpolation.
  bool UseCoarseStencil(const Interface &f)
  {
    BlockInfo &a = *f.infos[0];
    BlockInfo &b = *f.infos[1];
    if (a.level == 0|| (!use_averages)) return false;
    int imin[3];
    int imax[3];
    for (int d = 0; d < 3; d++)
    {
      imin[d] = (a.index[d] < b.index[d]) ? 0 : -1;
      imax[d] = (a.index[d] > b.index[d]) ? 0 : +1;
    }
    const int aux = 1 << a.level;
    if (grid->xperiodic)
    {
      if (a.index[0] == 0 && b.index[0] == grid->getMaxBlocks()[0] * aux - 1) imin[0] = -1;
      if (b.index[0] == 0 && a.index[0] == grid->getMaxBlocks()[0] * aux - 1) imax[0] = +1;
    }
    if (grid->yperiodic)
    {
      if (a.index[1] == 0 && b.index[1] == grid->getMaxBlocks()[1] * aux - 1) imin[1] = -1;
      if (b.index[1] == 0 && a.index[1] == grid->getMaxBlocks()[1] * aux - 1) imax[1] = +1;
    }
    if (grid->zperiodic)
    {
      if (a.index[2] == 0 && b.index[2] == grid->getMaxBlocks()[2] * aux - 1) imin[2] = -1;
      if (b.index[2] == 0 && a.index[2] == grid->getMaxBlocks()[2] * aux - 1) imax[2] = +1;
    }

    bool retval = false;
    for (int i2 = imin[2]; i2 <= imax[2]; i2++)
    for (int i1 = imin[1]; i1 <= imax[1]; i1++)
    for (int i0 = imin[0]; i0 <= imax[0]; i0++)
    {
      const long long n = a.Znei_(i0, i1, i2);
      if ((grid->Tree(a.level, n)).CheckCoarser())
      {
        retval = true;
        break;
      }
    }
    return retval;
  }

  /// Auxiliary function to average down data
  void AverageDownAndFill(Real * __restrict__ dst, const BlockInfo *const info, const int code[3])
  {
    const int s[3] = {code[0] < 1 ? (code[0] < 0 ? stencil.sx : 0) : nX,
                      code[1] < 1 ? (code[1] < 0 ? stencil.sy : 0) : nY,
                      code[2] < 1 ? (code[2] < 0 ? stencil.sz : 0) : nZ};
    const int e[3] = {code[0] < 1 ? (code[0] < 0 ? 0 : nX) : nX + stencil.ex - 1,
                      code[1] < 1 ? (code[1] < 0 ? 0 : nY) : nY + stencil.ey - 1,
                      code[2] < 1 ? (code[2] < 0 ? 0 : nZ) : nZ + stencil.ez - 1};
    #if DIMENSION == 3
      int pos = 0;
      const Real *src = (const Real *)(*info).ptrBlock;
      const int xStep = (code[0] == 0) ? 2 : 1;
      const int yStep = (code[1] == 0) ? 2 : 1;
      const int zStep = (code[2] == 0) ? 2 : 1;
      if (gptfloats == 1)
      {
        for (int iz = s[2]; iz < e[2]; iz += zStep)
        {
          const int ZZ = (abs(code[2]) == 1) ? 2 * (iz - code[2] * nZ) + std::min(0, code[2]) * nZ : iz;
          for (int iy = s[1]; iy < e[1]; iy += yStep)
          {
            const int YY = (abs(code[1]) == 1) ? 2 * (iy - code[1] * nY) + std::min(0, code[1]) * nY : iy;
            for (int ix = s[0]; ix < e[0]; ix += xStep)
            {
              const int XX = (abs(code[0]) == 1) ? 2 * (ix - code[0] * nX) + std::min(0, code[0]) * nX : ix;
              #ifdef PRESERVE_SYMMETRY
              dst[pos] = ConsistentAverage( src[XX  +(YY  +(ZZ  )*nY)*nX],
                                            src[XX  +(YY  +(ZZ+1)*nY)*nX],
                                            src[XX  +(YY+1+(ZZ  )*nY)*nX],
                                            src[XX  +(YY+1+(ZZ+1)*nY)*nX],
                                            src[XX+1+(YY  +(ZZ  )*nY)*nX],
                                            src[XX+1+(YY  +(ZZ+1)*nY)*nX],
                                            src[XX+1+(YY+1+(ZZ  )*nY)*nX],
                                            src[XX+1+(YY+1+(ZZ+1)*nY)*nX]);
              #else
              dst[pos] = 0.125 *(src[XX  +(YY  +(ZZ  )*nY)*nX]+
                                 src[XX  +(YY  +(ZZ+1)*nY)*nX]+
                                 src[XX  +(YY+1+(ZZ  )*nY)*nX]+
                                 src[XX  +(YY+1+(ZZ+1)*nY)*nX]+
                                 src[XX+1+(YY  +(ZZ  )*nY)*nX]+
                                 src[XX+1+(YY  +(ZZ+1)*nY)*nX]+
                                 src[XX+1+(YY+1+(ZZ  )*nY)*nX]+
                                 src[XX+1+(YY+1+(ZZ+1)*nY)*nX]);
              #endif
              pos ++;
            }
          }
        }
      }
      else
      {
        for (int iz = s[2]; iz < e[2]; iz += zStep)
        {
          const int ZZ = (abs(code[2]) == 1) ? 2 * (iz - code[2] * nZ) + std::min(0, code[2]) * nZ : iz;
          for (int iy = s[1]; iy < e[1]; iy += yStep)
          {
            const int YY = (abs(code[1]) == 1) ? 2 * (iy - code[1] * nY) + std::min(0, code[1]) * nY : iy;
            for (int ix = s[0]; ix < e[0]; ix += xStep)
            {
              const int XX = (abs(code[0]) == 1) ? 2 * (ix - code[0] * nX) + std::min(0, code[0]) * nX : ix;
              for (int c = 0; c < NC; c++)
              {
                int comp = stencil.selcomponents[c];
                  #ifdef PRESERVE_SYMMETRY
                  dst[pos] = ConsistentAverage( (*(src + gptfloats * ((XX    ) + ((YY    ) + (ZZ    )*nY) * nX) + comp)),
                                                (*(src + gptfloats * ((XX    ) + ((YY    ) + (ZZ + 1)*nY) * nX) + comp)),
                                                (*(src + gptfloats * ((XX    ) + ((YY + 1) + (ZZ    )*nY) * nX) + comp)),
                                                (*(src + gptfloats * ((XX    ) + ((YY + 1) + (ZZ + 1)*nY) * nX) + comp)),
                                                (*(src + gptfloats * ((XX + 1) + ((YY    ) + (ZZ    )*nY) * nX) + comp)),
                                                (*(src + gptfloats * ((XX + 1) + ((YY    ) + (ZZ + 1)*nY) * nX) + comp)),
                                                (*(src + gptfloats * ((XX + 1) + ((YY + 1) + (ZZ    )*nY) * nX) + comp)),
                                                (*(src + gptfloats * ((XX + 1) + ((YY + 1) + (ZZ + 1)*nY) * nX) + comp)));
                  #else
                  dst[pos] = 0.125 *
                          ((*(src + gptfloats * ((XX) + ((YY) + (ZZ)*nY) * nX) + comp)) +
                           (*(src + gptfloats * ((XX) + ((YY) + (ZZ + 1) * nY) * nX) + comp)) +
                           (*(src + gptfloats * ((XX) + ((YY + 1) + (ZZ)*nY) * nX) + comp)) +
                           (*(src + gptfloats * ((XX) + ((YY + 1) + (ZZ + 1) * nY) * nX) + comp)) +
                           (*(src + gptfloats * ((XX + 1) + ((YY) + (ZZ)*nY) * nX) + comp)) +
                           (*(src + gptfloats * ((XX + 1) + ((YY) + (ZZ + 1) * nY) * nX) + comp)) +
                           (*(src + gptfloats * ((XX + 1) + ((YY + 1) + (ZZ)*nY) * nX) + comp)) +
                           (*(src + gptfloats * ((XX + 1) + ((YY + 1) + (ZZ + 1) * nY) * nX) + comp)));
                  #endif
                pos++;
              }
            }
          }
        }
      }
    #endif
    #if DIMENSION == 2
      Real *src = (Real *)(*info).ptrBlock;
      const int xStep = (code[0] == 0) ? 2 : 1;
      const int yStep = (code[1] == 0) ? 2 : 1;
      int pos = 0;
      for (int iy = s[1]; iy < e[1]; iy += yStep)
      {
        const int YY = (abs(code[1]) == 1) ? 2 * (iy - code[1] * nY) + std::min(0, code[1]) * nY : iy;
        for (int ix = s[0]; ix < e[0]; ix += xStep)
        {
          const int XX = (abs(code[0]) == 1) ? 2 * (ix - code[0] * nX) + std::min(0, code[0]) * nX : ix;
          for (int c = 0; c < NC; c++)
          {
            int comp = stencil.selcomponents[c];
            dst[pos] = 0.25 *(((*(src + gptfloats*(XX  +(YY  )*nX) + comp)) +
                               (*(src + gptfloats*(XX+1+(YY+1)*nX) + comp)))+
                              ((*(src + gptfloats*(XX  +(YY+1)*nX) + comp)) +
                               (*(src + gptfloats*(XX+1+(YY  )*nX) + comp))));
            pos++;
          }
        }
      }
    #endif  
  }

  /// Auxiliary function to average down data
  void AverageDownAndFill2(Real *dst, const BlockInfo *const info, const int code[3])
  {
    const int eC[3] = {(stencil.ex) / 2 + Cstencil.ex, (stencil.ey) / 2 + Cstencil.ey, (stencil.ez) / 2 + Cstencil.ez};
    const int sC[3] = {(stencil.sx - 1) / 2 + Cstencil.sx, (stencil.sy - 1) / 2 + Cstencil.sy, (stencil.sz - 1) / 2 + Cstencil.sz};

    const int s[3] = {code[0] < 1 ? (code[0] < 0 ? sC[0] : 0) : nX / 2,
                      code[1] < 1 ? (code[1] < 0 ? sC[1] : 0) : nY / 2,
                      code[2] < 1 ? (code[2] < 0 ? sC[2] : 0) : nZ / 2};

    const int e[3] = {code[0] < 1 ? (code[0] < 0 ? 0 : nX / 2) : nX / 2 + eC[0] - 1,
                      code[1] < 1 ? (code[1] < 0 ? 0 : nY / 2) : nY / 2 + eC[1] - 1,
                      code[2] < 1 ? (code[2] < 0 ? 0 : nZ / 2) : nZ / 2 + eC[2] - 1};

    Real *src = (Real *)(*info).ptrBlock;

    int pos = 0;

    #if DIMENSION == 3
    for (int iz = s[2]; iz < e[2]; iz++)
    {
      const int ZZ = 2 * (iz - s[2]) + s[2] + std::max(code[2], 0) * nZ / 2 - code[2] * nZ + std::min(0, code[2]) * (e[2] - s[2]);
    #endif
      for (int iy = s[1]; iy < e[1]; iy++)
      {
        const int YY = 2 * (iy - s[1]) + s[1] + std::max(code[1], 0) * nY / 2 - code[1] * nY + std::min(0, code[1]) * (e[1] - s[1]);
        for (int ix = s[0]; ix < e[0]; ix++)
        {
          const int XX = 2 * (ix - s[0]) + s[0] + std::max(code[0], 0) * nX / 2 - code[0] * nX + std::min(0, code[0]) * (e[0] - s[0]);

          for (int c = 0; c < NC; c++)
          {
            int comp = stencil.selcomponents[c];
            #if DIMENSION == 3
              #ifdef PRESERVE_SYMMETRY
              dst[pos] = ConsistentAverage( (*(src + gptfloats * ((XX    ) + ((YY    ) + (ZZ    )*nY) * nX) + comp)),
                                            (*(src + gptfloats * ((XX    ) + ((YY    ) + (ZZ + 1)*nY) * nX) + comp)),
                                            (*(src + gptfloats * ((XX    ) + ((YY + 1) + (ZZ    )*nY) * nX) + comp)),
                                            (*(src + gptfloats * ((XX    ) + ((YY + 1) + (ZZ + 1)*nY) * nX) + comp)),
                                            (*(src + gptfloats * ((XX + 1) + ((YY    ) + (ZZ    )*nY) * nX) + comp)),
                                            (*(src + gptfloats * ((XX + 1) + ((YY    ) + (ZZ + 1)*nY) * nX) + comp)),
                                            (*(src + gptfloats * ((XX + 1) + ((YY + 1) + (ZZ    )*nY) * nX) + comp)),
                                            (*(src + gptfloats * ((XX + 1) + ((YY + 1) + (ZZ + 1)*nY) * nX) + comp)));
              #else
              dst[pos] = 0.125 *
                ((*(src + gptfloats * ((XX) + ((YY) + (ZZ)*nY) * nX) + comp)) +
                 (*(src + gptfloats * ((XX) + ((YY) + (ZZ + 1) * nY) * nX) + comp)) +
                 (*(src + gptfloats * ((XX) + ((YY + 1) + (ZZ)*nY) * nX) + comp)) +
                 (*(src + gptfloats * ((XX) + ((YY + 1) + (ZZ + 1) * nY) * nX) + comp)) +
                 (*(src + gptfloats * ((XX + 1) + ((YY) + (ZZ)*nY) * nX) + comp)) +
                 (*(src + gptfloats * ((XX + 1) + ((YY) + (ZZ + 1) * nY) * nX) + comp)) +
                 (*(src + gptfloats * ((XX + 1) + ((YY + 1) + (ZZ)*nY) * nX) + comp)) +
                 (*(src + gptfloats * ((XX + 1) + ((YY + 1) + (ZZ + 1) * nY) * nX) + comp)));
              #endif
            #else
              dst[pos] = 0.25 *
                      (((*(src + gptfloats*(XX  +(YY  )*nX) + comp)) +
                        (*(src + gptfloats*(XX+1+(YY+1)*nX) + comp)))+
                       ((*(src + gptfloats*(XX  +(YY+1)*nX) + comp)) +
                        (*(src + gptfloats*(XX+1+(YY  )*nX) + comp))));
            #endif
            pos++;
          }
        }
      }
    #if DIMENSION == 3
    }
    #endif
  }

  ///Auxiliary function called from '_Setup()' which will identify the boundary blocks and the type of interface they have with other processes.
  void DefineInterfaces()
  {
    Neighbors.clear();
    inner_blocks.clear();
    halo_blocks.clear();
    for (int r = 0; r < size; r++)
    {
      send_interfaces[r].clear();
      send_buffer_size[r] = 0;
    }

    UnpacksManager.clear();

    std::vector<size_t> lengths;
    std::vector<std::vector<std::pair<int, int>>> interface_ranks_and_positions;

    for (BlockInfo & info : grid->getBlocksInfo())
    {
      info.halo_block_id = -1;
      const int aux    = 1 << info.level;
      const bool xskin = info.index[0] == 0 || info.index[0] == grid->getMaxBlocks()[0] * aux - 1;
      const bool yskin = info.index[1] == 0 || info.index[1] == grid->getMaxBlocks()[1] * aux - 1;
      const bool zskin = info.index[2] == 0 || info.index[2] == grid->getMaxBlocks()[2] * aux - 1;
      const int xskip  = info.index[0] == 0 ? -1 : 1;
      const int yskip  = info.index[1] == 0 ? -1 : 1;
      const int zskip  = info.index[2] == 0 ? -1 : 1;

      bool isInner = true;

      std::vector<int> ToBeChecked;
      bool Coarsened = false;

      int l = 0;
      for (int icode = 0; icode < 27; icode++)
      {
         if (icode == 1 * 1 + 3 * 1 + 9 * 1) continue;
         const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, (icode / 9) % 3 - 1};
         #if DIMENSION == 2
            if (code[2] != 0) continue;
         #endif
         if (!grid->xperiodic && code[0] == xskip && xskin) continue;
         if (!grid->yperiodic && code[1] == yskip && yskin) continue;
         if (!grid->zperiodic && code[2] == zskip && zskin) continue;

         // if (!stencil.tensorial && !Cstencil.tensorial && abs(code[0])+abs(code[1])+abs(code[2])>1) continue;
         //if (!stencil.tensorial && use_averages == false && abs(code[0])+abs(code[1])+abs(code[2])>1) continue;

         BlockInfo &infoNei = grid->getBlockInfoAll(info.level, info.Znei_(code[0], code[1], code[2]));
         const TreePosition & infoNeiTree = grid->Tree(info.level,info.Znei_(code[0], code[1], code[2]));
         if (infoNeiTree.Exists() && infoNeiTree.rank() != rank)
         {
            if (isInner) interface_ranks_and_positions.resize(interface_ranks_and_positions.size() + 1);
            isInner    = false;
            int icode2 = (-code[0] + 1) + (-code[1] + 1) * 3 + (-code[2] + 1) * 9;
            send_interfaces[infoNeiTree.rank()].push_back(Interface(info, infoNei, icode, icode2));
            ToBeChecked.push_back(infoNeiTree.rank());
            ToBeChecked.push_back((int)send_interfaces[infoNeiTree.rank()].size() - 1);
            Neighbors.insert(infoNeiTree.rank());
            interface_ranks_and_positions.back().push_back({infoNeiTree.rank(), (int)send_interfaces[infoNeiTree.rank()].size() - 1});
            l++;
         }
         else if (infoNeiTree.CheckCoarser())
         {
            Coarsened = true;
            const int infoNeiCoarserrank = grid->Tree(info.level-1,infoNei.Zparent).rank();
            if (infoNeiCoarserrank != rank)
            {
               BlockInfo &infoNeiCoarser = grid->getBlockInfoAll(infoNei.level - 1, infoNei.Zparent);

               if (isInner) interface_ranks_and_positions.resize(interface_ranks_and_positions.size() + 1);
               isInner = false;
               int icode2 = (-code[0] + 1) + (-code[1] + 1) * 3 + (-code[2] + 1) * 9;


               const BlockInfo &test = grid->getBlockInfoAll(infoNeiCoarser.level, infoNeiCoarser.Znei_(-code[0], -code[1], -code[2]));

               if (info.index[0] / 2 == test.index[0] && 
                   info.index[1] / 2 == test.index[1] &&
                   info.index[2] / 2 == test.index[2])
               {
                  send_interfaces[infoNeiCoarserrank].push_back(Interface(info, infoNeiCoarser, icode, icode2));
                  interface_ranks_and_positions.back().push_back({infoNeiCoarserrank,(int)send_interfaces[infoNeiCoarserrank].size() - 1});
               }
               Neighbors.insert(infoNeiCoarserrank);
               l++;
            }
         }
         else if (infoNeiTree.CheckFiner())
         {
            int Bstep = 1;
            if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 2)) Bstep = 3; // edge
            else if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 3)) Bstep = 4; // corner

            for (int B = 0; B <= 3;B += Bstep) // loop over blocks that make up face/edge/corner (4/2/1 blocks)
            {
              #if DIMENSION == 2
              if (Bstep == 1 && B >=2) continue;
              if (Bstep >  1 && B >=1) continue;
              #endif
              const int temp = (abs(code[0]) == 1) ? (B % 2) : (B / 2);

              const long long nFine  = infoNei.Zchild[std::max(-code[0], 0) + (B % 2) * std::max(0, 1 - abs(code[0]))]
                                                     [std::max(-code[1], 0) + temp    * std::max(0, 1 - abs(code[1]))]
                                                     [std::max(-code[2], 0) + (B / 2) * std::max(0, 1 - abs(code[2]))];


              const int infoNeiFinerrank = grid->Tree(info.level+1,nFine).rank();

              if (infoNeiFinerrank != rank)
              {
                BlockInfo &infoNeiFiner = grid->getBlockInfoAll(info.level + 1, nFine);
                if (isInner) interface_ranks_and_positions.resize(interface_ranks_and_positions.size() + 1);
                isInner    = false;
                int icode2 = (-code[0] + 1) + (-code[1] + 1) * 3 + (-code[2] + 1) * 9;
                send_interfaces[infoNeiFinerrank].push_back(Interface(info, infoNeiFiner, icode, icode2));
                interface_ranks_and_positions.back().push_back({infoNeiFinerrank,(int)send_interfaces[infoNeiFinerrank].size() - 1});
                Neighbors.insert(infoNeiFinerrank);
                l++;
                #if DIMENSION ==3
                if (Bstep == 3) // if I'm filling an edge then I'm also filling a corner
                {
                   int code3[3];
                   code3[0]   = (code[0] == 0) ? (B == 0 ? 1 : -1) : -code[0];
                   code3[1]   = (code[1] == 0) ? (B == 0 ? 1 : -1) : -code[1];
                   code3[2]   = (code[2] == 0) ? (B == 0 ? 1 : -1) : -code[2];
                   int icode3 = (code3[0] + 1) + (code3[1] + 1) * 3 + (code3[2] + 1) * 9;
                   send_interfaces[infoNeiFinerrank].push_back( Interface(info, infoNeiFiner, icode, icode3));
                   interface_ranks_and_positions.back().push_back({infoNeiFinerrank, (int)send_interfaces[infoNeiFinerrank].size() - 1});
                }
                else
                #endif 
                if (Bstep == 1) // if I'm filling a face then I'm also filling two edges and a corner
                {
                  assert(abs(code[0]) + abs(code[1]) + abs(code[2]) == 1);

                  int code3[3];
                  int code4[3];
                  int code5[3];
                  int d0, d1, d2;
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
                  #if DIMENSION == 2
                    if (code3[2] == 0)
                    {
                      send_interfaces[infoNeiFinerrank].push_back(Interface(info, infoNeiFiner, icode, icode3));
                      interface_ranks_and_positions.back().push_back({infoNeiFinerrank,(int)send_interfaces[infoNeiFinerrank].size() - 1});
                    }
                    if (code4[2] == 0)
                    {
                      send_interfaces[infoNeiFinerrank].push_back(Interface(info, infoNeiFiner, icode, icode4));
                      interface_ranks_and_positions.back().push_back({infoNeiFinerrank,(int)send_interfaces[infoNeiFinerrank].size() - 1});
                    }
                    if (code5[2] == 0)
                    {
                      send_interfaces[infoNeiFinerrank].push_back(Interface(info, infoNeiFiner, icode, icode5));
                      interface_ranks_and_positions.back().push_back({infoNeiFinerrank,(int)send_interfaces[infoNeiFinerrank].size() - 1});
                    }
                  #else
                    send_interfaces[infoNeiFinerrank].push_back(Interface(info, infoNeiFiner, icode, icode3));
                    interface_ranks_and_positions.back().push_back({infoNeiFinerrank,(int)send_interfaces[infoNeiFinerrank].size() - 1});
                    send_interfaces[infoNeiFinerrank].push_back(Interface(info, infoNeiFiner, icode, icode4));
                    interface_ranks_and_positions.back().push_back({infoNeiFinerrank,(int)send_interfaces[infoNeiFinerrank].size() - 1});
                    send_interfaces[infoNeiFinerrank].push_back(Interface(info, infoNeiFiner, icode, icode5));
                    interface_ranks_and_positions.back().push_back({infoNeiFinerrank,(int)send_interfaces[infoNeiFinerrank].size() - 1});
                  #endif
                }
              }
            }
         }
      } // icode = 0,...,26

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
      grid->getBlockInfoAll(info.level, info.Z).halo_block_id = info.halo_block_id;
    } // i-loop

    DuplicatesManager DM(*(this));
    for (size_t i = 0; i < interface_ranks_and_positions.size(); i++)
    {
      const BlockInfo &info = *halo_blocks[i];

      UnpacksManager.MapOfInfos.push_back({info.blockID_2, info.halo_block_id});

      for (int r = 0; r < size; r++) DM.positions[r].clear();

      for (auto &rp : interface_ranks_and_positions[i]) DM.Add(rp.first, rp.second);

      for (int r = 0; r < size; r++) DM.RemoveDuplicates(r, send_interfaces[r].v, send_buffer_size[r]);
    }

    UnpacksManager._allocate(halo_blocks.size(), lengths.data());
  }

  public:
    //constructor
    SynchronizerMPI_AMR(StencilInfo a_stencil, StencilInfo a_Cstencil, TGrid * _grid) : stencil(a_stencil), Cstencil(a_Cstencil),
    UnpacksManager(_grid->getWorldComm()), SM(a_stencil, a_Cstencil, TGrid::Block::sizeX, TGrid::Block::sizeY, TGrid::Block::sizeZ),
    gptfloats(sizeof(typename TGrid::Block::ElementType) / sizeof(Real)), NC(a_stencil.selcomponents.size())
    {
      grid = _grid; 
      use_averages = (grid->FiniteDifferences == false || stencil.tensorial
                     || stencil.sx< -2 || stencil.sy < -2 || stencil.sz < -2
                     || stencil.ex>  3 || stencil.ey >  3 || stencil.ez >  3);

      comm = grid->getWorldComm();
      MPI_Comm_rank(comm, &rank);
      MPI_Comm_size(comm, &size);

      nX = TGrid::Block::sizeX;
      nY = TGrid::Block::sizeY;
      nZ = TGrid::Block::sizeZ;

      send_interfaces.resize(size);
      send_packinfos.resize(size);
      send_buffer_size.resize(size);
      recv_buffer_size.resize(size);
      send_buffer.resize(size);
      recv_buffer.resize(size);

      std::sort(stencil.selcomponents.begin(), stencil.selcomponents.end());

      if (sizeof(Real) == sizeof(double))
      {
        MPIREAL = MPI_DOUBLE;
      }
      else if (sizeof(Real) == sizeof(long double))
      {
        MPIREAL = MPI_LONG_DOUBLE;
      }
      else
      {
        MPIREAL = MPI_FLOAT;
        assert(sizeof(Real) == sizeof(float));
      }
    }

    ///Returns vector of pointers to inner blocks.
    std::vector<BlockInfo *> & avail_inner() { return inner_blocks; }

    ///Returns vector of pointers to halo blocks.
    std::vector<BlockInfo *> & avail_halo()
    {
      MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
      return halo_blocks;
    }

    ///Needs to be called whenever the grid changes because of refinement/compression
    void _Setup()
    {
      const int timestamp = grid->getTimeStamp();

      DefineInterfaces();

      //communicate send/recv buffer sizes with neighbors
      std::vector<MPI_Request> size_requests(2 * Neighbors.size());
      std::vector<int> temp_send(size);
      std::vector<int> temp_recv(size);
      for (int r = 0; r < size; r++)
      {
        send_buffer[r].resize(send_buffer_size[r] * NC);
        temp_send[r] = send_buffer[r].size();
        recv_buffer_size[r] = 0;
      }
      int k = 0;
      for (auto r : Neighbors)
      {
        MPI_Irecv(&temp_recv[r], 1, MPI_INT, r, timestamp, comm, &size_requests[k]    );
        MPI_Isend(&temp_send[r], 1, MPI_INT, r, timestamp, comm, &size_requests[k + 1]);
        k += 2;
      }

      //communicate packs meta-data (stencil sizes, ranges etc.)
      UnpacksManager.SendPacks(Neighbors, timestamp);

      ToBeAveragedDown.resize(size);
      for (int r = 0; r < size; r++)
      {
        send_packinfos[r].clear();
        ToBeAveragedDown[r].clear();

        for (int i = 0; i < (int)send_interfaces[r].size(); i++)
        {
          const Interface &f         = send_interfaces[r][i];

          if (!f.ToBeKept) continue;

          if (f.infos[0]->level <= f.infos[1]->level)
          {
            const MyRange & range = SM.DetermineStencil(f);
            send_packinfos[r].push_back({(Real *)f.infos[0]->ptrBlock, &send_buffer[r][f.dis], range.sx, range.sy, range.sz, range.ex, range.ey, range.ez});
            if (f.CoarseStencil)
            {
              const int V = (range.ex - range.sx) * (range.ey - range.sy) * (range.ez - range.sz);
              ToBeAveragedDown[r].push_back(i);
              ToBeAveragedDown[r].push_back(f.dis + V * NC);
            }
          }
          else // receiver is coarser, so sender averages down data first
          {
            ToBeAveragedDown[r].push_back(i);
            ToBeAveragedDown[r].push_back(f.dis);
          }
        }
      }
      MPI_Waitall(size_requests.size(), size_requests.data(), MPI_STATUSES_IGNORE);
      MPI_Waitall(UnpacksManager.pack_requests.size(), UnpacksManager.pack_requests.data(), MPI_STATUSES_IGNORE);
      UnpacksManager.MapIDs();

      for (auto r : Neighbors)
      {
        recv_buffer_size[r] = temp_recv[r] / NC;
        recv_buffer[r].resize(recv_buffer_size[r] * NC);
      }
    }

    ///Needs to be called to initiate communication and halo cells exchange.
    void sync()
    {
      const int timestamp = grid->getTimeStamp();
      requests.clear();

      //Post receive requests first
      for (auto r : Neighbors) if (recv_buffer_size[r] > 0)
      {
        requests.resize(requests.size() + 1);
        MPI_Irecv(&recv_buffer[r][0], recv_buffer_size[r] * NC, MPIREAL, r, timestamp, comm, &requests.back());
      }

      // Pack data
      for (int r = 0; r < size; r++) if (send_buffer_size[r] != 0)
      {
        #pragma omp parallel
        {
          #pragma omp for
          for (size_t j = 0; j < ToBeAveragedDown[r].size(); j += 2)
          {
            const int i        = ToBeAveragedDown[r][j];
            const int d        = ToBeAveragedDown[r][j + 1];
            const Interface &f = send_interfaces[r][i];
            const int code[3]  = {-(f.icode[0] % 3 - 1), -((f.icode[0] / 3) % 3 - 1), -((f.icode[0] / 9) % 3 - 1)};
            if (f.CoarseStencil) AverageDownAndFill2(send_buffer[r].data() + d, f.infos[0], code);
            else                 AverageDownAndFill (send_buffer[r].data() + d, f.infos[0], code);
          }
          #pragma omp for
          for (size_t i = 0; i < send_packinfos[r].size(); i++)
          {
            const PackInfo &info = send_packinfos[r][i];
            pack(info.block, info.pack, gptfloats, &stencil.selcomponents.front(), NC, info.sx, info.sy, info.sz, info.ex, info.ey, info.ez, nX, nY);
          }
        }
      }

      //Do the sends
      for (auto r : Neighbors) if (send_buffer_size[r] > 0)
      {
        requests.resize(requests.size() + 1);
        MPI_Isend(&send_buffer[r][0], send_buffer_size[r] * NC, MPIREAL, r, timestamp, comm, &requests.back());
      }
    }

    ///Get the StencilInfo of this Synchronizer
    const StencilInfo & getstencil() const { return stencil; }

    ///Used by BlockLabMPI, to get the data from the receive buffers owned by the Synchronizer and put them in its working copy of a GridBlock plus its halo cells. 
    void fetch(const BlockInfo &info, const unsigned int Length[3], const unsigned int CLength[3], Real *cacheBlock, Real *coarseBlock)
    {
      //fetch received data for blocks that are neighbors with 'info' but are owned by another rank
      const int id = info.halo_block_id;
      if (id < 0) return;

      //loop over all unpacks that correspond to block with this halo_block_id
      UnPackInfo **unpacks = UnpacksManager.unpacks[id].data();
      for (size_t jj = 0; jj < UnpacksManager.sizes[id]; jj++)
      {
        const UnPackInfo &unpack = *unpacks[jj];

        const int code[3] = {unpack.icode % 3 - 1, (unpack.icode / 3) % 3 - 1, (unpack.icode / 9) % 3 - 1};

        const int otherrank = unpack.rank;

        //Based on the current unpack's icode, regions starting from 's' and ending to 'e' of the
        //current block will be filled with ghost cells. 
        const int s[3] = {code[0] < 1 ? (code[0] < 0 ? stencil.sx : 0) : nX,
                          code[1] < 1 ? (code[1] < 0 ? stencil.sy : 0) : nY,
                          code[2] < 1 ? (code[2] < 0 ? stencil.sz : 0) : nZ};
        const int e[3] = {code[0] < 1 ? (code[0] < 0 ? 0 : nX) : nX + stencil.ex - 1,
                          code[1] < 1 ? (code[1] < 0 ? 0 : nY) : nY + stencil.ey - 1,
                          code[2] < 1 ? (code[2] < 0 ? 0 : nZ) : nZ + stencil.ez - 1};

        if (unpack.level == info.level) //same level neighbors
        {
          Real *dst = cacheBlock + ((s[2] - stencil.sz) * Length[0] * Length[1] + (s[1] - stencil.sy) * Length[0] + s[0] - stencil.sx) * gptfloats;

          unpack_subregion(&recv_buffer[otherrank][unpack.offset], &dst[0], 
                           gptfloats,&stencil.selcomponents[0], stencil.selcomponents.size(),
                           unpack.srcxstart, unpack.srcystart, unpack.srczstart, unpack.LX, unpack.LY, 
                           0               , 0               , 0               , unpack.lx, unpack.ly, unpack.lz, 
                           Length[0],Length[1], Length[2]);

          if (unpack.CoarseVersionOffset >= 0) //same level neighbors exchange averaged down ghosts
          {
            const int offset[3] = {(stencil.sx - 1) / 2 + Cstencil.sx, (stencil.sy - 1) / 2 + Cstencil.sy, (stencil.sz - 1) / 2 + Cstencil.sz};
            const int sC[3] = {code[0] < 1 ? (code[0] < 0 ? offset[0] : 0) : nX / 2,
                               code[1] < 1 ? (code[1] < 0 ? offset[1] : 0) : nY / 2,
                               code[2] < 1 ? (code[2] < 0 ? offset[2] : 0) : nZ / 2};
            Real *dst1 = coarseBlock + ((sC[2] - offset[2]) * CLength[0] * CLength[1] + (sC[1] - offset[1]) * CLength[0] + sC[0] - offset[0]) * gptfloats;

            int L[3];
            SM.CoarseStencilLength((-code[0]+1)+3*(-code[1]+1)+9*(-code[2]+1), L);

            unpack_subregion(
            &recv_buffer[otherrank][unpack.offset + unpack.CoarseVersionOffset], &dst1[0],
            gptfloats, &stencil.selcomponents[0], stencil.selcomponents.size(),
            unpack.CoarseVersionsrcxstart, unpack.CoarseVersionsrcystart,
            unpack.CoarseVersionsrczstart, unpack.CoarseVersionLX, unpack.CoarseVersionLY,
            0,0,0,L[0],L[1],L[2],CLength[0],CLength[1],CLength[2]);
          }
        }
        else if (unpack.level < info.level)
        {
          const int offset[3] = {(stencil.sx - 1) / 2 + Cstencil.sx, (stencil.sy - 1) / 2 + Cstencil.sy, (stencil.sz - 1) / 2 + Cstencil.sz};
          const int sC[3] = {code[0] < 1 ? (code[0] < 0 ? offset[0] : 0) : nX / 2,
                             code[1] < 1 ? (code[1] < 0 ? offset[1] : 0) : nY / 2,
                             code[2] < 1 ? (code[2] < 0 ? offset[2] : 0) : nZ / 2};
          Real *dst = coarseBlock + ((sC[2] - offset[2]) * CLength[0] * CLength[1] + sC[0] - offset[0] + (sC[1] - offset[1]) * CLength[0]) *gptfloats;
          unpack_subregion(
          &recv_buffer[otherrank][unpack.offset], &dst[0], 
          gptfloats,&stencil.selcomponents[0], stencil.selcomponents.size(),
          unpack.srcxstart,unpack.srcystart,unpack.srczstart,unpack.LX,unpack.LY, 
          0,0,0,unpack.lx,unpack.ly,unpack.lz,CLength[0],CLength[1], CLength[2]);
        }
        else
        {
          int B;
          if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 3)) B = 0; // corner
          else if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 2))   // edge
          {
            int t;
            if      (code[0] == 0) t = unpack.index_0 - 2 * info.index[0];
            else if (code[1] == 0) t = unpack.index_1 - 2 * info.index[1];
            else                   t = unpack.index_2 - 2 * info.index[2];
            assert(t == 0 || t == 1);
            B = (t == 1) ? 3:0;
          }
          else
          {
            int Bmod, Bdiv;
            if (abs(code[0]) == 1)
            {
              Bmod = unpack.index_1 - 2 * info.index[1];
              Bdiv = unpack.index_2 - 2 * info.index[2];
            }
            else if (abs(code[1]) == 1)
            {
              Bmod = unpack.index_0 - 2 * info.index[0];
              Bdiv = unpack.index_2 - 2 * info.index[2];
            }
            else
            {
              Bmod = unpack.index_0 - 2 * info.index[0];
              Bdiv = unpack.index_1 - 2 * info.index[1];
            }
            B = 2 * Bdiv + Bmod;
          }
          
          const int aux1 = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
          Real *dst = cacheBlock + (
                 (abs(code[2]) * (s[2] - stencil.sz) + (1 - abs(code[2])) * (- stencil.sz + (B / 2) * (e[2] - s[2]) / 2)) * Length[0]*Length[1] +
                 (abs(code[1]) * (s[1] - stencil.sy) + (1 - abs(code[1])) * (- stencil.sy + aux1    * (e[1] - s[1]) / 2)) * Length[0] +
                  abs(code[0]) * (s[0] - stencil.sx) + (1 - abs(code[0])) * (- stencil.sx + (B % 2) * (e[0] - s[0]) / 2)
                ) * gptfloats;

          unpack_subregion(
          &recv_buffer[otherrank][unpack.offset], &dst[0], gptfloats,
          &stencil.selcomponents[0],stencil.selcomponents.size(),
          unpack.srcxstart,unpack.srcystart,unpack.srczstart,unpack.LX,unpack.LY, 
          0, 0, 0, unpack.lx, unpack.ly, unpack.lz, Length[0],Length[1], Length[2]);
        }
      }
    }
};

}//namespace cubism
