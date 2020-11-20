#pragma once

#include "AMR_SynchronizerMPI.h"
#include "FluxCorrection.h"
#include <omp.h>

namespace cubism // AMR_CUBISM
{

template <typename TFluxCorrection, typename TGrid>
class FluxCorrectionMPI : public TFluxCorrection
{
 public:
   typedef typename TFluxCorrection::ElementType ElementType;
   typedef typename TFluxCorrection::Real Real;
   typedef typename TFluxCorrection::BlockType BlockType;
   typedef BlockCase<BlockType> Case;

 protected:
   struct face
   {
      BlockInfo *infos[2];
      int icode[2];
      int offset;
      // infos[0] : Fine block
      // infos[1] : Coarse block
      face(BlockInfo &i0, BlockInfo &i1, int a_icode0, int a_icode1)
      {
         infos[0] = &i0;
         infos[1] = &i1;
         icode[0] = a_icode0;
         icode[1] = a_icode1;
      }
      bool operator<(const face &other) const
      {
         if (infos[0]->Z == other.infos[0]->Z)
         {
            return (icode[0] < other.icode[0]);
         }
         else
         {
            return (infos[0]->Z < other.infos[0]->Z);
         }
      }
   };

   int rank, size;
   GrowingVector<std::vector<Real>> send_buffer;
   GrowingVector<std::vector<Real>> recv_buffer;
   GrowingVector<std::vector<face>> send_faces;
   GrowingVector<std::vector<face>> recv_faces;

 public:
   virtual void prepare(TGrid &grid) override
   {
      /*------------->*/ Clock.start(28, "FluxCorrectionMPI prepare");
      if (!grid.UpdateFluxCorrection) return;
      grid.UpdateFluxCorrection = false;

      MPI_Comm_size(grid.getWorldComm(), &size);
      MPI_Comm_rank(grid.getWorldComm(), &rank);

      if (rank == 0) std::cout << "FluxCorrectionMPI: prepare...\n";

      send_buffer.resize(size);
      recv_buffer.resize(size);
      send_faces.resize(size);
      recv_faces.resize(size);

      for (int r = 0; r < size; r++)
      {
         send_buffer[r].clear();
         recv_buffer[r].clear();
         send_faces[r].clear();
         recv_faces[r].clear();
      }

      std::vector<int> send_buffer_size(size, 0);
      std::vector<int> recv_buffer_size(size, 0);

      const int NC = 8;

      int blocksize[3];
      blocksize[0] = BlockType::sizeX;
      blocksize[1] = BlockType::sizeY;
      blocksize[2] = BlockType::sizeZ;

      TFluxCorrection::Cases.clear();
      TFluxCorrection::MapOfCases.clear();

      TFluxCorrection::m_refGrid = &grid;
      std::vector<BlockInfo> &BB = (*TFluxCorrection::m_refGrid).getBlocksInfo();

      TFluxCorrection::xperiodic = grid.xperiodic;
      TFluxCorrection::yperiodic = grid.yperiodic;
      TFluxCorrection::zperiodic = grid.zperiodic;
      TFluxCorrection::blocksPerDim = grid.getMaxBlocks();

      std::array<int, 6> icode = {1 * 2 + 3 * 1 + 9 * 1, 1 * 0 + 3 * 1 + 9 * 1,
                                  1 * 1 + 3 * 2 + 9 * 1, 1 * 1 + 3 * 0 + 9 * 1,
                                  1 * 1 + 3 * 1 + 9 * 2, 1 * 1 + 3 * 1 + 9 * 0};

      for (auto &info : BB)
      {
         (*TFluxCorrection::m_refGrid).getBlockInfoAll(info.level, info.Z).auxiliary = nullptr;
         info.auxiliary = nullptr;

         const int aux = 1 << info.level;

         const bool xskin = info.index[0] == 0 || info.index[0] == TFluxCorrection::blocksPerDim[0] * aux - 1;
         const bool yskin = info.index[1] == 0 || info.index[1] == TFluxCorrection::blocksPerDim[1] * aux - 1;
         const bool zskin = info.index[2] == 0 || info.index[2] == TFluxCorrection::blocksPerDim[2] * aux - 1;

         const int xskip = info.index[0] == 0 ? -1 : 1;
         const int yskip = info.index[1] == 0 ? -1 : 1;
         const int zskip = info.index[2] == 0 ? -1 : 1;

         bool storeFace[6] = {false, false, false, false, false, false};
         bool stored       = false;

         for (int f = 0; f < 6; f++)
         {
            const int code[3] = {icode[f] % 3 - 1, (icode[f] / 3) % 3 - 1, (icode[f] / 9) % 3 - 1};

            if (!TFluxCorrection::xperiodic && code[0] == xskip && xskin) continue;
            if (!TFluxCorrection::yperiodic && code[1] == yskip && yskin) continue;
            if (!TFluxCorrection::zperiodic && code[2] == zskip && zskin) continue;

            BlockInfo infoNei =
                (*TFluxCorrection::m_refGrid)
                    .getBlockInfoAll(info.level, info.Znei_(code[0], code[1], code[2]));

            if (infoNei.TreePos != Exists)
            {
               storeFace[abs(code[0]) * max(0, code[0]) + abs(code[1]) * (max(0, code[1]) + 2) +
                         abs(code[2]) * (max(0, code[2]) + 4)] = true;
               stored                                          = true;
            }

            int L[3];
            L[0]  = (code[0] == 0) ? blocksize[0] / 2 : 1;
            L[1]  = (code[1] == 0) ? blocksize[1] / 2 : 1;
            L[2]  = (code[2] == 0) ? blocksize[2] / 2 : 1;
            int V = L[0] * L[1] * L[2];

            if (infoNei.TreePos == CheckCoarser)
            {
               int nCoarse = infoNei.Zparent;
               BlockInfo &infoNeiCoarser = (*TFluxCorrection::m_refGrid).getBlockInfoAll(infoNei.level - 1, nCoarse);
               if (infoNeiCoarser.myrank != rank)
               {
                  int code2[3] = {-code[0], -code[1], -code[2]};
                  int icode2   = (code2[0] + 1) + (code2[1] + 1) * 3 + (code2[2] + 1) * 9;
                  send_faces[infoNeiCoarser.myrank].push_back(face(info, infoNeiCoarser, icode[f], icode2));
                  send_buffer_size[infoNeiCoarser.myrank] += V;
               }
            }
            else if (infoNei.TreePos == CheckFiner)
            {
               int Bstep = 1;                      // face
               for (int B = 0; B <= 3; B += Bstep) // loop over blocks that make up face
               {
                  const int temp = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
                  int nFine1 = infoNei.Zchild[max(code[0], 0) + (B % 2) * max(0, 1 - abs(code[0]))]
                                             [max(code[1], 0) + temp * max(0, 1 - abs(code[1]))]
                                             [max(code[2], 0) + (B / 2) * max(0, 1 - abs(code[2]))];

                  int nFine = (*TFluxCorrection::m_refGrid).getBlockInfoAll(infoNei.level + 1, nFine1).Znei_(-code[0], -code[1], -code[2]);
                  BlockInfo &infoNeiFiner = (*TFluxCorrection::m_refGrid).getBlockInfoAll(infoNei.level + 1, nFine);
                  if (infoNeiFiner.myrank != rank)
                  {
                     int icode2 = (-code[0] + 1) + (-code[1] + 1) * 3 + (-code[2] + 1) * 9;
                     recv_faces[infoNeiFiner.myrank].push_back(face(infoNeiFiner, info, icode2, icode[f]));
                     recv_buffer_size[infoNeiFiner.myrank] += V;
                  }
               }
            }
         } // icode = 0,...,26

         if (stored)
         {
            TFluxCorrection::Cases.push_back(Case(storeFace, BlockType::sizeX, BlockType::sizeY, BlockType::sizeZ));
            TFluxCorrection::Cases.back().SetupMetaData(info.level, info.Z);
         }
      }

      for (size_t i = 0; i < TFluxCorrection::Cases.size(); i++)
      {
         TFluxCorrection::MapOfCases.insert(std::pair<std::array<int, 2>, Case *>(
             {TFluxCorrection::Cases[i].level, TFluxCorrection::Cases[i].Z},
             &TFluxCorrection::Cases[i]));
         (*TFluxCorrection::m_refGrid).getBlockInfoAll(TFluxCorrection::Cases[i].level, TFluxCorrection::Cases[i].Z).auxiliary 
         = &TFluxCorrection::Cases[i];
      }

      // 2.Sort faces
      for (int r = 0; r < size; r++)
      {
         std::sort(send_faces[r].begin(), send_faces[r].end());
         std::sort(recv_faces[r].begin(), recv_faces[r].end());
      }

      // 3.Define map
      for (int r = 0; r < size; r++)
      {
         send_buffer[r].resize(send_buffer_size[r] * NC);
         recv_buffer[r].resize(recv_buffer_size[r] * NC);

         int offset = 0;
         for (int k = 0; k < (int)recv_faces[r].size(); k++)
         {
            face &f = recv_faces[r][k];

            const int code[3] = {f.icode[1] % 3 - 1, (f.icode[1] / 3) % 3 - 1,
                                 (f.icode[1] / 9) % 3 - 1};

            int L[3];
            L[0]  = (code[0] == 0) ? blocksize[0] / 2 : 1;
            L[1]  = (code[1] == 0) ? blocksize[1] / 2 : 1;
            L[2]  = (code[2] == 0) ? blocksize[2] / 2 : 1;
            int V = L[0] * L[1] * L[2];

            f.offset = offset;

            offset += V * NC;
         }
      }
      TFluxCorrection::m_refGrid->FillPos();
      /*------------->*/ Clock.finish(28);
   }

   virtual void FillBlockCases(bool Integrate = true) override
   {
      TFluxCorrection::TimeIntegration = Integrate;

      /*------------->*/ Clock.start(29, "FluxCorrectionMPI FillBlockCases");
      // This assumes that the BlockCases have been filled by the user somehow...
      std::vector<BlockInfo> &B = (*TFluxCorrection::m_refGrid).getBlocksInfo();

      // 1.Pack send data
      for (int r = 0; r < size; r++)
      {

         int displacement = 0;
         for (int k = 0; k < (int)send_faces[r].size(); k++)
         {
            face &f = send_faces[r][k];

            BlockInfo &info = *(f.infos[0]);

            auto search = TFluxCorrection::MapOfCases.find({info.level, info.Z});
            assert(search != TFluxCorrection::MapOfCases.end());

            Case &FineCase = (*search->second);

            int icode         = f.icode[0];
            const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, (icode / 9) % 3 - 1};
            int myFace = abs(code[0]) * max(0, code[0]) + abs(code[1]) * (max(0, code[1]) + 2) +
                         abs(code[2]) * (max(0, code[2]) + 4);
            std::vector<ElementType> &FineFace = FineCase.m_pData[myFace];

            int d  = myFace / 2;
            int d1 = max((d + 1) % 3, (d + 2) % 3);
            int d2 = min((d + 1) % 3, (d + 2) % 3);
            int N1 = FineCase.m_vSize[d1];
            int N2 = FineCase.m_vSize[d2];

            for (int i1 = 0; i1 < N1; i1 += 2)
               for (int i2 = 0; i2 < N2; i2 += 2)
               {
                  ElementType avg = ((FineFace[i2 + i1 * N2] + FineFace[i2 + 1 + i1 * N2]) + (FineFace[i2 + (i1 + 1) * N2] + FineFace[i2 + 1 + (i1 + 1) * N2]));
                  #if 0
                     send_buffer[r][displacement]     = avg.alpha1rho1;
                     send_buffer[r][displacement + 1] = avg.alpha2rho2;
                     send_buffer[r][displacement + 2] = avg.ru;
                     send_buffer[r][displacement + 3] = avg.rv;
                     send_buffer[r][displacement + 4] = avg.rw;
                     send_buffer[r][displacement + 5] = avg.energy;
                     send_buffer[r][displacement + 6] = avg.alpha2;
                     send_buffer[r][displacement + 7] = avg.dummy;
                  #else
                     assert(ElementType::DIM == 8);
                     for (int j = 0 ; j < ElementType::DIM; j++) send_buffer[r][displacement + j] = avg.member(j);
                  #endif
                  displacement += 8;
                  FineFace[i2 + i1 * N2].clear();
                  FineFace[i2 + 1 + i1 * N2].clear();
                  FineFace[i2 + (i1 + 1) * N2].clear();
                  FineFace[i2 + 1 + (i1 + 1) * N2].clear();
               }
         }
      }

      std::vector<MPI_Request> send_requests;
      std::vector<MPI_Request> recv_requests;

      for (int r = 0; r < size; r++)
      {
         if (recv_buffer[r].size() != 0)
         {
            MPI_Request req;
            recv_requests.push_back(req);
            MPI_Irecv(&recv_buffer[r][0], recv_buffer[r].size(), MPI_DOUBLE, r, 123456,
                      (*TFluxCorrection::m_refGrid).getWorldComm(), &recv_requests.back());
         }
         if (send_buffer[r].size() != 0)
         {
            MPI_Request req;
            send_requests.push_back(req);
            MPI_Isend(&send_buffer[r][0], send_buffer[r].size(), MPI_DOUBLE, r, 123456,
                      (*TFluxCorrection::m_refGrid).getWorldComm(), &send_requests.back());
         }
      }

      std::array<int, 6> icode = {1 * 2 + 3 * 1 + 9 * 1, 1 * 0 + 3 * 1 + 9 * 1,
                                  1 * 1 + 3 * 2 + 9 * 1, 1 * 1 + 3 * 0 + 9 * 1,
                                  1 * 1 + 3 * 1 + 9 * 2, 1 * 1 + 3 * 1 + 9 * 0};

      #pragma omp parallel for schedule(runtime)
      // for (auto & info: B)
      for (size_t jj = 0; jj < B.size(); jj++)
      {
         BlockInfo &info = B[jj];
         int aux         = 1 << info.level;

         const bool xskin =
             info.index[0] == 0 || info.index[0] == TFluxCorrection::blocksPerDim[0] * aux - 1;
         const bool yskin =
             info.index[1] == 0 || info.index[1] == TFluxCorrection::blocksPerDim[1] * aux - 1;
         const bool zskin =
             info.index[2] == 0 || info.index[2] == TFluxCorrection::blocksPerDim[2] * aux - 1;

         const int xskip = info.index[0] == 0 ? -1 : 1;
         const int yskip = info.index[1] == 0 ? -1 : 1;
         const int zskip = info.index[2] == 0 ? -1 : 1;

         for (int f = 0; f < 6; f++)
         {
            const int code[3] = {icode[f] % 3 - 1, (icode[f] / 3) % 3 - 1, (icode[f] / 9) % 3 - 1};

            if (!TFluxCorrection::xperiodic && code[0] == xskip && xskin) continue;
            if (!TFluxCorrection::yperiodic && code[1] == yskip && yskin) continue;
            if (!TFluxCorrection::zperiodic && code[2] == zskip && zskin) continue;

            BlockInfo infoNei =
                (*TFluxCorrection::m_refGrid)
                    .getBlockInfoAll(info.level, info.Znei_(code[0], code[1], code[2]));

            if (infoNei.TreePos == CheckFiner)
            {
               FillCase(info, code);
            }
         } // icode = 0,...,26
      }

      if (recv_requests.size() > 0)
         MPI_Waitall(recv_requests.size(), &recv_requests[0], MPI_STATUSES_IGNORE);

      for (int r = 0; r < size; r++)
      {
         for (int index = 0; index < (int)recv_faces[r].size(); index++)
         {
            face &f = recv_faces[r][index];
            FillCase_2(f);
         }
      }

      TFluxCorrection::Correct();

      if (send_requests.size() > 0)
         MPI_Waitall(send_requests.size(), &send_requests[0], MPI_STATUSES_IGNORE);

      /*------------->*/ Clock.finish(29);
   }

   void FillCase_2(face F)
   {
      BlockInfo info    = *F.infos[1];
      int icode         = F.icode[1];
      const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, (icode / 9) % 3 - 1};

      int myFace = abs(code[0]) * max(0, code[0]) + abs(code[1]) * (max(0, code[1]) + 2) +
                   abs(code[2]) * (max(0, code[2]) + 4);

      std::array<int, 2> temp = {info.level, info.Z};
      auto search             = TFluxCorrection::MapOfCases.find(temp);

      assert(search != TFluxCorrection::MapOfCases.end());

      Case &CoarseCase                     = (*search->second);
      std::vector<ElementType> &CoarseFace = CoarseCase.m_pData[myFace];

      for (int B = 0; B <= 3; B++) // loop over fine blocks that make up coarse face
      {
         const int aux = (abs(code[0]) == 1) ? (B % 2) : (B / 2);

         int Z = (*TFluxCorrection::m_refGrid)
                     .getZforward(info.level + 1,
                                  2 * info.index[0] + max(code[0], 0) + code[0] +
                                      (B % 2) * max(0, 1 - abs(code[0])),
                                  2 * info.index[1] + max(code[1], 0) + code[1] +
                                      aux * max(0, 1 - abs(code[1])),
                                  2 * info.index[2] + max(code[2], 0) + code[2] +
                                      (B / 2) * max(0, 1 - abs(code[2])));

         if (Z != F.infos[0]->Z) continue;

         int d  = myFace / 2;
         int d1 = max((d + 1) % 3, (d + 2) % 3);
         int d2 = min((d + 1) % 3, (d + 2) % 3);
         int N1 = CoarseCase.m_vSize[d1];
         int N2 = CoarseCase.m_vSize[d2];

         int base = 0; //(B%2)*(N1/2)+ (B/2)*(N2/2)*N1;
         if (B == 1) base = (N2 / 2) + (0) * N2;
         else if (B == 2)
            base = (0) + (N1 / 2) * N2;
         else if (B == 3)
            base = (N2 / 2) + (N1 / 2) * N2;


         int r   = F.infos[0]->myrank;
         int dis = 0;
         for (int i1 = 0; i1 < N1; i1 += 2)
            for (int i2 = 0; i2 < N2; i2 += 2)
            {
               #if 0
               CoarseFace[base + (i2 / 2) + (i1 / 2) * N2].alpha1rho1 +=recv_buffer[r][F.offset + dis];
               CoarseFace[base + (i2 / 2) + (i1 / 2) * N2].alpha2rho2 +=recv_buffer[r][F.offset + dis + 1];
               CoarseFace[base + (i2 / 2) + (i1 / 2) * N2].ru         +=recv_buffer[r][F.offset + dis + 2];
               CoarseFace[base + (i2 / 2) + (i1 / 2) * N2].rv         +=recv_buffer[r][F.offset + dis + 3];
               CoarseFace[base + (i2 / 2) + (i1 / 2) * N2].rw         +=recv_buffer[r][F.offset + dis + 4];
               CoarseFace[base + (i2 / 2) + (i1 / 2) * N2].energy     +=recv_buffer[r][F.offset + dis + 5];
               CoarseFace[base + (i2 / 2) + (i1 / 2) * N2].alpha2     +=recv_buffer[r][F.offset + dis + 6];
               CoarseFace[base + (i2 / 2) + (i1 / 2) * N2].dummy      +=recv_buffer[r][F.offset + dis + 7];
               #else
               assert(ElementType::DIM == 8);
               for (int j = 0; j < ElementType::DIM; j++)
                  CoarseFace[base + (i2 / 2) + (i1 / 2) * N2].member(j) +=recv_buffer[r][F.offset + dis + j];
               #endif
               dis += 8;
            }
      }
   }

   virtual void FillCase(BlockInfo info, const int *const code) override
   {
      int myFace = abs(code[0]) * max(0, code[0]) + abs(code[1]) * (max(0, code[1]) + 2) +
                   abs(code[2]) * (max(0, code[2]) + 4);
      int otherFace = abs(-code[0]) * max(0, -code[0]) + abs(-code[1]) * (max(0, -code[1]) + 2) +
                      abs(-code[2]) * (max(0, -code[2]) + 4);

      std::array<int, 2> temp = {info.level, info.Z};
      auto search             = TFluxCorrection::MapOfCases.find(temp);

      assert(myFace / 2 == otherFace / 2);
      assert(search != TFluxCorrection::MapOfCases.end());

      Case &CoarseCase = (*search->second);

      assert(CoarseCase.Z == info.Z);
      assert(CoarseCase.level == info.level);

      std::vector<ElementType> &CoarseFace = CoarseCase.m_pData[myFace];

      for (int B = 0; B <= 3; B++) // loop over fine blocks that make up coarse face
      {
         const int aux = (abs(code[0]) == 1) ? (B % 2) : (B / 2);

         int Z = (*TFluxCorrection::m_refGrid).getZforward(info.level + 1,
            2 * info.index[0] + max(code[0], 0) + code[0] + (B % 2) * max(0, 1 - abs(code[0])),
            2 * info.index[1] + max(code[1], 0) + code[1] +     aux * max(0, 1 - abs(code[1])),
            2 * info.index[2] + max(code[2], 0) + code[2] + (B / 2) * max(0, 1 - abs(code[2])));

         if ((*TFluxCorrection::m_refGrid).getBlockInfoAll(info.level + 1, Z).myrank != rank) continue;

         auto search1 = TFluxCorrection::MapOfCases.find({info.level + 1, Z});
         assert(search1 != TFluxCorrection::MapOfCases.end());

         Case &FineCase                     = (*search1->second);
         std::vector<ElementType> &FineFace = FineCase.m_pData[otherFace];

         int d   = myFace / 2;
         int d1  = max((d + 1) % 3, (d + 2) % 3);
         int d2  = min((d + 1) % 3, (d + 2) % 3);
         int N1F = FineCase.m_vSize[d1];
         int N2F = FineCase.m_vSize[d2];
         int N1  = N1F;
         int N2  = N2F;

         assert(N1F == (int)CoarseCase.m_vSize[d1]);
         assert(N2F == (int)CoarseCase.m_vSize[d2]);

         int base = 0; //(B%2)*(N1/2)+ (B/2)*(N2/2)*N1;
         if      (B == 1) base = (N2 / 2) + (0) * N2;
         else if (B == 2) base = (0) + (N1 / 2) * N2;
         else if (B == 3) base = (N2 / 2) + (N1 / 2) * N2;

         assert(FineFace.size() == CoarseFace.size());

         for (int i1 = 0; i1 < N1; i1 += 2)
            for (int i2 = 0; i2 < N2; i2 += 2)
            {
               CoarseFace[base + (i2 / 2) + (i1 / 2) * N2] += FineFace[i2     + i1      * N2] + 
                                                              FineFace[i2 + 1 + i1      * N2] +
                                                              FineFace[i2     +(i1 + 1) * N2] + 
                                                              FineFace[i2 + 1 +(i1 + 1) * N2] ;
               FineFace[i2 +      i1      * N2].clear();
               FineFace[i2 + 1 +  i1      * N2].clear();
               FineFace[i2 +     (i1 + 1) * N2].clear();
               FineFace[i2 + 1 + (i1 + 1) * N2].clear();
            }
      }
   }
};

} // namespace cubism