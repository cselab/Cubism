/*
 *  GridMPI.h
 *
 *  Created by Michalis Chatzimanolakis
 *  Copyright 2020 ETH Zurich. All rights reserved.
 *
 */
#pragma once

#include <map>
#include <mpi.h>
#include <set>
#include <vector>

#include "AMR_SynchronizerMPI.h"
#include "BlockInfo.h"
#include "StencilInfo.h"
#include "FluxCorrectionMPI.h"

#include <unistd.h>
#include <ios>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <string>
#include "sys/types.h"
#include "sys/sysinfo.h"

namespace cubism {

static inline void memory_usage(std::string msg)
{
  static int counter = 0;
  struct sysinfo memInfo;
  sysinfo (&memInfo);
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  long long totalPhysMem = memInfo.totalram;
  long long physMemUsed = memInfo.totalram - memInfo.freeram;
  long long freeMem     = memInfo.freeram;
  totalPhysMem *= memInfo.mem_unit;
  physMemUsed  *= memInfo.mem_unit;
  freeMem      *= memInfo.mem_unit;
  MPI_Allreduce(MPI_IN_PLACE,&totalPhysMem,1,MPI_LONG_LONG,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&physMemUsed ,1,MPI_LONG_LONG,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&freeMem     ,1,MPI_LONG_LONG,MPI_SUM,MPI_COMM_WORLD);
  if (rank == 0)
  {
   std::cout << msg;// << std::endl;
   std::cout << "totalPhysMem = " << totalPhysMem /(size*1024*1024) << " MB" << std::endl;
   std::cout << "--->physMemUsed  = " << physMemUsed  /(size*1024*1024) << " MB" << std::endl;
   std::cout << "--->freeMem  = " << freeMem  /(size*1024*1024) << " MB" << std::endl;
   std::cout << std::endl;

   std::stringstream s;
   s << counter << " " << to_string(physMemUsed  /(size*1024)) << " \n";
   std::string st = s.str();
   std::ofstream out("memory",std::ios_base::app);
   out << st;
   out.close();
   counter ++;
  }
}

template <typename TGrid>
class GridMPI : public TGrid
{
 public:
   typedef typename TGrid::Real Real;
   int myrank, world_size;

 private:
   size_t timestamp;

 protected:
   typedef SynchronizerMPI_AMR<Real, GridMPI<TGrid>> SynchronizerMPIType;
   MPI_Comm worldcomm, cartcomm;

 public:
   typedef typename TGrid::BlockType Block;
   typedef typename TGrid::BlockType BlockType;
   std::map<StencilInfo, SynchronizerMPIType *> SynchronizerMPIs;
   FluxCorrectionMPI<FluxCorrection<GridMPI<TGrid>,Block>,GridMPI<TGrid>> Corrector;

   GridMPI(const int npeX, const int npeY, const int npeZ, const int nX, const int nY = 1, const int nZ = 1,
           const double _maxextent = 1, const int a_levelStart = 0, const int a_levelMax = 1,
           const MPI_Comm comm = MPI_COMM_WORLD, const bool a_xperiodic = true, const bool a_yperiodic = true,
           const bool a_zperiodic = true)
       : TGrid(nX, nY, nZ, _maxextent, a_levelStart, a_levelMax, false, a_xperiodic, a_yperiodic, a_zperiodic),
         timestamp(0), worldcomm(comm)
   {
      MPI_Comm_size(worldcomm, &world_size);
      MPI_Comm_rank(worldcomm, &myrank);
      cartcomm = worldcomm;

      const long long total_blocks = nX * nY * nZ * pow(pow(2, a_levelStart), DIMENSION);
      long long my_blocks          = total_blocks / world_size;
      if ((long long)myrank < total_blocks % world_size) my_blocks++;
      long long n_start = myrank * (total_blocks / world_size);
      if (total_blocks % world_size > 0)
      {
         if ((long long)myrank < total_blocks % world_size) n_start += myrank;
         else
            n_start += total_blocks % world_size;
      }

      std::vector<short int> levels(my_blocks,a_levelStart);
      std::vector<long long> Zs (my_blocks);
      for (long long n = n_start; n < n_start + my_blocks; n++) Zs[n-n_start]=n;
      initialize_blocks(Zs,levels);

      if (myrank == 0)
      {
         std::cout << "Total blocks = " << total_blocks << ", each rank gets " << my_blocks << std::endl;
      }
      MPI_Barrier(worldcomm);
   }

   virtual ~GridMPI() override
   {
      for (auto it = SynchronizerMPIs.begin(); it != SynchronizerMPIs.end(); ++it) delete it->second;

      SynchronizerMPIs.clear();

      MPI_Barrier(worldcomm);
   }

   std::vector<BlockInfo> &getResidentBlocksInfo() { return TGrid::getBlocksInfo(); }

   const std::vector<BlockInfo> &getResidentBlocksInfo() const { return TGrid::getBlocksInfo(); }

   virtual Block *avail(int m, long long n) override
   {
      if (TGrid::Tree(m, n).rank() == myrank) return (Block *)TGrid::getBlockInfoAll(m, n).ptrBlock;
      else
         return nullptr;
   }

   std::vector<BlockInfo *> boundary;
   virtual void UpdateBoundary(bool clean=false) override
   {
      const auto blocksPerDim = TGrid::getMaxBlocks();

      int rank, size;
      MPI_Comm_rank(worldcomm, &rank);
      MPI_Comm_size(worldcomm, &size);

      std::vector<std::vector<long long>> send_buffer(size);

      std::vector<BlockInfo *> & bbb = boundary;
      std::set<int> Neighbors;
      for (size_t jjj = 0; jjj < bbb.size(); jjj++)
      {
         BlockInfo &info = *bbb[jjj];

         std::set<int> receivers;

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

            if (!TGrid::xperiodic && code[0] == xskip && xskin) continue;
            if (!TGrid::yperiodic && code[1] == yskip && yskin) continue;
            if (!TGrid::zperiodic && code[2] == zskip && zskin) continue;
            #if DIMENSION == 2
            if (code[2] != 0) continue;
            #endif

            BlockInfo &infoNei = TGrid::getBlockInfoAll(info.level, info.Znei_(code[0], code[1], code[2]));

            const TreePosition &infoNeiTree = TGrid::Tree(infoNei.level, infoNei.Z);
            if (infoNeiTree.Exists() && infoNeiTree.rank() != rank)
            {
               if (infoNei.state != Refine || clean) infoNei.state = Leave;
               receivers.insert(infoNeiTree.rank());
               Neighbors.insert(infoNeiTree.rank());
            }
            else if (infoNeiTree.CheckCoarser())
            {
               const long long nCoarse                  = infoNei.Zparent;
               BlockInfo &infoNeiCoarser    = TGrid::getBlockInfoAll(infoNei.level - 1, nCoarse);
               const int infoNeiCoarserrank = TGrid::Tree(infoNei.level - 1, nCoarse).rank();
               if (infoNeiCoarserrank != rank)
               {
                  assert(infoNeiCoarserrank >= 0);
                  if (infoNeiCoarser.state != Refine || clean) infoNeiCoarser.state = Leave;
                  receivers.insert(infoNeiCoarserrank);
                  Neighbors.insert(infoNeiCoarserrank);
               }
            }
            else if (infoNeiTree.CheckFiner())
            {
               int Bstep = 1;                                                    // face
               if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 2)) Bstep = 3; // edge
               else if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 3))
                  Bstep = 4; // corner

               #if DIMENSION == 3
               for (int B = 0; B <= 3; B += Bstep) // loop over blocks that make up face/edge/corner
                                                   // (respectively 4,2 or 1 blocks)
               #else
               for (int B = 0; B <= 1; B += Bstep)
               #endif
               {
                  const int temp         = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
                  const long long nFine1 = infoNei.Zchild[max(code[0], 0) + (B % 2) * max(0, 1 - abs(code[0]))]
                                                         [max(code[1], 0) + temp * max(0, 1 - abs(code[1]))]
                                                         [max(code[2], 0) + (B / 2) * max(0, 1 - abs(code[2]))];
                  const long long nFine =
                      TGrid::getBlockInfoAll(infoNei.level + 1, nFine1).Znei_(-code[0], -code[1], -code[2]);

                  BlockInfo &infoNeiFiner    = TGrid::getBlockInfoAll(infoNei.level + 1, nFine);
                  const int infoNeiFinerrank = TGrid::Tree(infoNei.level + 1, nFine).rank();
                  if (infoNeiFinerrank != rank)
                  {
                     if (infoNeiFiner.state != Refine || clean) infoNeiFiner.state = Leave;
                     receivers.insert(infoNeiFinerrank);
                     Neighbors.insert(infoNeiFinerrank);
                  }
               }
            }
         } // icode = 0,...,26

         if (info.changed2 && info.state != Leave)
         {
            if (info.state == Refine) info.changed2 = false;

            std::set<int>::iterator it = receivers.begin();
            while (it != receivers.end())
            {
               int temp = (info.state == Compress) ? 1 : 2;
               send_buffer[*it].push_back(info.level);
               send_buffer[*it].push_back(info.Z);
               send_buffer[*it].push_back(temp);
               it++;
            }
         }
      }

      std::vector<MPI_Request> send_requests;
      std::vector<MPI_Request> recv_requests;

      long long dummy = 0;
      for (int r : Neighbors)
         if (r != rank)
         {
            send_requests.resize(send_requests.size() + 1);
            if (send_buffer[r].size() != 0)
               MPI_Isend(&send_buffer[r][0], send_buffer[r].size(), MPI_LONG_LONG, r, 123, worldcomm,
                         &send_requests[send_requests.size() - 1]);
            else
            {
               MPI_Isend(&dummy, 1, MPI_LONG_LONG, r, 123, worldcomm, &send_requests[send_requests.size() - 1]);
            }
         }

      std::vector<std::vector<long long>> recv_buffer(size);
      for (int r : Neighbors)
         if (r != rank)
         {
            int recv_size;
            MPI_Status status;
            MPI_Probe(r, 123, worldcomm, &status);
            MPI_Get_count(&status, MPI_LONG_LONG, &recv_size);
            if (recv_size > 0)
            {
               recv_buffer[r].resize(recv_size);
               recv_requests.resize(recv_requests.size() + 1);
               MPI_Irecv(&recv_buffer[r][0], recv_buffer[r].size(), MPI_LONG_LONG, r, 123, worldcomm,
                         &recv_requests[recv_requests.size() - 1]);
            }
         }

      MPI_Waitall(send_requests.size(), send_requests.data(), MPI_STATUSES_IGNORE);
      MPI_Waitall(recv_requests.size(), recv_requests.data(), MPI_STATUSES_IGNORE);

      for (int r = 0; r < size; r++)
         if (recv_buffer[r].size() > 1)
            for (int index = 0; index < (int)recv_buffer[r].size(); index += 3)
            {
               int level                              = recv_buffer[r][index];
               long long Z                            = recv_buffer[r][index + 1];
               TGrid::getBlockInfoAll(level, Z).state = (recv_buffer[r][index + 2] == 1) ? Compress : Refine;
            }
   };

   void UpdateBlockInfoAll_States(bool UpdateIDs = false)
   {
      std::vector<int> myNeighbors = FindMyNeighbors();

      const auto blocksPerDim = TGrid::getMaxBlocks();
      std::vector<long long> myData;
      for (auto &info : TGrid::m_vInfo)
      {
         bool myflag = false;

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

            if (!TGrid::xperiodic && code[0] == xskip && xskin) continue;
            if (!TGrid::yperiodic && code[1] == yskip && yskin) continue;
            if (!TGrid::zperiodic && code[2] == zskip && zskin) continue;
            #if DIMENSION == 2
            if (code[2] != 0) continue;
            #endif

            BlockInfo &infoNei = TGrid::getBlockInfoAll(info.level, info.Znei_(code[0], code[1], code[2]));

            const TreePosition &infoNeiTree = TGrid::Tree(infoNei.level, infoNei.Z);
            if (infoNeiTree.Exists() && infoNeiTree.rank() != myrank)
            {
               myflag = true;
               break;
            }
            else if (infoNeiTree.CheckCoarser())
            {
               long long nCoarse            = infoNei.Zparent;
               const int infoNeiCoarserrank = TGrid::Tree(infoNei.level - 1, nCoarse).rank();
               if (infoNeiCoarserrank != myrank)
               {
                  myflag = true;
                  break;
               }
            }
            else if (infoNeiTree.CheckFiner())
            {
               int Bstep = 1;                                                    // face
               if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 2)) Bstep = 3; // edge
               else if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 3))
                  Bstep = 4; // corner

               for (int B = 0; B <= 3; B += Bstep) // loop over blocks that make up face/edge/corner
                                                   // (respectively 4,2 or 1 blocks)
               {
                  const int temp   = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
                  long long nFine1 = infoNei.Zchild[max(code[0], 0) + (B % 2) * max(0, 1 - abs(code[0]))]
                                                   [max(code[1], 0) + temp * max(0, 1 - abs(code[1]))]
                                                   [max(code[2], 0) + (B / 2) * max(0, 1 - abs(code[2]))];
                  long long nFine =
                      TGrid::getBlockInfoAll(infoNei.level + 1, nFine1).Znei_(-code[0], -code[1], -code[2]);
                  const int infoNeiFinerrank = TGrid::Tree(infoNei.level + 1, nFine).rank();
                  if (infoNeiFinerrank != myrank)
                  {
                     myflag = true;
                     break;
                  }
               }
            }
            else if (infoNeiTree.rank() < 0)
            {
               myflag = true;
               break;
            }
         } // icode = 0,...,26

         if (myflag)
         {
            myData.push_back(info.level);
            myData.push_back(info.Z);
            if (UpdateIDs) myData.push_back(info.blockID);
         }
      }

      std::vector<std::vector<long long>> recv_buffer(myNeighbors.size());
      std::vector<std::vector<long long>> send_buffer(myNeighbors.size());
      std::vector<int> recv_size(myNeighbors.size());
      std::vector<MPI_Request> send_size_requests(myNeighbors.size());
      std::vector<MPI_Request> recv_size_requests(myNeighbors.size());
      int mysize = (int)myData.size();
      int kk     = 0;
      for (auto r : myNeighbors)
      {
         MPI_Irecv(&recv_size[kk], 1, MPI_INT, r, timestamp, worldcomm, &recv_size_requests[kk]);
         MPI_Isend(&mysize, 1, MPI_INT, r, timestamp, worldcomm, &send_size_requests[kk]);
         kk++;
      }
      MPI_Waitall(recv_size_requests.size(), recv_size_requests.data(), MPI_STATUSES_IGNORE);
      MPI_Waitall(send_size_requests.size(), send_size_requests.data(), MPI_STATUSES_IGNORE);

      kk = 0;
      for (size_t j = 0; j < myNeighbors.size(); j++)
      {
         recv_buffer[kk].resize(recv_size[kk]);
         send_buffer[kk].resize(myData.size());
         for (size_t i = 0; i < myData.size(); i++) send_buffer[kk][i] = myData[i];
         kk++;
      }
      std::vector<MPI_Request> send_requests(myNeighbors.size());
      std::vector<MPI_Request> recv_requests(myNeighbors.size());
      kk = 0;
      for (auto r : myNeighbors)
      {
         MPI_Irecv(recv_buffer[kk].data(), recv_buffer[kk].size(), MPI_LONG_LONG, r, timestamp, worldcomm,
                   &recv_requests[kk]);
         MPI_Isend(send_buffer[kk].data(), send_buffer[kk].size(), MPI_LONG_LONG, r, timestamp, worldcomm,
                   &send_requests[kk]);
         kk++;
      }
      MPI_Waitall(recv_requests.size(), recv_requests.data(), MPI_STATUSES_IGNORE);
      MPI_Waitall(send_requests.size(), send_requests.data(), MPI_STATUSES_IGNORE);
      kk = -1;
      const int increment = UpdateIDs ? 3 : 2;
      for (auto r : myNeighbors)
      {
         kk++;
         for (size_t index__ = 0; index__ < recv_buffer[kk].size(); index__ += increment)
         {
            const int level   = (int)recv_buffer[kk][index__];
            const long long Z = recv_buffer[kk][index__ + 1];
            TGrid::Tree(level, Z).setrank(r);
            if (UpdateIDs) TGrid::getBlockInfoAll(level, Z).blockID = recv_buffer[kk][index__ + 2];
           #if DIMENSION == 3
            int p[3];
            BlockInfo::inverse(Z, level, p[0], p[1], p[2]);

            if (level < TGrid::levelMax - 1)
               for (int k = 0; k < 2; k++)
                  for (int j = 0; j < 2; j++)
                     for (int i = 0; i < 2; i++)
                     {
                        const long long nc = TGrid::getZforward(level + 1, 2 * p[0] + i, 2 * p[1] + j, 2 * p[2] + k);
                        TGrid::Tree(level + 1, nc).setCheckCoarser();
                     }
            if (level > 0)
            {
               const long long nf = TGrid::getZforward(level - 1, p[0] / 2, p[1] / 2, p[2] / 2);
               TGrid::Tree(level - 1, nf).setCheckFiner();
            }
           #else
            int p[2];
            BlockInfo::inverse(Z, level, p[0], p[1]);
            if (level < TGrid::levelMax - 1)
               for (int j = 0; j < 2; j++)
                  for (int i = 0; i < 2; i++)
                  {
                     const long long nc = TGrid::getZforward(level + 1, 2 * p[0] + i, 2 * p[1] + j);
                     TGrid::Tree(level + 1, nc).setCheckCoarser();
                  }
            if (level > 0)
            {
               const long long nf = TGrid::getZforward(level - 1, p[0] / 2, p[1] / 2);
               TGrid::Tree(level - 1, nf).setCheckFiner();
            }
           #endif
         }
      }
   }

   std::vector<int> FindMyNeighbors()
   {
      std::vector<int> myNeighbors;
      double low[3] = {+1e20, +1e20, +1e20};
      double high[3] = {-1e20, -1e20, -1e20};
      double p_low[3];
      double p_high[3];
      for (auto &info : TGrid::m_vInfo)
      {
        #if DIMENSION == 3
         const double h = 2 * info.h;
         info.pos(p_low, 0, 0, 0);
         info.pos(p_high, Block::sizeX - 1, Block::sizeY - 1, Block::sizeZ - 1);
         p_low[0] -= h;
         p_low[1] -= h;
         p_low[2] -= h;
         p_high[0] += h;
         p_high[1] += h;
         p_high[2] += h;
         low[0]  = std::min(low[0], p_low[0]);
         low[1]  = std::min(low[1], p_low[1]);
         low[2]  = std::min(low[2], p_low[2]);
         high[0] = std::max(high[0], p_high[0]);
         high[1] = std::max(high[1], p_high[1]);
         high[2] = std::max(high[2], p_high[2]);
        #else
         const double h = 2 * info.h;
         info.pos(p_low, 0, 0);
         info.pos(p_high, Block::sizeX - 1, Block::sizeY - 1);
         p_low[0] -= h;
         p_low[1] -= h;
         p_low[2]  = 0;
         p_high[0] += h;
         p_high[1] += h;
         p_high[2]  = 0;
         low[0]  = std::min(low[0], p_low[0]);
         low[1]  = std::min(low[1], p_low[1]);
         low[2]  = 0;
         high[0] = std::max(high[0], p_high[0]);
         high[1] = std::max(high[1], p_high[1]);
         high[2] = 0;
        #endif
      }
      std::vector<double> all_boxes(world_size * 6);
      double my_box[6] = {low[0], low[1], low[2], high[0], high[1], high[2]};
      MPI_Allgather(my_box, 6, MPI_DOUBLE, all_boxes.data(), 6, MPI_DOUBLE, worldcomm);
      for (int i = 0; i < world_size; i++)
      {
         if ( i == myrank) continue;
         if (Intersect(low, high, &all_boxes[i * 6], &all_boxes[i * 6 + 3])) myNeighbors.push_back(i);
      }
      return myNeighbors;
   }

   bool Intersect(double *l1, double *h1, double *l2, double *h2)
   {
      const double h0 =
          (TGrid::maxextent /
           std::max(TGrid::NX * Block::sizeX, std::max(TGrid::NY * Block::sizeY, TGrid::NZ * Block::sizeZ)));
      const double extent[3] = {TGrid::NX * Block::sizeX * h0, TGrid::NY * Block::sizeY * h0,
                                TGrid::NZ * Block::sizeZ * h0};

      const Real intersect[3][2] = {{std::max(l1[0], l2[0]), std::min(h1[0], h2[0])},
                                    {std::max(l1[1], l2[1]), std::min(h1[1], h2[1])},
                                    {std::max(l1[2], l2[2]), std::min(h1[2], h2[2])}};

      const bool normal_intersection_x = intersect[0][1] - intersect[0][0] > 0.0;
      const bool normal_intersection_y = intersect[1][1] - intersect[1][0] > 0.0;
      const bool normal_intersection_z = DIMENSION == 3 ? (intersect[2][1] - intersect[2][0] > 0.0) : true;

      if (normal_intersection_x && normal_intersection_y && normal_intersection_z) return true;

      bool periodic_intersection_x = false;
      bool periodic_intersection_y = false;
      bool periodic_intersection_z = false;
      if (TGrid::xperiodic)
      {
         if (h2[0] > extent[0] && (h2[0] - extent[0] > l1[0]) && (h2[0] - extent[0] < h1[0]))
            periodic_intersection_x = true;
         if (h1[0] > extent[0] && (h1[0] - extent[0] > l2[0]) && (h1[0] - extent[0] < h2[0]))
            periodic_intersection_x = true;
      }
      if (TGrid::yperiodic)
      {
         if (h2[1] > extent[1] && (h2[1] - extent[1] > l1[1]) && (h2[1] - extent[1] < h1[1]))
            periodic_intersection_y = true;
         if (h1[1] > extent[1] && (h1[1] - extent[1] > l2[1]) && (h1[1] - extent[1] < h2[1]))
            periodic_intersection_y = true;
      }
      if (TGrid::zperiodic)
      {
         if (h2[2] > extent[2] && (h2[2] - extent[2] > l1[2]) && (h2[2] - extent[2] < h1[2]))
            periodic_intersection_z = true;
         if (h1[2] > extent[2] && (h1[2] - extent[2] > l2[2]) && (h1[2] - extent[2] < h2[2]))
            periodic_intersection_z = true;
      }

      if (normal_intersection_x && normal_intersection_y && periodic_intersection_z) return true;
      if (normal_intersection_x && periodic_intersection_y && normal_intersection_z) return true;
      if (periodic_intersection_x && normal_intersection_y && normal_intersection_z) return true;

      if (normal_intersection_x && periodic_intersection_y && periodic_intersection_z) return true;
      if (periodic_intersection_x && normal_intersection_y && periodic_intersection_z) return true;
      if (periodic_intersection_x && periodic_intersection_y && normal_intersection_z) return true;

      if (periodic_intersection_x && periodic_intersection_y && periodic_intersection_z) return true;

      return false;
   }

   template <typename Processing>
   SynchronizerMPIType *sync(const Processing &p)
   {
      assert(p.stencil.isvalid());

      // temporarily hardcoded Cstencil
      StencilInfo Cstencil = p.stencil;
      Cstencil.sx          = -1;
      Cstencil.sy          = -1;
      Cstencil.sz          = DIMENSION == 3 ? -1:0;
      Cstencil.ex          = 2;
      Cstencil.ey          = 2;
      Cstencil.ez          = DIMENSION == 3 ? 2:1;
      Cstencil.tensorial   = true;

      SynchronizerMPIType *queryresult = nullptr;

      typename std::map<StencilInfo, SynchronizerMPIType *>::iterator itSynchronizerMPI = SynchronizerMPIs.find(p.stencil);

      if (itSynchronizerMPI == SynchronizerMPIs.end())
      {
         queryresult = new SynchronizerMPIType(p.stencil, Cstencil, this);
         queryresult->_Setup();
         SynchronizerMPIs[p.stencil] = queryresult;
      }
      else
      {
         queryresult = itSynchronizerMPI->second;
      }

      queryresult->sync();

      timestamp = (timestamp + 1) % 32768;
      return queryresult;
   }

   template <typename Processing>
   const SynchronizerMPIType &get_SynchronizerMPI(Processing &p) const
   {
      assert((SynchronizerMPIs.find(p.stencil) != SynchronizerMPIs.end()));

      return *SynchronizerMPIs.find(p.stencil)->second;
   }

   virtual void initialize_blocks(const std::vector<long long> & blocksZ, const std::vector<short int> & blockslevel) override
   {
      TGrid::initialize_blocks(blocksZ,blockslevel);
      UpdateBlockInfoAll_States(false);
      for (auto it = SynchronizerMPIs.begin(); it != SynchronizerMPIs.end(); ++it)
        (*it->second)._Setup();
   }

   virtual int rank() const override { return myrank; }

   size_t getTimeStamp() const { return timestamp; }

   MPI_Comm getCartComm() const { return cartcomm; }

   MPI_Comm getWorldComm() const { return worldcomm; }

   virtual int get_world_size() const override { return world_size; }
};

}//namespace cubism
