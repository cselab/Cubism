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

CUBISM_NAMESPACE_BEGIN

template <typename TGrid>
class GridMPI : public TGrid
{
 public:
   typedef typename TGrid::Real Real;

 private:
   size_t timestamp;

 protected:
   typedef SynchronizerMPI_AMR<Real> SynchronizerMPIType;
   int myrank,world_size;
   int blocksize[3];
   MPI_Comm worldcomm,cartcomm;

 public:
   typedef typename TGrid::BlockType Block;
   typedef typename TGrid::BlockType BlockType;
   std::map<StencilInfo, SynchronizerMPIType *> SynchronizerMPIs;

   GridMPI(const int npeX, const int npeY, const int npeZ, 
           const int nX, 
           const int nY = 1,
           const int nZ = 1, 
           const double _maxextent = 1, 
           const int a_levelStart = 0,
           const int a_levelMax = 1, 
           const MPI_Comm comm = MPI_COMM_WORLD,
           const bool a_xperiodic = true, 
           const bool a_yperiodic = true,
           const bool a_zperiodic = true)
       : TGrid(nX, nY, nZ, _maxextent, a_levelStart, a_levelMax, false, a_xperiodic, a_yperiodic, a_zperiodic), timestamp(0), worldcomm(comm)
   {
      blocksize[0] = Block::sizeX;
      blocksize[1] = Block::sizeY;
      blocksize[2] = Block::sizeZ;

      MPI_Comm_size(worldcomm, &world_size);
      MPI_Comm_rank(worldcomm, &myrank);

      cartcomm     = worldcomm;

      int total_blocks = nX * nY * nZ * pow(pow(2, a_levelStart), 3);

      if (myrank == 0) std::cout << "Total blocks = " << total_blocks << "\n";

      int my_blocks = total_blocks / world_size;
      if (myrank < total_blocks % world_size) my_blocks++;
      int n_start = myrank * (total_blocks / world_size);

      if (total_blocks % world_size > 0)
      {
         if (myrank < total_blocks % world_size) n_start += myrank;
         else
            n_start += total_blocks % world_size;
      }

      std::cout << "rank " << myrank << " gets " << my_blocks << " \n";

      for (int n = n_start; n < n_start + my_blocks; n++) TGrid::_alloc(a_levelStart, n);

      for (int m = 0; m < a_levelMax; m++)
         for (int n = 0; n < nX * nY * nZ * pow(pow(2, m), 3); n++)
         {
            if (m == a_levelStart)
            {
               int r;
               if (total_blocks % world_size > 0)
               {
                  if (n + 1 > (total_blocks / world_size + 1) * (total_blocks % world_size))
                  {
                     int aux = (total_blocks / world_size + 1) * (total_blocks % world_size);

                     r = (n - aux) / (total_blocks / world_size) + total_blocks % world_size;
                  }
                  else
                  {
                     r = n / (total_blocks / world_size + 1);
                  }
               }
               else
               {
                  r = n / my_blocks;
               }
               TGrid::getBlockInfoAll(m,n).myrank = r;
            }
            else
            {
               TGrid::getBlockInfoAll(m,n).myrank = -1;
            }
         }

      FillPos(true);

      MPI_Barrier(worldcomm);
      std::cout << "GridMPI constructor called (ok)\n";
   }
   virtual ~GridMPI() override
   {
      for (auto it = SynchronizerMPIs.begin(); it != SynchronizerMPIs.end(); ++it)
         delete it->second;

      SynchronizerMPIs.clear();

      Clock.display();

      MPI_Barrier(worldcomm);

      _deallocAll();
   }

   virtual void _deallocAll() override // called in class destructor
   {
      TGrid::m_blocks.clear();
      TGrid::m_vInfo.clear();
      for (int m = 0; m < TGrid::levelMax; m++)
      {
         for (int n = 0; n < TGrid::NX * TGrid::NY * TGrid::NZ * pow(pow(2, m), 3); n++)
         {
            if (TGrid::getBlockInfoAll(m,n).TreePos == Exists &&
                TGrid::getBlockInfoAll(m,n).myrank == myrank)
            {
               allocator<Block> alloc;
               alloc.deallocate((Block *)TGrid::getBlockInfoAll(m,n).ptrBlock, 1);
            }
         }
      }
   }

   virtual void _alloc(int m, int n) override
   {
      TGrid::_alloc(m, n);
      TGrid::getBlockInfoAll(m,n).myrank = myrank;
      TGrid::m_vInfo.back().myrank     = myrank;
   }

   std::vector<BlockInfo> &getBlocksInfo() override
   { 
      return TGrid::getBlocksInfo();
   }

   const std::vector<BlockInfo> &getBlocksInfo() const override
   {
      return TGrid::getBlocksInfo();
   }

   std::vector<BlockInfo> &getResidentBlocksInfo()
   { 
      return TGrid::getBlocksInfo();
   }

   const std::vector<BlockInfo> &getResidentBlocksInfo() const
   { 
      return TGrid::getBlocksInfo();
   }

   virtual Block *avail(int m, int n) const override
   {
      if (TGrid::getBlockInfoAll(m,n).myrank == myrank)
         return (Block *)TGrid::getBlockInfoAll(m,n).ptrBlock;
      else
         return nullptr;
   }

   virtual Block *avail1(int ix, int iy, int iz, int m) const override
   {
      int n = TGrid::getZforward(m, ix, iy, iz);
      return avail(m, n);
   }

   std::vector<BlockInfo *> boundary;
   void UpdateBoundary()
   {
      static auto blocksPerDim = TGrid::getMaxBlocks();

      int rank, size;
      MPI_Comm_rank(worldcomm, &rank);
      MPI_Comm_size(worldcomm, &size);

      std::vector<std::vector<int>> send_buffer(size);

      std::vector<BlockInfo *> bbb = boundary;
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

            BlockInfo &infoNei =
                TGrid::getBlockInfoAll(info.level, info.Znei_(code[0], code[1], code[2]));

            if (infoNei.TreePos == Exists && infoNei.myrank != rank)
            {
               if (infoNei.state != Refine) infoNei.state = Leave;
               receivers.insert(infoNei.myrank);
               Neighbors.insert(infoNei.myrank);
            }
            else if (infoNei.TreePos == CheckCoarser)
            {
               int nCoarse               = infoNei.Zparent;
               BlockInfo &infoNeiCoarser = TGrid::getBlockInfoAll(infoNei.level - 1, nCoarse);
               if (infoNeiCoarser.myrank != rank)
               {
                  if (infoNeiCoarser.state != Refine) infoNeiCoarser.state = Leave;
                  receivers.insert(infoNeiCoarser.myrank);
                  Neighbors.insert(infoNeiCoarser.myrank);
               }
            }
            else if (infoNei.TreePos == CheckFiner)
            {
               int Bstep = 1;                                                    // face
               if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 2)) Bstep = 3; // edge
               else if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 3))
                  Bstep = 4; // corner

               for (int B = 0; B <= 3; B += Bstep) // loop over blocks that make up face/edge/corner
                                                   // (respectively 4,2 or 1 blocks)
               {
                  const int temp = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
                  int nFine1 = infoNei.Zchild[max(code[0], 0) + (B % 2) * max(0, 1 - abs(code[0]))]
                                             [max(code[1], 0) + temp * max(0, 1 - abs(code[1]))]
                                             [max(code[2], 0) + (B / 2) * max(0, 1 - abs(code[2]))];
                  int nFine = TGrid::getBlockInfoAll(infoNei.level + 1, nFine1)
                                  .Znei_(-code[0], -code[1], -code[2]);

                  BlockInfo &infoNeiFiner = TGrid::getBlockInfoAll(infoNei.level + 1, nFine);
                  if (infoNeiFiner.myrank != rank)
                  {
                     if (infoNeiFiner.state != Refine) infoNeiFiner.state = Leave;
                     receivers.insert(infoNeiFiner.myrank);
                     Neighbors.insert(infoNeiFiner.myrank);
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
               int temp = (info.state == Compress) ? 1:2;
               send_buffer[*it].push_back(info.level);
               send_buffer[*it].push_back(info.Z);
               send_buffer[*it].push_back(temp);
               it++;
            }
         }
      }

      std::vector<MPI_Request> send_requests;
      std::vector<MPI_Request> recv_requests;

      int dummy = 0;
      //for (int r = 0; r < size; r++)
      for (int r : Neighbors)
         if (r != rank)
         {
            send_requests.resize(send_requests.size() + 1);
            if (send_buffer[r].size() != 0)
               MPI_Isend(&send_buffer[r][0], send_buffer[r].size(), MPI_INT, r, 123, worldcomm,
                         &send_requests[send_requests.size() - 1]);
            else
            {
               MPI_Isend(&dummy, 1, MPI_INT, r, 123, worldcomm,
                         &send_requests[send_requests.size() - 1]);
            }
         }

      std::vector<std::vector<int>> recv_buffer(size);
      //for (int r = 0; r < size; r++)
      for (int r : Neighbors)  
         if (r != rank)
         {
            int recv_size;
            MPI_Status status;
            MPI_Probe(r, 123, worldcomm, &status);
            MPI_Get_count(&status, MPI_INT, &recv_size);
            if (recv_size > 0)
            {
               recv_buffer[r].resize(recv_size);
               recv_requests.resize(recv_requests.size() + 1);
               MPI_Irecv(&recv_buffer[r][0], recv_buffer[r].size(), MPI_INT, r, 123, worldcomm,
                         &recv_requests[recv_requests.size() - 1]);
            }
         }

      MPI_Waitall(send_requests.size(), send_requests.data(), MPI_STATUSES_IGNORE);
      MPI_Waitall(recv_requests.size(), recv_requests.data(), MPI_STATUSES_IGNORE);

      for (int r = 0; r < size; r++)
         if (recv_buffer[r].size() > 1)
            for (int index = 0; index < (int)recv_buffer[r].size(); index += 3)
            {
               int level = recv_buffer[r][index];
               int Z     = recv_buffer[r][index + 1];
               TGrid::getBlockInfoAll(level,Z).state =
                   (recv_buffer[r][index + 2] == 1) ? Compress : Refine;
            }
   };

   void UpdateBlockInfoAll_States(bool GlobalUpdate = false)
   {
      int rank, size;
      MPI_Comm_rank(worldcomm, &rank);
      MPI_Comm_size(worldcomm, &size);

      std::vector<BlockInfo> ChangedInfos;
      for (auto &info : TGrid::m_vInfo)
      {
         int m = info.level;
         int n = info.Z;

         assert(info.TreePos == Exists);

         if (GlobalUpdate)
         {
            ChangedInfos.push_back(info);
            info.changed                         = false;
            TGrid::getBlockInfoAll(m, n).changed = false;
         }
         else if (TGrid::getBlockInfoAll(m, n).changed)
         {
            TGrid::getBlockInfoAll(m, n).changed = false;
            info.changed                         = false;
            ChangedInfos.push_back(info);
         }
      }
      size_t myLength = 2 * ChangedInfos.size();
      int *myData     = new int[myLength];

      for (size_t i = 0; i < myLength; i += 2)
      {
         myData[i]     = ChangedInfos[i / 2].level;
         myData[i + 1] = ChangedInfos[i / 2].Z;
      }

      // 2.Gather lengths of all processes and use them to allocate memory on each process
      int *AllLengths = new int[size];
      MPI_Allgather(&myLength, 1, MPI_INT, AllLengths, 1, MPI_INT, worldcomm);
      assert((size_t)AllLengths[rank] == myLength);

      int sumL = 0;
      for (int r = 0; r < size; r++) sumL += AllLengths[r];

      int *All_data = new int[sumL];
      sumL          = 0;
      std::vector<int *> All_ptr(size);
      std::vector<int> displacement(size);
      for (int r = 0; r < size; r++)
      {
         All_ptr[r]      = &All_data[sumL];
         displacement[r] = sumL;
         sumL += AllLengths[r];
      }

      MPI_Allgatherv(myData, myLength, MPI_INT, All_data, AllLengths, displacement.data(), MPI_INT,
                     worldcomm);

      for (int r = 0; r < size; r++)
      {
         int *ptr = All_ptr[r];
         for (int index__ = 0; index__ < AllLengths[r]; index__ += 2)
         {
            int level       = ptr[index__];
            int Z           = ptr[index__ + 1];
            BlockInfo &info = TGrid::getBlockInfoAll(level,Z);

            info.TreePos = Exists;
            info.state   = Leave;
            info.myrank  = r;

            int p[3] = {info.index[0], info.index[1], info.index[2]};

            if (level < TGrid::levelMax - 1)
               for (int k = 0; k < 2; k++)
                  for (int j = 0; j < 2; j++)
                     for (int i = 0; i < 2; i++)
                     {
                        int nc =
                            TGrid::getZforward(level + 1, 2 * p[0] + i, 2 * p[1] + j, 2 * p[2] + k);
                        TGrid::getBlockInfoAll(level + 1,nc).TreePos = CheckCoarser;
                        TGrid::getBlockInfoAll(level + 1,nc).myrank  = -1;
                     }
            if (level > 0)
            {
               int nf = TGrid::getZforward(level - 1, p[0] / 2, p[1] / 2, p[2] / 2);
               TGrid::getBlockInfoAll(level - 1,nf).TreePos = CheckFiner;
               TGrid::getBlockInfoAll(level - 1,nf).myrank  = -1;
            }
         }
      }
      delete[] All_data;
      delete[] AllLengths;
      delete[] myData;
   }

   template <typename Processing>
   SynchronizerMPIType *sync(Processing &p)
   {
      bool per[3] = {TGrid::xperiodic, TGrid::yperiodic, TGrid::zperiodic};

      // temporarily hardcoded Cstencil
      StencilInfo Cstencil = p.stencil;
      Cstencil.sx          = -1;
      Cstencil.sy          = -1;
      Cstencil.sz          = -1;
      Cstencil.ex          = 2;
      Cstencil.ey          = 2;
      Cstencil.ez          = 2;
      Cstencil.tensorial   = true;

      auto blockperDim          = TGrid::getMaxBlocks();
      const StencilInfo stencil = p.stencil;
      assert(stencil.isvalid());

      SynchronizerMPIType *queryresult = nullptr;

      typename std::map<StencilInfo, SynchronizerMPIType *>::iterator itSynchronizerMPI =
          SynchronizerMPIs.find(stencil);

      if (itSynchronizerMPI == SynchronizerMPIs.end())
      {
         queryresult = new SynchronizerMPIType(
             p.stencil, Cstencil, worldcomm, per, TGrid::getlevelMax(), Block::sizeX, Block::sizeY,
             Block::sizeZ, blockperDim[0], blockperDim[1], blockperDim[2]);

         if (myrank == 0) std::cout << "GRIDMPI IS CALLING SETUP!!!!\n";
         queryresult->_Setup(&(TGrid::getBlocksInfo())[0], (TGrid::getBlocksInfo()).size(),
                             TGrid::getBlockInfoAll(), timestamp, true);
         SynchronizerMPIs[stencil] = queryresult;
      }
      else
      {
         queryresult = itSynchronizerMPI->second;
      }
      queryresult->sync(sizeof(typename Block::element_type) / sizeof(Real),
                        sizeof(Real) > 4 ? MPI_DOUBLE : MPI_FLOAT,timestamp);
      timestamp = (timestamp + 1) % 32768;
      return queryresult;
   }

   template <typename Processing>
   const SynchronizerMPIType &get_SynchronizerMPI(Processing &p) const
   {
      assert((SynchronizerMPIs.find(p.stencil) != SynchronizerMPIs.end()));

      return *SynchronizerMPIs.find(p.stencil)->second;
   }

   int rank() const override { return myrank; }

   virtual void FillPos(bool CopyInfos = true) override { TGrid::FillPos(CopyInfos); }

   size_t getTimeStamp() const { return timestamp; }

   MPI_Comm getCartComm() const { return cartcomm; }

   MPI_Comm getWorldComm() const { return worldcomm; }

   virtual int get_world_size() const override {return world_size;}
};

CUBISM_NAMESPACE_END
