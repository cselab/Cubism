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

struct BlockGroup
{
    int i_min[3];
    int i_max[3];
    int level;
    std::vector<int> Z;
    size_t ID;
    double origin[3];
    double h;
    int NXX;
    int NYY;
    int NZZ;
};

template <typename TGrid>
class GridMPI : public TGrid
{
 public:
   typedef typename TGrid::Real Real;
   int myrank,world_size;
 private:
   size_t timestamp;

 protected:
   typedef SynchronizerMPI_AMR<Real, GridMPI<TGrid> > SynchronizerMPIType;
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
      MPI_Comm_size(worldcomm, &world_size);
      MPI_Comm_rank(worldcomm, &myrank);
      cartcomm     = worldcomm;

      int total_blocks = nX * nY * nZ * pow(pow(2, a_levelStart), 3);
      int my_blocks = total_blocks / world_size;
      if (myrank < total_blocks % world_size) my_blocks++;
      int n_start = myrank * (total_blocks / world_size);
      if (total_blocks % world_size > 0)
      {
         if (myrank < total_blocks % world_size) n_start += myrank;
         else n_start += total_blocks % world_size;
      }
      for (int n = n_start; n < n_start + my_blocks; n++) _alloc(a_levelStart, n);

      for (int m = 0 ; m < a_levelMax ; m ++)
      {
        int nmax = nX * nY * nZ * pow(pow(2, m), 3);
        if (m == a_levelStart-1)
            for (int n = 0 ; n < nmax ; n ++) TGrid::Tree(m,n).setCheckFiner(); 
        else if (m == a_levelStart+1)
            for (int n = 0 ; n < nmax ; n ++) TGrid::Tree(m,n).setCheckCoarser();
        else if (m == TGrid::levelStart)
            for (int n = 0 ; n < nmax ; n ++)
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
                TGrid::Tree(m,n).setrank(r);
            }
      }
      if (myrank == 0) std::cout << "Total blocks = " << total_blocks << "\n";
      std::cout << "rank " << myrank << " gets " << my_blocks << " \n";

      FillPos(true);

      MPI_Barrier(worldcomm);
   }

   virtual ~GridMPI() override
   {
      for (auto it = SynchronizerMPIs.begin(); it != SynchronizerMPIs.end(); ++it)
         delete it->second;

      SynchronizerMPIs.clear();

      Clock.display();

      MPI_Barrier(worldcomm);
   }

   virtual void _alloc(int m, int n) override
   {
      allocator<Block> alloc;
      TGrid::getBlockInfoAll(m,n).ptrBlock = alloc.allocate(1);
      TGrid::getBlockInfoAll(m,n).changed  = true;
      TGrid::getBlockInfoAll(m,n).h_gridpoint = TGrid::getBlockInfoAll(m,n).h;

      TGrid::m_blocks.push_back((Block *)TGrid::getBlockInfoAll(m,n).ptrBlock);
      
      TGrid::m_vInfo.push_back(*TGrid::BlockInfoAll[m][n]);

      TGrid::Tree(m,n).setrank(myrank);
   }


   std::vector<BlockInfo> &getResidentBlocksInfo()
   { 
      return TGrid::getBlocksInfo();
   }

   const std::vector<BlockInfo> &getResidentBlocksInfo() const
   { 
      return TGrid::getBlocksInfo();
   }

   virtual Block *avail(int m, int n) override
   {
      if (TGrid::Tree(m,n).rank() == myrank)
         return (Block *)TGrid::getBlockInfoAll(m,n).ptrBlock;
      else
         return nullptr;
   }

   virtual Block *avail1(int ix, int iy, int iz, int m) override
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

            const TreePosition & infoNeiTree = TGrid::Tree(infoNei.level,infoNei.Z);
            if (infoNeiTree.Exists() && infoNeiTree.rank() != rank)
            {
               if (infoNei.state != Refine) infoNei.state = Leave;
               receivers.insert(infoNeiTree.rank());
               Neighbors.insert(infoNeiTree.rank());
            }
            else if (infoNeiTree.CheckCoarser())
            {
               int nCoarse               = infoNei.Zparent;
               BlockInfo &infoNeiCoarser = TGrid::getBlockInfoAll(infoNei.level - 1, nCoarse);
               const int infoNeiCoarserrank = TGrid::Tree(infoNei.level-1,nCoarse).rank();
               if (infoNeiCoarserrank != rank)
               {
                  if (infoNeiCoarser.state != Refine) infoNeiCoarser.state = Leave;
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
                  const int infoNeiFinerrank = TGrid::Tree(infoNei.level+1,nFine).rank();
                  if (infoNeiFinerrank != rank)
                  {
                     if (infoNeiFiner.state != Refine) infoNeiFiner.state = Leave;
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

         //assert(info.TreePos == Exists);

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
            TGrid::Tree(level,Z).setrank(r);
            int p[3];
            BlockInfo::inverse(Z,level,p[0],p[1],p[2]);

            if (level < TGrid::levelMax - 1)
               for (int k = 0; k < 2; k++)
                  for (int j = 0; j < 2; j++)
                     for (int i = 0; i < 2; i++)
                     {
                        int nc =
                            TGrid::getZforward(level + 1, 2 * p[0] + i, 2 * p[1] + j, 2 * p[2] + k);
                        TGrid::Tree(level + 1,nc).setCheckCoarser();
                     }
            if (level > 0)
            {
               int nf = TGrid::getZforward(level - 1, p[0] / 2, p[1] / 2, p[2] / 2);
               TGrid::Tree(level - 1,nf).setCheckFiner();
            }
         }
      }
      delete[] All_data;
      delete[] AllLengths;
      delete[] myData;
   }

   template <typename Processing>
   SynchronizerMPIType *sync(const Processing &p)
   {
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
             p.stencil, Cstencil, TGrid::getlevelMax(), Block::sizeX, Block::sizeY,
             Block::sizeZ, blockperDim[0], blockperDim[1], blockperDim[2], this);

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

   int getResidentBlocksPerDimension(int idim) const
   {
    MPI_Abort(worldcomm,1);
    return 1;
   }

   bool UpdateGroups{true};
   std::vector<BlockGroup> MyGroups;
   std::vector<unsigned int> Groups_per_rank;
   std::vector<unsigned int> allN;
   void UpdateMyGroups()
   {
      if (!UpdateGroups) return;
      if (myrank == 0) std::cout << "Updating groups..." << std::endl;

      typedef typename TGrid::BlockType B;
      const unsigned int nX = B::sizeX;
      const unsigned int nY = B::sizeY;
      const unsigned int nZ = B::sizeZ;
      const size_t Ngrids = TGrid::getBlocksInfo().size();
      const auto & MyInfos = TGrid::getBlocksInfo();
      UpdateGroups = false;

      long long int id_min = MyInfos[0].blockID;
      long long int id_max = MyInfos[0].blockID;
      for (size_t i = 1 ; i < MyInfos.size(); i++)
      {
        id_min = std::min(MyInfos[i].blockID,id_min);
        id_max = std::max(MyInfos[i].blockID,id_max);
      }
      std::vector <bool> added(id_max-id_min+1,false);


      MyGroups.clear();
      for (unsigned int m = 0; m < Ngrids; m++)
      {
        const BlockInfo & I = MyInfos[m];

        if (added[I.blockID - id_min]) continue;
        BlockGroup newGroup;
    
        newGroup.level = I.level;
        newGroup.h = I.h;
        newGroup.Z.push_back(I.Z);
           
        std::vector<int > base (3);
        base[0] = I.index[0];
        base[1] = I.index[1];
        base[2] = I.index[2];
        std::vector<int > i_off(6,0);
        std::vector<bool> ready_(6,false);

        int d = 0;
        auto blk  = TGrid::getMaxBlocks();
        do
        {
          if (ready_[d] == false)
          {
            bool valid = true;    
            i_off[d] ++;
            const int i0 = (d<3) ? (base[d] - i_off[d]) : (base[d-3] + i_off[d]);
            const int d0 = (d<3) ? (d  )%3 : (d-3    )%3;
            const int d1 = (d<3) ? (d+1)%3 : (d-3 + 1)%3;
            const int d2 = (d<3) ? (d+2)%3 : (d-3 + 2)%3;
    
            for (int i2 = base[d2] - i_off[d2]; i2 <= base[d2] + i_off[d2+3]; i2++)
            for (int i1 = base[d1] - i_off[d1]; i1 <= base[d1] + i_off[d1+3]; i1++)
            {
              if (valid == false) break;

                if (i0 < 0 || i1 < 0 || i2 < 0 || i0 >= blk[d0]*(1<<I.level) ||i1 >= blk[d1]*(1<<I.level) || i2 >= blk[d2]*(1<<I.level) )
                {
                    valid = false;
                    break;
                }
                int n;
                if      (d==0||d==3)     n = TGrid::getZforward(I.level,i0,i1,i2);
                else if (d==1||d==4)     n = TGrid::getZforward(I.level,i2,i0,i1);
                else /*if (d==2||d==5)*/ n = TGrid::getZforward(I.level,i1,i2,i0);  
                
                int blockrank = -1;               
                if (TGrid::BlockInfoAll[I.level][n] != nullptr)
                  blockrank = TGrid::Tree(I.level,n).rank();
                  //blockrank = TGrid::BlockInfoAll[I.level][n]->myrank;
                //if (blockrank != myrank || TGrid::BlockInfoAll[I.level][n]->TreePos != Exists)
                if (blockrank != myrank || blockrank< 0)
                {
                  valid = false;
                  break;
                }
                
                if (TGrid::getBlockInfoAll(I.level,n).blockID < id_min || TGrid::getBlockInfoAll(I.level,n).blockID > id_max)
                {
                    valid = false;
                }
                else 
                if (added[TGrid::getBlockInfoAll(I.level,n).blockID - id_min] == true)
                {
                    valid = false;
                }                                   
            }

             if (valid == false)
             {
                 i_off[d] --;
                 ready_[d] = true;
             }
                    else
                    {
                        for (int i2 = base[d2] - i_off[d2]; i2 <= base[d2] + i_off[d2+3]; i2++)
                        for (int i1 = base[d1] - i_off[d1]; i1 <= base[d1] + i_off[d1+3]; i1++)
                        {
                            int n;
                            if      (d==0||d==3) n = TGrid::getZforward(I.level,i0,i1,i2);
                            else if (d==1||d==4) n = TGrid::getZforward(I.level,i2,i0,i1);
                            else /*if (d==2||d==5)*/ n = TGrid::getZforward(I.level,i1,i2,i0);
                            newGroup.Z.push_back(n);
                            added[TGrid::getBlockInfoAll(I.level,n).blockID - id_min] = true;                                   
                        }
                    }
                }
                d = (d+1)%6;         
            }while( ready_[0] == false || ready_[1] == false || ready_[2] == false || ready_[3] == false || ready_[4] == false || ready_[5] == false);
    
            const int ix_min = base[0] - i_off[0];
            const int iy_min = base[1] - i_off[1];
            const int iz_min = base[2] - i_off[2];      
            const int ix_max = base[0] + i_off[3];
            const int iy_max = base[1] + i_off[4];
            const int iz_max = base[2] + i_off[5];
    
            int n_base = TGrid::getZforward(I.level,ix_min,iy_min,iz_min);
    
            newGroup.i_min[0] = ix_min;
            newGroup.i_min[1] = iy_min;
            newGroup.i_min[2] = iz_min;
    
            newGroup.i_max[0] = ix_max;
            newGroup.i_max[1] = iy_max;
            newGroup.i_max[2] = iz_max;
    
            newGroup.origin[0] =  TGrid::BlockInfoAll[I.level][n_base]->origin[0];
            newGroup.origin[1] =  TGrid::BlockInfoAll[I.level][n_base]->origin[1];
            newGroup.origin[2] =  TGrid::BlockInfoAll[I.level][n_base]->origin[2];
    
            newGroup.NXX = (newGroup.i_max[0] - newGroup.i_min[0] + 1)*nX + 1;
            newGroup.NYY = (newGroup.i_max[1] - newGroup.i_min[1] + 1)*nY + 1;
            newGroup.NZZ = (newGroup.i_max[2] - newGroup.i_min[2] + 1)*nZ + 1;
    
            MyGroups.push_back(newGroup);
        }


      size_t mysize = MyGroups.size();
      Groups_per_rank.resize(world_size,0);
      allN.clear();
      std::vector<unsigned int> myN;
      std::vector<int> sendcounts;
      std::vector<int> recvcounts;
      std::vector<int> sdispls;
      std::vector<int> rdispls;
      MPI_Allgather(&mysize, 1, MPI_UNSIGNED, &Groups_per_rank[0], 1, MPI_UNSIGNED, worldcomm);
      for (size_t groupID = 0 ; groupID < MyGroups.size() ; groupID ++)
      {
          const BlockGroup & group = MyGroups[groupID];
          myN.push_back(group.NXX);
          myN.push_back(group.NYY);
          myN.push_back(group.NZZ);
      }

      int dis = 0;
      for (int r = 0; r < world_size; r++)
      {
          allN.resize(allN.size() + 3*Groups_per_rank[r]);
          sendcounts.push_back(myN.size());
          recvcounts.push_back(3*Groups_per_rank[r]);
          sdispls.push_back(0);
          rdispls.push_back(dis);
          dis += 3*Groups_per_rank[r];
      }  
      MPI_Alltoallv(myN.data(), sendcounts.data(), sdispls.data(), MPI_UNSIGNED, allN.data(),recvcounts.data(), rdispls.data(), MPI_UNSIGNED,worldcomm);
    }
};

CUBISM_NAMESPACE_END
