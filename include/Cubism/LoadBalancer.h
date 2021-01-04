#pragma once

#include "BlockInfo.h"
#include "GridMPI.h"
#include <algorithm>
#include <cstring>
#include <omp.h>
#include <string>

namespace cubism
{

template <typename TGrid>
class LoadBalancer
{

 public:
   typedef typename TGrid::Block BlockType;
   typedef typename TGrid::Block::ElementType ElementType;
   bool movedBlocks;

   static int counterg;
   double beta;
   int flux_left_old;
   int flux_right_old;

   MPI_Comm comm;

 protected:
   TGrid *m_refGrid;
   int rank, size;
   MPI_Datatype MPI_BLOCK;
   struct MPI_Block
   {
      int mn[2];
      Real data[sizeof(BlockType) / sizeof(Real)];
      MPI_Block(BlockInfo &info, bool Fillptr = true)
      {
         mn[0] = info.level;
         mn[1] = info.Z;
         if (Fillptr)
         {
            Real *aux = &((BlockType *)info.ptrBlock)->data[0][0][0].member(0);
            std::memcpy(&data[0], aux, sizeof(BlockType));
         }
      }

      void prepare(BlockInfo &info, bool Fillptr = true)
      {
         mn[0] = info.level;
         mn[1] = info.Z;
         if (Fillptr)
         {
            Real *aux = &((BlockType *)info.ptrBlock)->data[0][0][0].member(0);
            std::memcpy(&data[0], aux, sizeof(BlockType));
         }
      }

      MPI_Block() {}
   };

 public:
   LoadBalancer(TGrid &grid)
   {
      comm = grid.getWorldComm();
      MPI_Comm_size(comm, &size);
      MPI_Comm_rank(comm, &rank);
      m_refGrid   = &grid;
      movedBlocks = false;
      MPI_Block dummy;
      int array_of_blocklengths[2]       = {2, sizeof(BlockType) / sizeof(Real)};
      MPI_Aint array_of_displacements[2] = {0, 2 * sizeof(int)};
      MPI_Datatype array_of_types[2]     = {MPI_INT, MPI_DOUBLE};
      MPI_Type_create_struct(2, array_of_blocklengths, array_of_displacements, array_of_types, &MPI_BLOCK);
      MPI_Type_commit(&MPI_BLOCK);
      flux_left_old  = 0;
      flux_right_old = 0;
   }

   ~LoadBalancer() { MPI_Type_free(&MPI_BLOCK); }

   void PrepareCompression()
   {
      m_refGrid->FillPos();
      std::vector<BlockInfo> &I = m_refGrid->getBlocksInfo();
      std::vector<std::vector<MPI_Block>> send_blocks(size);
      std::vector<std::vector<MPI_Block>> recv_blocks(size);

      for (auto &b : I)
      {
         assert(b.level >= 0);
         assert(b.index[0] >= 0);
         assert(b.index[1] >= 0);
         assert(b.index[2] >= 0);

         int nBlock = m_refGrid->getZforward(b.level, 2 * (b.index[0] / 2), 2 * (b.index[1] / 2), 2 * (b.index[2] / 2));

         assert(nBlock >= 0);
         assert(b.Z >= 0);

         BlockInfo &base  = m_refGrid->getBlockInfoAll(b.level, nBlock);
         BlockInfo &bCopy = m_refGrid->getBlockInfoAll(b.level, b.Z);

         assert(b.TreePos == Exists);

         if (base.TreePos != Exists) continue;

         if (b.Z != nBlock && base.state == Compress)
         {
            if (base.myrank != rank && b.myrank == rank)
            {
               assert(base.TreePos == Exists);
               assert(base.myrank >= 0);
               send_blocks[base.myrank].push_back({bCopy});
               b.myrank     = base.myrank;
               bCopy.myrank = base.myrank;
            }
         }
         else if (b.Z == nBlock && base.state == Compress)
         {
            for (int k = 0; k < 2; k++)
            for (int j = 0; j < 2; j++)
            for (int i = 0; i < 2; i++)
            {
               int n = m_refGrid->getZforward(b.level, b.index[0] + i, b.index[1] + j, b.index[2] + k);
               if (n == nBlock) continue;
               BlockInfo &temp = m_refGrid->getBlockInfoAll(b.level, n);
               if (temp.myrank != rank)
               {
                  assert(base.TreePos == Exists);
                  assert(base.myrank >= 0);
                  assert(temp.myrank >= 0);
                  recv_blocks[temp.myrank].push_back({temp, false});
                  temp.myrank = base.myrank;
               }
            }
         }
      }

      int BlockBytes = sizeof(BlockType);

      std::vector<MPI_Request> requests;

      for (int r = 0; r < size; r++)
         if (r != rank)
         {
            if (recv_blocks[r].size() != 0)
            {
               MPI_Request req;
               requests.push_back(req);
               MPI_Irecv(&recv_blocks[r][0], recv_blocks[r].size(), MPI_BLOCK, r, 123450, comm, &requests.back());
            }
            if (send_blocks[r].size() != 0)
            {
               MPI_Request req;
               requests.push_back(req);
               MPI_Isend(&send_blocks[r][0], send_blocks[r].size(), MPI_BLOCK, r, 123450, comm, &requests.back());
            }
         }

      if (requests.size() != 0)
      {
         movedBlocks = true;
         MPI_Waitall(requests.size(), &requests[0], MPI_STATUSES_IGNORE);
      }

      for (int r = 0; r < size; r++)
         for (int i = 0; i < (int)send_blocks[r].size(); i++)
         {
            m_refGrid->_dealloc(send_blocks[r][i].mn[0], send_blocks[r][i].mn[1]);
         }

      for (int r = 0; r < size; r++)
         for (int i = 0; i < (int)recv_blocks[r].size(); i++)
         {
            int level = recv_blocks[r][i].mn[0];
            int Z     = recv_blocks[r][i].mn[1];

            m_refGrid->_alloc(level, Z);

            BlockInfo info = m_refGrid->getBlockInfoAll(level, Z);
            BlockType *b1  = (BlockType *)info.ptrBlock;
            assert(b1 != NULL);
            Real *a1 = &b1->data[0][0][0].member(0);
            std::memcpy(a1, recv_blocks[r][i].data, BlockBytes);
         }
   }

   void Balance_Diffusion()
   {
      counterg++;
      if (counterg < 5 || (counterg % 10 ==0) )
      {
         int b = m_refGrid->getBlocksInfo().size();
         std::vector<int> all_b(size);
         int max_b,min_b;
         MPI_Allreduce(&b, &max_b, 1, MPI_INT,  MPI_MAX, comm);
         MPI_Allreduce(&b, &min_b, 1, MPI_INT,  MPI_MIN, comm);
         const double ratio = max_b / min_b;
         if (rank == 0) std::cout << "Load imbalance ratio = " << ratio << std::endl;
         if (ratio > 1.5)
         {
            Balance_Global();
            counterg = 11;
            return;
         }
      }

      int right = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
      int left  = (rank == 0       ) ? MPI_PROC_NULL : rank - 1;

      int my_blocks = m_refGrid->getBlocksInfo().size();

      int right_blocks, left_blocks;

      std::vector<MPI_Request> reqs(4);
      MPI_Irecv(&left_blocks , 1, MPI_INT, left , 123, comm, &reqs[0]);
      MPI_Irecv(&right_blocks, 1, MPI_INT, right, 456, comm, &reqs[1]);
      MPI_Isend(&my_blocks   , 1, MPI_INT, left , 456, comm, &reqs[2]);
      MPI_Isend(&my_blocks   , 1, MPI_INT, right, 123, comm, &reqs[3]);

      MPI_Waitall(4, &reqs[0], MPI_STATUSES_IGNORE);

      int nu         = 4;
      int flux_left  = (my_blocks - left_blocks ) / nu;
      int flux_right = (my_blocks - right_blocks) / nu;
      if (rank == size - 1) flux_right = 0;
      if (rank == 0       ) flux_left  = 0;

      std::vector<BlockInfo> SortedInfos = m_refGrid->getBlocksInfo();

      if (flux_right != 0 || flux_left != 0) std::sort(SortedInfos.begin(), SortedInfos.end());

      std::vector<MPI_Block> send_left;
      std::vector<MPI_Block> recv_left;
      std::vector<MPI_Block> send_right;
      std::vector<MPI_Block> recv_right;

      int BlockBytes = sizeof(BlockType);
      std::vector<MPI_Request> request;

      if (flux_left > 0) // then I will send blocks to my left rank
      {
         send_left.resize(flux_left);
         #pragma omp parallel for schedule(runtime)
         for (int i = 0; i < flux_left; i++) send_left[i].prepare(SortedInfos[i]);
         MPI_Request req;
         request.push_back(req);
         MPI_Isend(&send_left[0], send_left.size(), MPI_BLOCK, left, 7890, comm, &request.back());
      }
      else if (flux_left < 0) // then I will receive blocks from my left rank
      {
         recv_left.resize(abs(flux_left));
         MPI_Request req;
         request.push_back(req);
         MPI_Irecv(&recv_left[0], recv_left.size(), MPI_BLOCK, left, 4560, comm, &request.back());
      }
      if (flux_right > 0) // then I will send blocks to my right rank
      {
         send_right.resize(flux_right);
         #pragma omp parallel for schedule(runtime)
         for (int i = 0; i < flux_right; i++) send_right[i].prepare(SortedInfos[my_blocks - i - 1]);
         MPI_Request req;
         request.push_back(req);
         MPI_Isend(&send_right[0], send_right.size(), MPI_BLOCK, right, 4560, comm, &request.back());
      }
      else if (flux_right < 0) // then I will receive blocks from my right rank
      {
         recv_right.resize(abs(flux_right));
         MPI_Request req;
         request.push_back(req);
         MPI_Irecv(&recv_right[0], recv_right.size(), MPI_BLOCK, right, 7890, comm, &request.back());
      }

      if (request.size() != 0)
      {
         movedBlocks = true;
         MPI_Waitall(request.size(), &request[0], MPI_STATUSES_IGNORE);
      }

      for (int i = 0; i < flux_right; i++)
      {
         BlockInfo &info = SortedInfos[my_blocks - i - 1];
         m_refGrid->_dealloc(info.level, info.Z);
         BlockInfo &info1 = m_refGrid->getBlockInfoAll(info.level, info.Z);
         info1.myrank     = right;
      }

      for (int i = 0; i < flux_left; i++)
      {
         BlockInfo &info = SortedInfos[i];
         m_refGrid->_dealloc(info.level, info.Z);
         BlockInfo &info1 = m_refGrid->getBlockInfoAll(info.level, info.Z);
         info1.myrank     = left;
      }

      for (int i = 0; i < -flux_left; i++)
      {
         int level = recv_left[i].mn[0];
         int Z     = recv_left[i].mn[1];
         m_refGrid->_alloc(level, Z);
         BlockInfo &info = m_refGrid->getBlockInfoAll(level, Z);
         info.TreePos    = Exists;
         BlockType *b1   = (BlockType *)info.ptrBlock;
         Real *a1        = &b1->data[0][0][0].member(0);
         std::memcpy(a1, recv_left[i].data, BlockBytes);
         assert(m_refGrid->getBlockInfoAll(level, Z).myrank == rank);
      }

      for (int i = 0; i < -flux_right; i++)
      {
         int level = recv_right[i].mn[0];
         int Z     = recv_right[i].mn[1];
         m_refGrid->_alloc(level, Z);
         BlockInfo &info = m_refGrid->getBlockInfoAll(level, Z);
         info.TreePos    = Exists;
         BlockType *b1   = (BlockType *)info.ptrBlock;
         assert(b1 != NULL);
         Real *a1 = &b1->data[0][0][0].member(0);
         std::memcpy(a1, recv_right[i].data, BlockBytes);
         assert(m_refGrid->getBlockInfoAll(level, Z).myrank == rank);
      }

      int temp = movedBlocks ? 1 : 0;
      MPI_Allreduce(MPI_IN_PLACE, &temp, 1, MPI_INT, MPI_LOR, comm);
      movedBlocks = (temp == 1);
      #if 0
      if (movedBlocks == true)
      {
         int b = m_refGrid->getBlocksInfo().size();
         std::vector<int> all_b(size);
         MPI_Gather(&b, 1, MPI_INT, &all_b[0], 1, MPI_INT, 0, comm);

         if (rank == 0)
         {
            std::cout << "&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~\n";
            std::cout << " Distribution of blocks among ranks: \n";
            for (int r = 0; r < size; r++) std::cout << all_b[r] << " | ";
            std::cout << "\n";
            std::cout << "&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~\n";
            std::cout << std::endl;
         }
      }
      #endif
      m_refGrid->FillPos();
      for (auto &info : m_refGrid->getBlocksInfo())
      {
         assert(info.TreePos == Exists);
         assert(info.myrank == rank);
         info.myrank = rank;
      }
      flux_left_old  = flux_left;
      flux_right_old = flux_right;
   }

   void Balance_Global()
   {
      int BlockBytes = sizeof(BlockType);
      int b = m_refGrid->getBlocksInfo().size();
      std::vector<int> all_b(size);
      MPI_Allgather(&b, 1, MPI_INT, all_b.data(), 1, MPI_INT, comm);

      std::vector<BlockInfo> SortedInfos = m_refGrid->getBlocksInfo();
      std::sort(SortedInfos.begin(), SortedInfos.end());

      std::vector< std::vector<MPI_Block> > send_blocks(size);
      std::vector< std::vector<MPI_Block> > recv_blocks(size);

      int total_load = 0;
      for (int r = 0 ; r < size ; r++) total_load+= all_b[r];
      int my_load =  total_load / size;
      if (rank < (total_load % size) ) my_load += 1;

      std::vector<int> index_start(size);
      index_start[0] = 0;
      for (int r = 1 ; r < size ; r++) index_start[r] = index_start[r-1] + all_b[r-1];

      int ideal_index = ( total_load / size ) * rank;
      ideal_index += (rank < (total_load % size)) ? rank : (total_load % size);

      for (int r = 0 ; r < size ; r ++) if (rank != r)
      {
         {  //check if I need to receive blocks
            const int a1 = ideal_index;
            const int a2 = ideal_index + my_load -1;
            const int b1 = index_start[r];
            const int b2 = index_start[r]+all_b[r]-1;
            const int c1 = max(a1,b1);
            const int c2 = min(a2,b2);
            if (c2-c1 + 1>0) recv_blocks[r].resize(c2-c1+1);
         }
         {  //check if I need to send blocks
            int other_ideal_index = ( total_load / size ) * r;
            other_ideal_index += (r < (total_load % size)) ? r : (total_load % size); 
            int other_load =  total_load / size;
            if (r < (total_load%size)) other_load += 1;
            const int a1 = other_ideal_index;
            const int a2 = other_ideal_index + other_load -1;
            const int b1 = index_start[rank];
            const int b2 = index_start[rank]+all_b[rank]-1;
            const int c1 = max(a1,b1);
            const int c2 = min(a2,b2);
            if (c2-c1 + 1>0) send_blocks[r].resize(c2-c1+1);
         }
      }

      std::vector<MPI_Request> recv_request;
      for (int r = 0 ; r < size ; r ++) if (recv_blocks[r].size() != 0)
      {
         MPI_Request req;
         recv_request.push_back(req);
         MPI_Irecv(recv_blocks[r].data(), recv_blocks[r].size(), MPI_BLOCK, r, r*size+rank, comm, &recv_request.back());
      }

      std::vector<MPI_Request> send_request;
      int counter = 0;
      for (int r = 0 ; r < size ; r ++) if (send_blocks[r].size() != 0)
      {
         for (size_t i = 0 ; i < send_blocks[r].size() ; i ++)
            send_blocks[r][i].prepare(SortedInfos[counter + i]);
         counter += send_blocks[r].size();
         MPI_Request req;
         send_request.push_back(req);
         MPI_Isend(send_blocks[r].data(), send_blocks[r].size(), MPI_BLOCK, r, r +rank*size, comm, &send_request.back());
      }

      MPI_Waitall(send_request.size(), send_request.data() , MPI_STATUSES_IGNORE);
      MPI_Waitall(recv_request.size(), recv_request.data() , MPI_STATUSES_IGNORE);

      movedBlocks = true;

      counter = 0;
      for (int r = 0 ; r < size ; r ++) if (send_blocks[r].size() != 0)
      {
         for (size_t i = 0 ; i < send_blocks[r].size() ; i ++)
         {
            BlockInfo &info = SortedInfos[counter+i];
            m_refGrid->_dealloc(info.level, info.Z);
            BlockInfo &info1 = m_refGrid->getBlockInfoAll(info.level, info.Z);
            info1.myrank     = r;
         }
         counter += send_blocks[r].size();
      }
      for (int r = 0 ; r < size ; r ++) if (recv_blocks[r].size() != 0)
      {
         for (size_t i = 0 ; i < recv_blocks[r].size() ; i ++)
         {
            int level = recv_blocks[r][i].mn[0];
            int Z     = recv_blocks[r][i].mn[1];
            m_refGrid->_alloc(level, Z);
  
           BlockInfo &info = m_refGrid->getBlockInfoAll(level, Z);
           info.TreePos    = Exists;
           BlockType *b1   = (BlockType *)info.ptrBlock;
           Real *a1        = &b1->data[0][0][0].member(0);
           std::memcpy(a1, recv_blocks[r][i].data, BlockBytes);
         }
      }
      m_refGrid->FillPos();
      for (auto &info : m_refGrid->getBlocksInfo())
      {
         assert(info.TreePos == Exists);
         assert(info.myrank == rank);
         info.myrank = rank;
      }
      
      m_refGrid->UpdateBlockInfoAll_States(true);//is this call necessary? Probably yes.
   }
};
template <typename TGrid> int LoadBalancer<TGrid>::counterg=0;

} // namespace cubism