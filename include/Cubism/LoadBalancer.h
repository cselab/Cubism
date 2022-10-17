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
   /*
    * This class will redistribute Blocks among different MPI ranks for two reasons:
    * 1) Eight (in 3D) or four (in 2D) blocks need to be compressed to one, but they are owned by
    *    different ranks. PrepareCompression() will collect them all to one rank, so that 
    *    they can be compressed.
    * 2) There is a load imbalance after the grid is refined or compressed. If the imbalance is 
    *    not great (load imbalance ratio < 1.1), a 1D-diffusion based scheme is used to redistribute
    *    blocks along the 1D Space-Filling-Curve. Otherwise, all blocks are simply evenly 
    *    redistributed among all ranks.
    */
 public:
   typedef typename TGrid::Block BlockType;
   typedef typename TGrid::Block::ElementType ElementType;
   bool movedBlocks;

 protected:
   TGrid *grid;
   int rank, size;
   MPI_Comm comm;
   MPI_Datatype MPI_BLOCK;
   struct MPI_Block
   {
      long long mn[2];
      Real data[sizeof(BlockType) / sizeof(Real)];
      MPI_Block(const BlockInfo &info, const bool Fillptr = true)
      {
         mn[0] = info.level;
         mn[1] = info.Z;
         if (Fillptr)
         {
            Real *aux = &((BlockType *)info.ptrBlock)->data[0][0][0].member(0);
            std::memcpy(&data[0], aux, sizeof(BlockType));
         }
      }

      void prepare(const BlockInfo &info, const bool Fillptr = true)
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

   void AddBlock(const int level, const long long Z, Real * data)
   {
      grid->_alloc(level, Z);
      BlockInfo &info = grid->getBlockInfoAll(level, Z);
      BlockType *b1 = (BlockType *)info.ptrBlock;
      assert(b1 != NULL);
      Real *a1 = &b1->data[0][0][0].member(0);
      std::memcpy(a1, data, sizeof(BlockType));
      #if DIMENSION == 3
         int p[3];
         BlockInfo::inverse(Z, level, p[0], p[1], p[2]);
         if (level < grid->getlevelMax() - 1)
            for (int k1 = 0; k1 < 2; k1++)
            for (int j1 = 0; j1 < 2; j1++)
            for (int i1 = 0; i1 < 2; i1++)
            {
               const long long nc = grid->getZforward(level + 1, 2 * p[0] + i1, 2 * p[1] + j1, 2 * p[2] + k1);
               grid->Tree(level + 1, nc).setCheckCoarser();
            }
         if (level > 0)
         {
            const long long nf = grid->getZforward(level - 1, p[0] / 2, p[1] / 2, p[2] / 2);
            grid->Tree(level - 1, nf).setCheckFiner();
         }
      #else
         int p[2];
         BlockInfo::inverse(Z, level, p[0], p[1]);
         if (level < grid->getlevelMax() - 1)
            for (int j1 = 0; j1 < 2; j1++)
            for (int i1 = 0; i1 < 2; i1++)
            {
               const long long nc = grid->getZforward(level + 1, 2 * p[0] + i1, 2 * p[1] + j1);
               grid->Tree(level + 1, nc).setCheckCoarser();
            }
         if (level > 0)
         {
            const long long nf = grid->getZforward(level - 1, p[0] / 2, p[1] / 2);
            grid->Tree(level - 1, nf).setCheckFiner();
         }
      #endif
   }

 public:
   LoadBalancer(TGrid &a_grid)
   {
      grid   = &a_grid;
      comm = grid->getWorldComm();
      MPI_Comm_size(comm, &size);
      MPI_Comm_rank(comm, &rank);
      movedBlocks = false;
      MPI_Block dummy;
      int array_of_blocklengths[2]       = {2, sizeof(BlockType) / sizeof(Real)};
      MPI_Aint array_of_displacements[2] = {0, 2 * sizeof(long long)};
      MPI_Datatype array_of_types[2];
      array_of_types[0] = MPI_LONG_LONG;
      if      (sizeof(Real) == sizeof(float)      ) array_of_types[1] = MPI_FLOAT;
      else if (sizeof(Real) == sizeof(double)     ) array_of_types[1] = MPI_DOUBLE;
      else if (sizeof(Real) == sizeof(long double)) array_of_types[1] = MPI_LONG_DOUBLE;
      MPI_Type_create_struct(2, array_of_blocklengths, array_of_displacements, array_of_types, &MPI_BLOCK);
      MPI_Type_commit(&MPI_BLOCK);
   }

   ~LoadBalancer() { MPI_Type_free(&MPI_BLOCK); }

   void PrepareCompression()
   {
      grid->FillPos();
      std::vector<BlockInfo> &I = grid->getBlocksInfo();
      std::vector<std::vector<MPI_Block>> send_blocks(size);
      std::vector<std::vector<MPI_Block>> recv_blocks(size);

      for (auto &b : I)
      {
         #if DIMENSION == 3
            const long long nBlock = grid->getZforward(b.level, 2 * (b.index[0] / 2), 2 * (b.index[1] / 2), 2 * (b.index[2] / 2));
         #else
            const long long nBlock = grid->getZforward(b.level, 2 * (b.index[0] / 2), 2 * (b.index[1] / 2));
         #endif

         const BlockInfo &base  = grid->getBlockInfoAll(b.level, nBlock);
         const BlockInfo &bCopy = grid->getBlockInfoAll(b.level, b.Z);

         const int baserank = grid->Tree(b.level, nBlock).rank();
         const int brank    = grid->Tree(b.level, b.Z).rank();

         if (!grid->Tree(base).Exists()) continue;

         if (b.Z != nBlock && base.state == Compress)
         {
            if (baserank != rank && brank == rank)
            {
               send_blocks[baserank].push_back({bCopy});
               grid->Tree(b.level, b.Z).setrank(baserank);
            }
         }
         else if (b.Z == nBlock && base.state == Compress)
         {
            #if DIMENSION ==3
               for (int k = 0; k < 2; k++)
               for (int j = 0; j < 2; j++)
               for (int i = 0; i < 2; i++)
               {
                  const long long n = grid->getZforward(b.level, b.index[0] + i, b.index[1] + j, b.index[2] + k);
                  if (n == nBlock) continue;
                  BlockInfo &temp    = grid->getBlockInfoAll(b.level, n);
                  const int temprank = grid->Tree(b.level, n).rank();
                  if (temprank != rank)
                  {
                     recv_blocks[temprank].push_back({temp, false});
                     grid->Tree(b.level, n).setrank(baserank);
                  }
               }
            #else
               for (int j = 0; j < 2; j++)
               for (int i = 0; i < 2; i++)
               {
                  const long long n = grid->getZforward(b.level, b.index[0] + i, b.index[1] + j);
                  if (n == nBlock) continue;
                  BlockInfo &temp    = grid->getBlockInfoAll(b.level, n);
                  const int temprank = grid->Tree(b.level, n).rank();
                  if (temprank != rank)
                  {
                     recv_blocks[temprank].push_back({temp, false});
                     grid->Tree(b.level, n).setrank(baserank);
                  }
               }
            #endif
         }
      }

      std::vector<MPI_Request> requests;

      for (int r = 0; r < size; r++)
         if (r != rank)
         {
            if (recv_blocks[r].size() != 0)
            {
               MPI_Request req{};
               requests.push_back(req);
               MPI_Irecv(&recv_blocks[r][0], recv_blocks[r].size(), MPI_BLOCK, r, 123450, comm, &requests.back());
            }
            if (send_blocks[r].size() != 0)
            {
               MPI_Request req{};
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
            grid->_dealloc(send_blocks[r][i].mn[0], send_blocks[r][i].mn[1]);
            grid->Tree(send_blocks[r][i].mn[0], send_blocks[r][i].mn[1]).setCheckCoarser();
         }

      for (int r = 0; r < size; r++)
         for (int i = 0; i < (int)recv_blocks[r].size(); i++)
         {
            const int level = (int) recv_blocks[r][i].mn[0];
            const long long Z = recv_blocks[r][i].mn[1];

            grid->_alloc(level, Z);
            BlockInfo & info = grid->getBlockInfoAll(level, Z);
            BlockType *b1  = (BlockType *)info.ptrBlock;
            assert(b1 != NULL);
            Real *a1 = &b1->data[0][0][0].member(0);
            std::memcpy(a1, recv_blocks[r][i].data, sizeof(BlockType));
         }
   }

   void Balance_Diffusion(const bool verbose)
   {
      movedBlocks = false;
      {
         const long bsize = grid->getBlocksInfo().size();
         const long b[2] = {bsize,-bsize};
         long res[2];
         MPI_Allreduce(b, res, 2, MPI_LONG, MPI_MAX, comm);
         const long max_b =  res[0];
         const long min_b = -res[1];
         const double ratio = static_cast<double>(max_b) / min_b;
         if (rank == 0 && verbose)
         {
           std::cout << "Load imbalance ratio = " << ratio << std::endl;
         }
         //if (ratio > 1.1 || min_b == 0)
         if (ratio > 1.01 || min_b == 0)
         {
            Balance_Global();
            return;
         }
         //return;
      }

      const int right = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
      const int left  = (rank == 0) ? MPI_PROC_NULL : rank - 1;

      const int my_blocks = grid->getBlocksInfo().size();

      int right_blocks, left_blocks;

      MPI_Request reqs[4];
      MPI_Irecv(&left_blocks, 1, MPI_INT, left, 123, comm, &reqs[0]);
      MPI_Irecv(&right_blocks, 1, MPI_INT, right, 456, comm, &reqs[1]);
      MPI_Isend(&my_blocks, 1, MPI_INT, left, 456, comm, &reqs[2]);
      MPI_Isend(&my_blocks, 1, MPI_INT, right, 123, comm, &reqs[3]);

      MPI_Waitall(4, &reqs[0], MPI_STATUSES_IGNORE);

      int nu         = 4;
      int flux_left  = (my_blocks - left_blocks) / nu;
      int flux_right = (my_blocks - right_blocks) / nu;
      if (rank == size - 1) flux_right = 0;
      if (rank == 0) flux_left = 0;

      std::vector<BlockInfo> SortedInfos = grid->getBlocksInfo();

      if (flux_right != 0 || flux_left != 0) std::sort(SortedInfos.begin(), SortedInfos.end());

      std::vector<MPI_Block> send_left;
      std::vector<MPI_Block> recv_left;
      std::vector<MPI_Block> send_right;
      std::vector<MPI_Block> recv_right;

      std::vector<MPI_Request> request;

      if (flux_left > 0) // then I will send blocks to my left rank
      {
         send_left.resize(flux_left);
         #pragma omp parallel for schedule(runtime)
         for (int i = 0; i < flux_left; i++) send_left[i].prepare(SortedInfos[i]);
         MPI_Request req{};
         request.push_back(req);
         MPI_Isend(&send_left[0], send_left.size(), MPI_BLOCK, left, 7890, comm, &request.back());
      }
      else if (flux_left < 0) // then I will receive blocks from my left rank
      {
         recv_left.resize(abs(flux_left));
         MPI_Request req{};
         request.push_back(req);
         MPI_Irecv(&recv_left[0], recv_left.size(), MPI_BLOCK, left, 4560, comm, &request.back());
      }
      if (flux_right > 0) // then I will send blocks to my right rank
      {
         send_right.resize(flux_right);
         #pragma omp parallel for schedule(runtime)
         for (int i = 0; i < flux_right; i++) send_right[i].prepare(SortedInfos[my_blocks - i - 1]);
         MPI_Request req{};
         request.push_back(req);
         MPI_Isend(&send_right[0], send_right.size(), MPI_BLOCK, right, 4560, comm, &request.back());
      }
      else if (flux_right < 0) // then I will receive blocks from my right rank
      {
         recv_right.resize(abs(flux_right));
	      MPI_Request req{};
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
         grid->_dealloc(info.level, info.Z);
         grid->Tree(info.level, info.Z).setrank(right);
      }

      for (int i = 0; i < flux_left; i++)
      {
         BlockInfo &info = SortedInfos[i];
         grid->_dealloc(info.level, info.Z);
         grid->Tree(info.level, info.Z).setrank(left);
      }

      for (int i = 0; i < -flux_left; i++)
         AddBlock(recv_left[i].mn[0],recv_left[i].mn[1],recv_left[i].data);

      for (int i = 0; i < -flux_right; i++)
         AddBlock(recv_right[i].mn[0],recv_right[i].mn[1],recv_right[i].data);

      int temp = movedBlocks ? 1 : 0;
      MPI_Allreduce(MPI_IN_PLACE, &temp, 1, MPI_INT, MPI_SUM, comm);
      movedBlocks = (temp >= 1);
      #if 0
         if (movedBlocks == true)
         {
            int b = grid->getBlocksInfo().size();
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
      grid->FillPos();
   }

   void Balance_Global()
   {
      const long long b = grid->getBlocksInfo().size();
      std::vector<long long> all_b(size);
      MPI_Allgather(&b, 1, MPI_LONG_LONG, all_b.data(), 1, MPI_LONG_LONG, comm);

      std::vector<BlockInfo> SortedInfos = grid->getBlocksInfo();
      std::sort(SortedInfos.begin(), SortedInfos.end());

      std::vector<std::vector<MPI_Block>> send_blocks(size);
      std::vector<std::vector<MPI_Block>> recv_blocks(size);

      long long total_load = 0;
      for (int r = 0; r < size; r++) total_load += all_b[r];
      long long my_load = total_load / size;
      if (rank < (total_load % size)) my_load += 1;

      std::vector<long long> index_start(size);
      index_start[0] = 0;
      for (int r = 1; r < size; r++) index_start[r] = index_start[r - 1] + all_b[r - 1];

      long long ideal_index = (total_load / size) * rank;
      ideal_index += (rank < (total_load % size)) ? rank : (total_load % size);

      for (int r = 0; r < size; r++) if (rank != r)
      {
         { // check if I need to receive blocks
            const long long a1 = ideal_index;
            const long long a2 = ideal_index + my_load - 1;
            const long long b1 = index_start[r];
            const long long b2 = index_start[r] + all_b[r] - 1;
            const long long c1 = max(a1, b1);
            const long long c2 = min(a2, b2);
            if (c2 - c1 + 1 > 0) recv_blocks[r].resize(c2 - c1 + 1);
         }
         { // check if I need to send blocks
            long long other_ideal_index = (total_load / size) * r;
            other_ideal_index += (r < (total_load % size)) ? r : (total_load % size);
            long long other_load = total_load / size;
            if (r < (total_load % size)) other_load += 1;
            const long long a1 = other_ideal_index;
            const long long a2 = other_ideal_index + other_load - 1;
            const long long b1 = index_start[rank];
            const long long b2 = index_start[rank] + all_b[rank] - 1;
            const long long c1 = max(a1, b1);
            const long long c2 = min(a2, b2);
            if (c2 - c1 + 1 > 0) send_blocks[r].resize(c2 - c1 + 1);
         }
      }

      int tag = 12345;
      std::vector<MPI_Request> requests;
      for (int r = 0; r < size; r++) if (recv_blocks[r].size() != 0)
      {
         MPI_Request req{};
         requests.push_back(req);
         MPI_Irecv(recv_blocks[r].data(), recv_blocks[r].size(), MPI_BLOCK, r, tag, comm, &requests.back());
      }

      long long counter_S = 0;
      long long counter_E = 0;
      for (int r = 0; r < rank; r++) if (send_blocks[r].size() != 0)
      {
         for (size_t i = 0; i < send_blocks[r].size(); i++) send_blocks[r][i].prepare(SortedInfos[counter_S + i]);
         counter_S += send_blocks[r].size();
         MPI_Request req{};
         requests.push_back(req);
         MPI_Isend(send_blocks[r].data(), send_blocks[r].size(), MPI_BLOCK, r, tag, comm, &requests.back());
      }
      for (int r = size-1; r > rank; r--) if (send_blocks[r].size() != 0)
      {
         for (size_t i = 0; i < send_blocks[r].size(); i++) send_blocks[r][i].prepare(SortedInfos[SortedInfos.size() - 1 - (counter_E + i)]);
         counter_E += send_blocks[r].size();
         MPI_Request req{};
         requests.push_back(req);
         MPI_Isend(send_blocks[r].data(), send_blocks[r].size(), MPI_BLOCK, r, tag, comm, &requests.back());
      }

      MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

      movedBlocks = true;

      std::vector<long long> deallocIDs;
      counter_S = 0;
      counter_E = 0;
      for (int r = 0; r < size; r++) if (send_blocks[r].size() != 0)
      {
         if (r < rank)
         {
            for (size_t i = 0; i < send_blocks[r].size(); i++)
            {
               BlockInfo &info = SortedInfos[counter_S + i];
               deallocIDs.push_back(info.blockID_2);
               grid->Tree(info.level, info.Z).setrank(r);
            }
            counter_S += send_blocks[r].size();
         }
         else
         {
            for (size_t i = 0; i < send_blocks[r].size(); i++)
            {
               BlockInfo &info = SortedInfos[SortedInfos.size() - 1 - (counter_E + i)];
               deallocIDs.push_back(info.blockID_2);
               grid->Tree(info.level, info.Z).setrank(r);
            }
            counter_E += send_blocks[r].size();
         }
      }
      grid->dealloc_many(deallocIDs);

      #pragma omp parallel
      {
         for (int r = 0; r < size; r++) if (recv_blocks[r].size() != 0)
         {
            #pragma omp for
            for (size_t i = 0; i < recv_blocks[r].size(); i++)
               AddBlock(recv_blocks[r][i].mn[0],recv_blocks[r][i].mn[1],recv_blocks[r][i].data);
         }
      }
      grid->FillPos();
   }
};

} // namespace cubism
