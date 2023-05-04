#pragma once

#include "BlockInfo.h"
#include "LoadBalancer.h"

namespace cubism
{

template <typename TLab>
class MeshAdaptation
{
 protected:
   typedef typename TLab::GridType TGrid;
   typedef typename TGrid::Block BlockType;
   typedef typename TGrid::BlockType::ElementType ElementType;
   typedef SynchronizerMPI_AMR<Real,TGrid> SynchronizerMPIType;

   struct stencilWrapper{StencilInfo stencil;};
   stencilWrapper kernel;

   bool CallValidStates;
   bool boundary_needed;
   LoadBalancer<TGrid> *Balancer;
   TGrid *grid;
   TLab *labs;
   double time;
   bool LabsPrepared;
   bool basic_refinement;
   double tolerance_for_refinement;
   double tolerance_for_compression;
   std::vector<long long> dealloc_IDs;

 public:
   MeshAdaptation(TGrid &g, double Rtol, double Ctol)
   {
      labs = nullptr;
      grid = &g;

      tolerance_for_refinement  = Rtol;
      tolerance_for_compression = Ctol;

      boundary_needed = false;

      constexpr int Gx = 1;
      constexpr int Gy = 1;
      constexpr int Gz = DIMENSION == 3? 1:0;
      kernel.stencil.sx = -Gx;
      kernel.stencil.sy = -Gy;
      kernel.stencil.sz = -Gz;
      kernel.stencil.ex = Gx+1;
      kernel.stencil.ey = Gy+1;
      kernel.stencil.ez = Gz+1;
      kernel.stencil.tensorial = true;
      for (int i = 0 ; i < ElementType::DIM ; i++)
         kernel.stencil.selcomponents.push_back(i);

      Balancer = new LoadBalancer<TGrid>(*grid);
   }

   virtual ~MeshAdaptation() {delete Balancer;}

   void Tag(double t = 0)
   {
      time = t;
      boundary_needed = true;

      SynchronizerMPI_AMR<Real,TGrid> * Synch = grid->sync(kernel);

      const int nthreads = omp_get_max_threads();
      labs = new TLab[nthreads];
      for (int i = 0; i < nthreads; i++) labs[i].prepare(*grid, Synch->getstencil());

      CallValidStates = false;
      bool Reduction = false;
      MPI_Request Reduction_req;
      int tmp;

      std::vector<BlockInfo * > & inner = Synch->avail_inner();
      TagBlocksVector(inner, Reduction, Reduction_req, tmp);

      std::vector<BlockInfo * > & halo  = Synch->avail_halo();
      TagBlocksVector(halo , Reduction, Reduction_req, tmp);

      LabsPrepared = true;

      if (!Reduction)
      {
         tmp = CallValidStates ? 1 : 0;
         Reduction = true;
         MPI_Iallreduce(MPI_IN_PLACE, &tmp, 1, MPI_INT, MPI_SUM, grid->getWorldComm(),&Reduction_req);
      }

      MPI_Wait(&Reduction_req,MPI_STATUS_IGNORE);
      CallValidStates     = (tmp > 0);

      grid->boundary = halo;

      if (CallValidStates) ValidStates();
   }

   void Adapt(double t = 0, bool verbosity = false, bool basic = false)
   {
      basic_refinement = basic;
      if (LabsPrepared == false && basic == false)
      {
         SynchronizerMPI_AMR<Real,TGrid> * Synch = grid->sync(kernel);
         const int nthreads = omp_get_max_threads();
         labs = new TLab[nthreads];
         for (int i = 0; i < nthreads; i++) labs[i].prepare(*grid, Synch->getstencil());
         //TODO: the line below means there's no computation & communication overlap here
         grid->boundary  = Synch->avail_halo();
         if (boundary_needed)
           grid->UpdateBoundary();
      }

      int r = 0;
      int c = 0;

      std::vector<int> m_com;
      std::vector<int> m_ref;
      std::vector<long long> n_com;
      std::vector<long long> n_ref;

      std::vector<BlockInfo> &I = grid->getBlocksInfo();

      long long blocks_after = I.size();

      for (auto &info : I)
      {
         if (info.state == Refine)
         {
            m_ref.push_back(info.level);
            n_ref.push_back(info.Z);
            blocks_after += (1 << DIMENSION) - 1; 
            r++;
         }
         else if (info.state == Compress
            && info.index[0]%2 == 0
            && info.index[1]%2 == 0
            && info.index[2]%2 == 0)
         {
            m_com.push_back(info.level);
            n_com.push_back(info.Z);
            c++;
         }
         else if (info.state == Compress)
         {
            blocks_after --;
         }
      }
      MPI_Request requests[2];
      int temp[2] = {r, c};
      int result[2];
      int size;
      MPI_Comm_size(grid->getWorldComm(),&size);
      std::vector<long long> block_distribution(size);
      MPI_Iallreduce(&temp, &result, 2, MPI_INT, MPI_SUM, grid->getWorldComm(),&requests[0]);
      MPI_Iallgather(&blocks_after, 1, MPI_LONG_LONG, block_distribution.data(), 1, MPI_LONG_LONG, grid->getWorldComm(), &requests[1]);

      dealloc_IDs.clear();

      #ifdef CUBISM_USE_ONETBB
      #pragma omp parallel
      #endif
      {
         #ifdef CUBISM_USE_ONETBB
         #pragma omp for
         #endif
         for (size_t i = 0; i < m_ref.size(); i++)
         {
            refine_1(m_ref[i], n_ref[i]);
         }
         #ifdef CUBISM_USE_ONETBB
         #pragma omp for
         #endif
         for (size_t i = 0; i < m_ref.size(); i++)
         {
            refine_2(m_ref[i], n_ref[i]);
         }
      }
      grid->dealloc_many(dealloc_IDs);

      Balancer->PrepareCompression();

      dealloc_IDs.clear();

      #ifdef CUBISM_USE_ONETBB
      #pragma omp parallel for
      #endif
      for (size_t i = 0; i < m_com.size(); i++)
      {
         compress(m_com[i], n_com[i]);
      }

      grid->dealloc_many(dealloc_IDs);

      MPI_Waitall(2,requests,MPI_STATUS_IGNORE);
      if (verbosity)
      {
         std::cout << "==============================================================\n";
         std::cout << " refined:" << result[0] << "   compressed:" << result[1] << std::endl;
         std::cout << "==============================================================" << std::endl;
      }

      Balancer->Balance_Diffusion(verbosity,block_distribution);

      if (labs != nullptr)
      {
         delete[] labs;
         labs = nullptr;
      }

      if ( result[0] > 0 || result[1] > 0 || Balancer->movedBlocks)
      {
         grid->UpdateFluxCorrection = true;
         grid->UpdateGroups = true;

         grid->UpdateBlockInfoAll_States(false);

         auto it = grid->SynchronizerMPIs.begin();
         while (it != grid->SynchronizerMPIs.end())
         {
            (*it->second)._Setup();
            it++;
         }
      }
      LabsPrepared = false;
   }

   void TagLike(const std::vector<BlockInfo> & I1)
   {
      std::vector<BlockInfo> &I2 = grid->getBlocksInfo();
      for (size_t i1 = 0; i1 < I2.size(); i1++)
      {
         BlockInfo &ary0      = I2[i1];
         BlockInfo &info      = grid->getBlockInfoAll(ary0.level, ary0.Z);
         for (int i = 2 * (info.index[0] / 2); i <= 2 * (info.index[0] / 2) + 1; i++)
         for (int j = 2 * (info.index[1] / 2); j <= 2 * (info.index[1] / 2) + 1; j++)
         #if DIMENSION == 3
         for (int k = 2 * (info.index[2] / 2); k <= 2 * (info.index[2] / 2) + 1; k++)
         {
            const long long n = grid->getZforward(info.level, i, j, k);
            BlockInfo &infoNei = grid->getBlockInfoAll(info.level, n);
            infoNei.state = Leave;
         }
         #else
         {
            const long long n = grid->getZforward(info.level, i, j);
            BlockInfo &infoNei = grid->getBlockInfoAll(info.level, n);
            infoNei.state = Leave;
         }
         #endif
         info.state = Leave;
         ary0.state = Leave;
      }
      #pragma omp parallel for
      for (size_t i = 0 ; i < I1.size(); i++)
      {
         const BlockInfo & info1 = I1[i];
         BlockInfo & info2 = I2[i];
         BlockInfo & info3 = grid->getBlockInfoAll(info2.level, info2.Z);
         info2.state = info1.state;
         info3.state = info1.state;
         if (info2.state == Compress)
         {
            const int i2 = 2 * (info2.index[0] / 2);
            const int j2 = 2 * (info2.index[1] / 2);
            #if DIMENSION == 3
            const int k2 = 2 * (info2.index[2] / 2);
            const long long n = grid->getZforward(info2.level, i2, j2, k2);
            #else
            const long long n = grid->getZforward(info2.level, i2, j2);
            #endif
            BlockInfo &infoNei = grid->getBlockInfoAll(info2.level, n);
            infoNei.state = Compress;
         }
      }
      LabsPrepared = false;
   }

 protected:
   void TagBlocksVector(std::vector<BlockInfo *> & I, bool & Reduction, MPI_Request & Reduction_req, int & tmp)
   {
      const int levelMax = grid->getlevelMax();
      #pragma omp parallel
      {
         const int tid = omp_get_thread_num();
         #pragma omp for schedule(dynamic, 1)
         for (size_t i = 0; i < I.size(); i++)
         {
            labs[tid].load(*I[i], time);

            BlockInfo &info = grid->getBlockInfoAll(I[i]->level, I[i]->Z);

            I[i]->state     = TagLoadedBlock(info);

            const bool maxLevel = (I[i]->state == Refine  ) && (I[i]->level == levelMax - 1);
            const bool minLevel = (I[i]->state == Compress) && (I[i]->level == 0);

            if (maxLevel || minLevel) I[i]->state = Leave;

            info.state = I[i]->state;
            if (info.state != Leave)
            {
               #pragma omp critical
               {
                  CallValidStates = true;
                  if (!Reduction) 
                  {
                     tmp = 1;
                     Reduction = true;
                     MPI_Iallreduce(MPI_IN_PLACE, &tmp, 1, MPI_INT, MPI_SUM, grid->getWorldComm(),&Reduction_req);
                  }
               }
            }
         }
      }
   }

   void refine_1(const int level, const long long Z)
   {
      const int tid = omp_get_thread_num();

      BlockInfo &parent = grid->getBlockInfoAll(level, Z);
      parent.state      = Leave;
      if (basic_refinement == false)
         labs[tid].load(parent, time, true);

      const int p[3] = {parent.index[0], parent.index[1], parent.index[2]};

      assert(parent.ptrBlock != NULL);
      assert(level <= grid->getlevelMax() - 1);
      #if DIMENSION == 3
         BlockType *Blocks[8];
         for (int k = 0; k < 2; k++)
         for (int j = 0; j < 2; j++)
         for (int i = 0; i < 2; i++)
         {
            const long long nc = grid->getZforward(level + 1, 2 * p[0] + i, 2 * p[1] + j, 2 * p[2] + k);
            BlockInfo &Child   = grid->getBlockInfoAll(level + 1, nc);
            Child.state        = Leave;
            grid->_alloc(level + 1, nc);
            grid->Tree(level + 1, nc).setCheckCoarser();
            Blocks[k * 4 + j * 2 + i] = (BlockType *)Child.ptrBlock;
         }
      #else
         BlockType *Blocks[4];
         for (int j = 0; j < 2; j++)
         for (int i = 0; i < 2; i++)
         {
            const long long nc = grid->getZforward(level + 1, 2 * p[0] + i, 2 * p[1] + j);
            BlockInfo &Child = grid->getBlockInfoAll(level + 1, nc);
            Child.state = Leave;
            grid->_alloc(level + 1, nc);
            grid->Tree(level + 1, nc).setCheckCoarser();
            Blocks[j * 2 + i] = (BlockType *)Child.ptrBlock;
         }
      #endif
      if (basic_refinement == false)
         RefineBlocks(Blocks);
   }

   void refine_2(const int level, const long long Z)
   {
      #pragma omp critical
      {
         dealloc_IDs.push_back(grid->getBlockInfoAll(level, Z).blockID_2);
      }

      BlockInfo &parent = grid->getBlockInfoAll(level, Z);
      grid->Tree(parent).setCheckFiner();
      parent.state = Leave;

      int p[3] = {parent.index[0], parent.index[1], parent.index[2]};
      #if DIMENSION == 3
         for (int k = 0; k < 2; k++)
         for (int j = 0; j < 2; j++)
         for (int i = 0; i < 2; i++)
         {
            const long long nc = grid->getZforward(level + 1, 2 * p[0] + i, 2 * p[1] + j, 2 * p[2] + k);
            BlockInfo &Child   = grid->getBlockInfoAll(level + 1, nc);
            grid->Tree(Child).setrank(grid->rank());
            if (level + 2 < grid->getlevelMax())
               for (int i0 = 0; i0 < 2; i0++)
               for (int i1 = 0; i1 < 2; i1++)
               for (int i2 = 0; i2 < 2; i2++)
                  grid->Tree(level + 2, Child.Zchild[i0][i1][i2]).setCheckCoarser();
         }
      #else
         for (int j = 0; j < 2; j++)
         for (int i = 0; i < 2; i++)
         {
            const long long nc = grid->getZforward(level + 1, 2 * p[0] + i, 2 * p[1] + j);
            BlockInfo &Child   = grid->getBlockInfoAll(level + 1, nc);
            grid->Tree(Child).setrank(grid->rank());
            if (level + 2 < grid->getlevelMax())
               for (int i0 = 0; i0 < 2; i0++)
               for (int i1 = 0; i1 < 2; i1++) 
                  grid->Tree(level + 2, Child.Zchild[i0][i1][1]).setCheckCoarser();
         }
      #endif
   }

   void compress(const int level, const long long Z)
   {
      assert(level > 0);

      BlockInfo &info = grid->getBlockInfoAll(level, Z);

      assert(info.state == Compress);

      #if DIMENSION == 3
      BlockType *Blocks[8];
      for (int K = 0; K < 2; K++)
      for (int J = 0; J < 2; J++)
      for (int I = 0; I < 2; I++)
      {
         const int blk = K * 4 + J * 2 + I;
         const long long n = grid->getZforward(level, info.index[0] + I, info.index[1] + J, info.index[2] + K);
         Blocks[blk] = (BlockType *)(grid->getBlockInfoAll(level, n)).ptrBlock;
      }

      const int nx         = BlockType::sizeX;
      const int ny         = BlockType::sizeY;
      const int nz         = BlockType::sizeZ;
      const int offsetX[2] = {0, nx / 2};
      const int offsetY[2] = {0, ny / 2};
      const int offsetZ[2] = {0, nz / 2};
      if (basic_refinement == false)
      for (int K = 0; K < 2; K++)
      for (int J = 0; J < 2; J++)
      for (int I = 0; I < 2; I++)
      {
         BlockType &b = *Blocks[K * 4 + J * 2 + I];
         for (int k = 0; k < nz; k += 2)
         for (int j = 0; j < ny; j += 2)
         for (int i = 0; i < nx; i += 2)
         {
          #ifdef PRESERVE_SYMMETRY
            const ElementType B1 = b(i  ,j  ,k  ) + b(i+1,j+1,k+1);
            const ElementType B2 = b(i+1,j  ,k  ) + b(i  ,j+1,k+1);
            const ElementType B3 = b(i  ,j+1,k  ) + b(i+1,j  ,k+1);
            const ElementType B4 = b(i  ,j  ,k+1) + b(i+1,j+1,k  );
            (*Blocks[0])(i / 2 + offsetX[I], j / 2 + offsetY[J], k / 2 + offsetZ[K]) = 0.125*ConsistentSum<ElementType>(B1,B2,B3,B4);
          #else
            (*Blocks[0])(i / 2 + offsetX[I], j / 2 + offsetY[J], k / 2 + offsetZ[K]) =
                0.125 * ( (b(i  , j  ,k) + b(i+1,j+1,k+1)) 
                        + (b(i+1, j  ,k) + b(i  ,j+1,k+1))
                        + (b(i  , j+1,k) + b(i+1,j  ,k+1))
                        + (b(i+1, j+1,k) + b(i  ,j  ,k+1)) );
          #endif
         }
      }

      const long long np = grid->getZforward(level - 1, info.index[0] / 2, info.index[1] / 2, info.index[2] / 2);
      BlockInfo &parent  = grid->getBlockInfoAll(level - 1, np);
      grid->Tree(parent.level, parent.Z).setrank(grid->rank());
      parent.ptrBlock    = info.ptrBlock;
      parent.state       = Leave;
      if (level - 2 >= 0) grid->Tree(level - 2, parent.Zparent).setCheckFiner();

      for (int K = 0; K < 2; K++)
      for (int J = 0; J < 2; J++)
      for (int I = 0; I < 2; I++)
      {
         const long long n = grid->getZforward(level, info.index[0] + I, info.index[1] + J, info.index[2] + K);
         if (I + J + K == 0)
         {
            grid->FindBlockInfo(level, n, level - 1, np);
         }
         else
         {
            #pragma omp critical
            {
               dealloc_IDs.push_back(grid->getBlockInfoAll(level, n).blockID_2);
            }
         }
         grid->Tree(level, n).setCheckCoarser();
         grid->getBlockInfoAll(level, n).state = Leave;
      }
      #endif
      #if DIMENSION == 2
      BlockType *Blocks[4];
      for (int J = 0; J < 2; J++)
      for (int I = 0; I < 2; I++)
      {
         const int blk     = J * 2 + I;
         const long long n = grid->getZforward(level, info.index[0] + I, info.index[1] + J);
         Blocks[blk]       = (BlockType *)(grid->getBlockInfoAll(level, n)).ptrBlock;
      }

      const int nx         = BlockType::sizeX;
      const int ny         = BlockType::sizeY;
      const int offsetX[2] = {0, nx / 2};
      const int offsetY[2] = {0, ny / 2};
      if (basic_refinement == false)
      for (int J = 0; J < 2; J++)
      for (int I = 0; I < 2; I++)
      {
         BlockType &b = *Blocks[J * 2 + I];
         for (int j = 0; j < ny; j += 2)
         for (int i = 0; i < nx; i += 2)
         {
            ElementType average = 0.25 * ( (b(i, j, 0)+ b(i + 1, j + 1, 0)) + (b(i + 1, j, 0) + b(i, j + 1, 0)) );
            (*Blocks[0])(i / 2 + offsetX[I], j / 2 + offsetY[J], 0) = average;
         }
      }
      const long long np = grid->getZforward(level - 1, info.index[0] / 2, info.index[1] / 2);
      BlockInfo &parent  = grid->getBlockInfoAll(level - 1, np);
      grid->Tree(parent.level, parent.Z).setrank(grid->rank());
      parent.ptrBlock    = info.ptrBlock;
      parent.state       = Leave;
      if (level - 2 >= 0) grid->Tree(level - 2, parent.Zparent).setCheckFiner();

      for (int J = 0; J < 2; J++)
      for (int I = 0; I < 2; I++)
      {
         const long long n = grid->getZforward(level, info.index[0] + I, info.index[1] + J);
         if (I + J == 0)
         {
            grid->FindBlockInfo(level, n, level - 1, np);
         }
         else
         {
            #pragma omp critical
            {
               dealloc_IDs.push_back(grid->getBlockInfoAll(level, n).blockID_2);
            }
         }
         grid->Tree(level, n).setCheckCoarser();
         grid->getBlockInfoAll(level, n).state = Leave;
      }
      #endif
   }

   void ValidStates()
   {
      const std::array<int, 3> blocksPerDim = grid->getMaxBlocks();
      const int levelMin              = 0;
      const int levelMax              = grid->getlevelMax();
      const bool xperiodic            = grid->xperiodic;
      const bool yperiodic            = grid->yperiodic;
      const bool zperiodic            = grid->zperiodic;

      std::vector<BlockInfo> &I = grid->getBlocksInfo();

      #pragma omp parallel for
      for (size_t j = 0; j < I.size(); j++)
      {
         BlockInfo &info = I[j];

         if ((info.state == Refine && info.level == levelMax - 1) || (info.state == Compress && info.level == levelMin))
         {
            info.state                                             = Leave;
            (grid->getBlockInfoAll(info.level, info.Z)).state = Leave;
         }
         if (info.state != Leave)
         {
            info.changed2                                             = true;
            (grid->getBlockInfoAll(info.level, info.Z)).changed2 = info.changed2;
         }
      }

      // 1.Change states of blocks next to finer resolution blocks
      // 2.Change states of blocks next to same resolution blocks
      // 3.Compress a block only if all blocks with the same parent need compression
      bool clean_boundary = true;
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
                  if (info.state == Refine) break;
                  if (icode == 1 * 1 + 3 * 1 + 9 * 1) continue;
                  const int code[3] = {icode % 3 - 1, (icode / 3) % 3 - 1, (icode / 9) % 3 - 1};
                  if (!xperiodic && code[0] == xskip && xskin) continue;
                  if (!yperiodic && code[1] == yskip && yskin) continue;
                  if (!zperiodic && code[2] == zskip && zskin) continue;
                  #if DIMENSION == 2
                  if (code[2] != 0) continue;
                  #endif

                  if (grid->Tree(info.level, info.Znei_(code[0], code[1], code[2])).CheckFiner())
                  {
                     if (info.state == Compress)
                     {
                        info.state                                             = Leave;
                        (grid->getBlockInfoAll(info.level, info.Z)).state = Leave;
                     }
                     // if (info.level == levelMax - 1) break;

                     const int tmp = abs(code[0]) + abs(code[1]) + abs(code[2]); 
                     int Bstep = 1;// face
                     if (tmp == 2) Bstep = 3; //edge
                     else if (tmp == 3) Bstep = 4; //corner                                                    

                     //loop over blocks that make up face/edge/corner(respectively 4,2 or 1 blocks)
                     #if DIMENSION == 3
                     for (int B = 0; B <= 3; B += Bstep)
                     #else
                     for (int B = 0; B <= 1; B += Bstep)
                     #endif
                     {
                        const int aux = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
                        const int iNei = 2 * info.index[0] + max(code[0], 0) + code[0] + (B % 2) * max(0, 1 - abs(code[0]));
                        const int jNei = 2 * info.index[1] + max(code[1], 0) + code[1] + aux * max(0, 1 - abs(code[1]));
                        #if DIMENSION == 3
                           const int kNei = 2 * info.index[2] + max(code[2], 0) + code[2] + (B / 2) * max(0, 1 - abs(code[2]));
                           const long long zzz = grid->getZforward(m + 1, iNei, jNei, kNei);
                        #else
                           const long long zzz = grid->getZforward(m + 1, iNei, jNei);
                        #endif
                        BlockInfo &FinerNei = grid->getBlockInfoAll(m + 1, zzz);
                        State NeiState      = FinerNei.state;
                        if (NeiState == Refine)
                        {
                           info.state                                                = Refine;
                           (grid->getBlockInfoAll(info.level, info.Z)).state    = Refine;
                           info.changed2                                             = true;
                           (grid->getBlockInfoAll(info.level, info.Z)).changed2 = true;
                           break;
                        }
                     }
                  }
               }
            }
         }

         grid->UpdateBoundary(clean_boundary);
         clean_boundary = false;
         if (m == levelMin) break;

         // 2.
         for (size_t j = 0; j < I.size(); j++)
         {
            BlockInfo &info = I[j];
            if (info.level == m && info.state == Compress)
            {
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
                  if (!xperiodic && code[0] == xskip && xskin) continue;
                  if (!yperiodic && code[1] == yskip && yskin) continue;
                  if (!zperiodic && code[2] == zskip && zskin) continue;
                  #if DIMENSION == 2
                  if (code[2] != 0) continue;
                  #endif

                  BlockInfo &infoNei = grid->getBlockInfoAll(info.level, info.Znei_(code[0], code[1], code[2]));
                  if (grid->Tree(infoNei).Exists() && infoNei.state == Refine)
                  {
                     info.state                                             = Leave;
                     (grid->getBlockInfoAll(info.level, info.Z)).state = Leave;
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
               const long long n = grid->getZforward(m, i, j, k);
            #else
               const long long n = grid->getZforward(m, i, j);
            #endif
            BlockInfo &infoNei = grid->getBlockInfoAll(m, n);
            if (grid->Tree(infoNei).Exists() == false || infoNei.state != Compress)
            {
               found = true;
               if (info.state == Compress)
               {
                  info.state                                             = Leave;
                  (grid->getBlockInfoAll(info.level, info.Z)).state = Leave;
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
                  const long long n = grid->getZforward(m, i, j, k);
               #else
                  const long long n = grid->getZforward(m, i, j);
               #endif
               BlockInfo &infoNei = grid->getBlockInfoAll(m, n);
               if (grid->Tree(infoNei).Exists() && infoNei.state == Compress) infoNei.state = Leave;
            }
      }
   }

   ////////////////////////////////////////////////////////////////////////////////////////////////
   // Virtual functions that can be overwritten by user
   ////////////////////////////////////////////////////////////////////////////////////////////////

   //How cells are interpolated after refinement
   virtual void RefineBlocks(BlockType *B[8])
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
                        const int x = i / 2 + offsetX[I];
                        const int y = j / 2 + offsetY[J];
                        const int z = k / 2 + offsetZ[K];
                        const ElementType dudx  = 0.5 * (Lab(x+1,y,z) - Lab(x-1,y,z));
                        const ElementType dudy  = 0.5 * (Lab(x,y+1,z) - Lab(x,y-1,z));
                        const ElementType dudz  = 0.5 * (Lab(x,y,z+1) - Lab(x,y,z-1));
                        const ElementType dudx2 = (Lab(x+1,y,z)  + Lab(x-1,y,z)) - 2.0*Lab(x,y,z);
                        const ElementType dudy2 = (Lab(x,y+1,z)  + Lab(x,y-1,z)) - 2.0*Lab(x,y,z);
                        const ElementType dudz2 = (Lab(x,y,z+1)  + Lab(x,y,z-1)) - 2.0*Lab(x,y,z);
                        const ElementType dudxdy = 0.25*((Lab(x+1,y+1,z)+Lab(x-1,y-1,z)) - (Lab(x+1,y-1,z)+Lab(x-1,y+1,z)));
                        const ElementType dudxdz = 0.25*((Lab(x+1,y,z+1)+Lab(x-1,y,z-1)) - (Lab(x+1,y,z-1)+Lab(x-1,y,z+1)));
                        const ElementType dudydz = 0.25*((Lab(x,y+1,z+1)+Lab(x,y-1,z-1)) - (Lab(x,y+1,z-1)+Lab(x,y-1,z+1)));

                        #ifdef PRESERVE_SYMMETRY
                        const ElementType d2 = 0.03125 * ConsistentSum<ElementType>(dudx2,dudy2,dudz2);
                        b(i  , j  , k  ) = Lab(x,y,z) + (0.25*ConsistentSum<ElementType>(-(1.0)*dudx,-(1.0)*dudy,-(1.0)*dudz) + d2) + 0.0625*ConsistentSum(       dudxdy,       dudxdz,       dudydz);
                        b(i+1, j  , k  ) = Lab(x,y,z) + (0.25*ConsistentSum<ElementType>(       dudx,-(1.0)*dudy,-(1.0)*dudz) + d2) + 0.0625*ConsistentSum(-(1.0)*dudxdy,-(1.0)*dudxdz,       dudydz);
                        b(i  , j+1, k  ) = Lab(x,y,z) + (0.25*ConsistentSum<ElementType>(-(1.0)*dudx,       dudy,-(1.0)*dudz) + d2) + 0.0625*ConsistentSum(-(1.0)*dudxdy,       dudxdz,-(1.0)*dudydz);
                        b(i+1, j+1, k  ) = Lab(x,y,z) + (0.25*ConsistentSum<ElementType>(       dudx,       dudy,-(1.0)*dudz) + d2) + 0.0625*ConsistentSum(       dudxdy,-(1.0)*dudxdz,-(1.0)*dudydz);
                        b(i  , j  , k+1) = Lab(x,y,z) + (0.25*ConsistentSum<ElementType>(-(1.0)*dudx,-(1.0)*dudy,       dudz) + d2) + 0.0625*ConsistentSum(       dudxdy,-(1.0)*dudxdz,-(1.0)*dudydz);
                        b(i+1, j  , k+1) = Lab(x,y,z) + (0.25*ConsistentSum<ElementType>(       dudx,-(1.0)*dudy,       dudz) + d2) + 0.0625*ConsistentSum(-(1.0)*dudxdy,       dudxdz,-(1.0)*dudydz);
                        b(i  , j+1, k+1) = Lab(x,y,z) + (0.25*ConsistentSum<ElementType>(-(1.0)*dudx,       dudy,       dudz) + d2) + 0.0625*ConsistentSum(-(1.0)*dudxdy,-(1.0)*dudxdz,       dudydz);
                        b(i+1, j+1, k+1) = Lab(x,y,z) + (0.25*ConsistentSum<ElementType>(       dudx,       dudy,       dudz) + d2) + 0.0625*ConsistentSum(       dudxdy,       dudxdz,       dudydz);
                        #else
                        b(i  , j  , k  ) = Lab(x,y,z) + 0.25*(-(1.0)* dudx - dudy - dudz) + 0.03125 *(dudx2+dudy2+dudz2) + 0.0625*(       dudxdy + dudxdz + dudydz);
                        b(i+1, j  , k  ) = Lab(x,y,z) + 0.25*(        dudx - dudy - dudz) + 0.03125 *(dudx2+dudy2+dudz2) + 0.0625*(-(1.0)*dudxdy - dudxdz + dudydz);
                        b(i  , j+1, k  ) = Lab(x,y,z) + 0.25*(-(1.0)* dudx + dudy - dudz) + 0.03125 *(dudx2+dudy2+dudz2) + 0.0625*(-(1.0)*dudxdy + dudxdz - dudydz);
                        b(i+1, j+1, k  ) = Lab(x,y,z) + 0.25*(        dudx + dudy - dudz) + 0.03125 *(dudx2+dudy2+dudz2) + 0.0625*(       dudxdy - dudxdz - dudydz);
                        b(i  , j  , k+1) = Lab(x,y,z) + 0.25*(-(1.0)* dudx - dudy + dudz) + 0.03125 *(dudx2+dudy2+dudz2) + 0.0625*(       dudxdy - dudxdz - dudydz);
                        b(i+1, j  , k+1) = Lab(x,y,z) + 0.25*(        dudx - dudy + dudz) + 0.03125 *(dudx2+dudy2+dudz2) + 0.0625*(-(1.0)*dudxdy + dudxdz - dudydz);
                        b(i  , j+1, k+1) = Lab(x,y,z) + 0.25*(-(1.0)* dudx + dudy + dudz) + 0.03125 *(dudx2+dudy2+dudz2) + 0.0625*(-(1.0)*dudxdy - dudxdz + dudydz);
                        b(i+1, j+1, k+1) = Lab(x,y,z) + 0.25*(        dudx + dudy + dudz) + 0.03125 *(dudx2+dudy2+dudz2) + 0.0625*(       dudxdy + dudxdz + dudydz);
                        #endif
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

                  ElementType dudx2 = (Lab(i/2+offsetX[I]+1, j/2+offsetY[J]  )  + Lab(i/2+offsetX[I]-1,j/2+offsetY[J]  ))- 2.0*Lab(i/2+offsetX[I], j/2+offsetY[J]);
                  ElementType dudy2 = (Lab(i/2+offsetX[I]  , j/2+offsetY[J]+1)  + Lab(i/2+offsetX[I]  ,j/2+offsetY[J]-1))- 2.0*Lab(i/2+offsetX[I], j/2+offsetY[J]);  

                  ElementType dudxdy = 0.25*( (Lab(i/2+offsetX[I]+1, j/2+offsetY[J]+1)+Lab(i/2+offsetX[I]-1, j/2+offsetY[J]-1))
                                             -(Lab(i/2+offsetX[I]+1, j/2+offsetY[J]-1)+Lab(i/2+offsetX[I]-1, j/2+offsetY[J]+1)));

                  b(i  ,j  ,0) = (Lab(i/2+offsetX[I], j/2+offsetY[J]) + (- 0.25*dudx - 0.25*dudy) ) + ( (0.03125*dudx2 + 0.03125*dudy2) + 0.0625*dudxdy);
                  b(i+1,j  ,0) = (Lab(i/2+offsetX[I], j/2+offsetY[J]) + (+ 0.25*dudx - 0.25*dudy) ) + ( (0.03125*dudx2 + 0.03125*dudy2) - 0.0625*dudxdy);
                  b(i  ,j+1,0) = (Lab(i/2+offsetX[I], j/2+offsetY[J]) + (- 0.25*dudx + 0.25*dudy) ) + ( (0.03125*dudx2 + 0.03125*dudy2) - 0.0625*dudxdy);
                  b(i+1,j+1,0) = (Lab(i/2+offsetX[I], j/2+offsetY[J]) + (+ 0.25*dudx + 0.25*dudy) ) + ( (0.03125*dudx2 + 0.03125*dudy2) + 0.0625*dudxdy);
               }
         }
      #endif
   }

   //Refinement criterion
   virtual State TagLoadedBlock(BlockInfo &info)
   {
      const int nx = BlockType::sizeX;
      const int ny = BlockType::sizeY;
      BlockType & b = *(BlockType *)info.ptrBlock;

      double Linf = 0.0;
      #if DIMENSION == 3
      const int nz = BlockType::sizeZ;
      for (int k = 0; k < nz; k++)
         for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++)
            {
               Linf = max(Linf, std::fabs(b(i, j, k).magnitude()));
            }
      #endif
      #if DIMENSION == 2
      for (int j = 0; j < ny; j++)
         for (int i = 0; i < nx; i++)
         {
            Linf = max(Linf, std::fabs(b(i, j).magnitude()));
         }
      #endif

      if (Linf > tolerance_for_refinement) return Refine;
      else if (Linf < tolerance_for_compression)
         return Compress;

      return Leave;
   }
};

} // namespace cubism
