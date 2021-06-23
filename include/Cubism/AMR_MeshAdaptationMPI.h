#pragma once
#include "AMR_MeshAdaptation.h"
#include "BlockLabMPI.h"
#include "GridMPI.h"
#include "LoadBalancer.h"

namespace cubism
{

template <typename TGrid, typename TLab, typename otherTGRID = TGrid>
class MeshAdaptationMPI : public MeshAdaptation<TGrid,TLab,otherTGRID>
{

 public:
   typedef typename TGrid::Block BlockType;
   typedef typename TGrid::BlockType::ElementType ElementType;
   typedef SynchronizerMPI_AMR<Real,TGrid> SynchronizerMPIType;
   typedef typename TGrid::BlockType Block;

 protected:

   SynchronizerMPI_AMR<Real,TGrid> *Synch;
   int timestamp;
   bool flag;

   using AMR = MeshAdaptation<TGrid,TLab,otherTGRID>;

 public:
   MeshAdaptationMPI(TGrid &grid, double Rtol, double Ctol): MeshAdaptation<TGrid,TLab,otherTGRID>(grid,Rtol,Ctol)
   {
      bool tensorial = false;

      const int Gx = (WENOWAVELET == 3) ? 1 : 2;
      const int Gy = (WENOWAVELET == 3) ? 1 : 2;
      const int Gz = (WENOWAVELET == 3) ? 1 : 2;

      for (int i = 0 ; i < ElementType::DIM ; i++) AMR::components.push_back(i);

      StencilInfo stencil(-Gx, -Gy, -Gz, Gx + 1, Gy + 1, Gz + 1, tensorial, AMR::components);

      AMR::m_refGrid = &grid;

      AMR::s[0]        = stencil.sx;
      AMR::e[0]        = stencil.ex;
      AMR::s[1]        = stencil.sy;
      AMR::e[1]        = stencil.ey;
      AMR::s[2]        = stencil.sz;
      AMR::e[2]        = stencil.ez;
      AMR::istensorial = stencil.tensorial;

      AMR::Is[0] = stencil.sx;
      AMR::Ie[0] = stencil.ez;
      AMR::Is[1] = stencil.sy;
      AMR::Ie[1] = stencil.ey;
      AMR::Is[2] = stencil.sz;
      AMR::Ie[2] = stencil.ez;


      timestamp = 0;
      flag      = true;
      auto blockperDim     = AMR::m_refGrid->getMaxBlocks();
      StencilInfo Cstencil = stencil;
      Synch                = new SynchronizerMPIType(stencil, Cstencil, AMR::m_refGrid->getlevelMax(), TGrid::Block::sizeX,TGrid::Block::sizeY, TGrid::Block::sizeZ, blockperDim[0], blockperDim[1], blockperDim[2], &grid);
      Synch->_Setup(&(AMR::m_refGrid->getBlocksInfo())[0], (AMR::m_refGrid->getBlocksInfo()).size(), timestamp, true);
   }

   virtual ~MeshAdaptationMPI() { delete Synch; }

   virtual void AdaptTheMesh(double t = 0) override
   {
      const int levelMax = AMR::m_refGrid->getlevelMax();
      const int levelMin = 0;
      AMR::time = t;
      Synch->sync(sizeof(typename Block::element_type) / sizeof(Real),sizeof(Real) > 4 ? MPI_DOUBLE : MPI_FLOAT, timestamp);

      timestamp = (timestamp + 1) % 32768;

      vector<BlockInfo *> avail0, avail1;

      const int nthreads = omp_get_max_threads();

      AMR::labs = new TLab[nthreads];
      for (int i = 0; i < nthreads; i++) AMR::labs[i].prepare(*AMR::m_refGrid, *Synch);

      bool CallValidStates = false;

      avail0           = Synch->avail_inner();
      const int Ninner = avail0.size();
      bool Reduction = false;
      MPI_Request Reduction_req;
      int tmp;
      #pragma omp parallel num_threads(nthreads)
      {
         int tid     = omp_get_thread_num();
         TLab &mylab = AMR::labs[tid];

         #pragma omp for schedule(dynamic, 1)
         for (int i = 0; i < Ninner; i++)
         {
            BlockInfo &ary0 = *avail0[i];
            mylab.load(ary0, t);
            BlockInfo &info = AMR::m_refGrid->getBlockInfoAll(ary0.level, ary0.Z);
            ary0.state      = TagLoadedBlock(AMR::labs[tid],info);
            if ((ary0.state == Refine && ary0.level == levelMax - 1) ||
                (ary0.state == Compress && ary0.level == levelMin))
            {
               ary0.state = Leave;
            }
            info.state = ary0.state;
            #pragma omp critical
            {
                if (info.state != Leave)
                {
                    CallValidStates = true;
                    if (!Reduction) 
                    {
                      tmp = 1;
                      Reduction = true;
                      MPI_Iallreduce(MPI_IN_PLACE, &tmp, 1, MPI_INT, MPI_SUM, AMR::m_refGrid->getWorldComm(),&Reduction_req);
                    }
                }                
            }
         }
      }
      avail1 = Synch->avail_halo();

      const long long Nhalo = avail1.size();
      #pragma omp parallel num_threads(nthreads)
      {
         int tid     = omp_get_thread_num();
         TLab &mylab = AMR::labs[tid];

         #pragma omp for schedule(dynamic, 1)
         for (int i = 0; i < Nhalo; i++)
         {
            BlockInfo &ary1 = *avail1[i];
            mylab.load(ary1, t);
            BlockInfo &info = AMR::m_refGrid->getBlockInfoAll(ary1.level, ary1.Z);
            ary1.state      = TagLoadedBlock(AMR::labs[tid],info);
            if ((ary1.state == Refine && ary1.level == levelMax - 1) ||
                (ary1.state == Compress && ary1.level == levelMin))
            {
               ary1.state = Leave;
            }                       
            info.state = ary1.state;
            #pragma omp critical
            {
                if (info.state != Leave)
                {
                    CallValidStates = true;
                    if (!Reduction) 
                    {
                      tmp = 1;
                      Reduction = true;
                      MPI_Iallreduce(MPI_IN_PLACE, &tmp, 1, MPI_INT, MPI_SUM, AMR::m_refGrid->getWorldComm(),&Reduction_req);
                    }
                }                
            }
         }
      }

      if (!Reduction) 
      {
         tmp = CallValidStates ? 1 : 0;
         Reduction = true;
         MPI_Iallreduce(MPI_IN_PLACE, &tmp, 1, MPI_INT, MPI_SUM, AMR::m_refGrid->getWorldComm(),&Reduction_req);
      }
      LoadBalancer<TGrid> Balancer(*AMR::m_refGrid);

      MPI_Wait(&Reduction_req,MPI_STATUS_IGNORE);
      CallValidStates     = (tmp > 0);
      AMR::m_refGrid->boundary = avail1;

      if (CallValidStates) ValidStates();

      // Refinement/compression of blocks
      /*************************************************/
      int r = 0;
      int c = 0;

      std::vector<int> m_com;
      std::vector<int> m_ref;
      std::vector<long long> n_com;
      std::vector<long long> n_ref;

      std::vector<BlockInfo> &I = AMR::m_refGrid->getBlocksInfo();

      for (auto &info : I)
      {
         if (info.state == Refine)
         {
            m_ref.push_back(info.level);
            n_ref.push_back(info.Z);
         }
         else if (info.state == Compress)
         {
            m_com.push_back(info.level);
            n_com.push_back(info.Z);
         }
      }

      #pragma omp parallel
      {
         #pragma omp for
         for (size_t i = 0; i < m_ref.size(); i++)
         {
            AMR::refine_1(m_ref[i], n_ref[i]);
            #pragma omp atomic
            r++;
         }
         #pragma omp for
         for (size_t i = 0; i < m_ref.size(); i++)
         {
            AMR::refine_2(m_ref[i], n_ref[i]);
         }
      }

      Balancer.PrepareCompression();

      #pragma omp parallel for
      for (size_t i = 0; i < m_com.size(); i++)
      {
         AMR::compress(m_com[i], n_com[i]);
         #pragma omp atomic
         c++;
      }

      #if 1
      int temp[2] = {r, c};
      int result[2];
      MPI_Allreduce(&temp, &result, 2, MPI_INT, MPI_SUM, AMR::m_refGrid->getWorldComm());
      int rank;
      MPI_Comm_rank(AMR::m_refGrid->getWorldComm(), &rank);
      if (rank == 0)
      {
         std::cout << "==============================================================\n";
         std::cout << " refined:" << result[0] << "   compressed:" << result[1] << std::endl;
         std::cout << "==============================================================\n";
      }
      #endif

      AMR::m_refGrid->FillPos();     
      Balancer.Balance_Diffusion();

      delete[] AMR::labs;

      if ( result[0] > 0 || result[1] > 0 || Balancer.movedBlocks)
      {
         AMR::m_refGrid->UpdateFluxCorrection = true;
         AMR::m_refGrid->UpdateGroups = true;

         AMR::m_refGrid->UpdateBlockInfoAll_States(false);

         Synch->_Setup(&(AMR::m_refGrid->getBlocksInfo())[0], (AMR::m_refGrid->getBlocksInfo()).size(), timestamp, true);

         // typename std::map<StencilInfo,SynchronizerMPIType*>::iterator
         auto it = AMR::m_refGrid->SynchronizerMPIs.begin();
         while (it != AMR::m_refGrid->SynchronizerMPIs.end())
         {
            (*it->second)._Setup(&(AMR::m_refGrid->getBlocksInfo())[0], (AMR::m_refGrid->getBlocksInfo()).size(), timestamp);
            it++;
         }
      }
      else
      {
         //AMR::m_refGrid->UpdateFluxCorrection = flag;
         //flag                            = false;
      }
      std::cout << std::flush;
   }

////////////////////////////////////////////////////////////////////////////////////////////////////
   virtual void AdaptLikeOther(otherTGRID &OtherGrid) override
   {
      otherTGRID *m_OtherGrid = &OtherGrid;

      std::vector<BlockInfo> &I = AMR::m_refGrid->getBlocksInfo();

      #pragma omp parallel
      {
         #pragma omp for
         for (size_t i1 = 0; i1 < I.size(); i1++)
         {
            BlockInfo &ary0      = I[i1];
            BlockInfo &info      = AMR::m_refGrid->getBlockInfoAll(ary0.level, ary0.Z);
            for (int i = 2 * (info.index[0] / 2); i <= 2 * (info.index[0] / 2) + 1; i++)
            for (int j = 2 * (info.index[1] / 2); j <= 2 * (info.index[1] / 2) + 1; j++)
            for (int k = 2 * (info.index[2] / 2); k <= 2 * (info.index[2] / 2) + 1; k++)
            {
               const long long n = AMR::m_refGrid->getZforward(info.level, i, j, k);
               BlockInfo &infoNei = AMR::m_refGrid->getBlockInfoAll(info.level, n);
               infoNei.state = Leave;
            }
            info.state = Leave; 
            ary0.state = Leave;
         }

         #pragma omp for
         for (size_t i1 = 0; i1 < I.size(); i1++)
         {
            BlockInfo &ary0      = I[i1];
            BlockInfo &info      = AMR::m_refGrid->getBlockInfoAll(ary0.level, ary0.Z);
            BlockInfo &infoOther = m_OtherGrid->getBlockInfoAll(ary0.level, ary0.Z);
            if      (m_OtherGrid->Tree(infoOther).Exists()      ) ary0.state = Leave;
            else if (m_OtherGrid->Tree(infoOther).CheckFiner()  ) ary0.state = Refine;
            else if (m_OtherGrid->Tree(infoOther).CheckCoarser()) ary0.state = Compress;
            if (ary0.state == Compress)
            {
               const int i = 2 * (info.index[0] / 2);
               const int j = 2 * (info.index[1] / 2);
               const int k = 2 * (info.index[2] / 2);
               const long long n = AMR::m_refGrid->getZforward(info.level, i, j, k);
               BlockInfo &infoNei = AMR::m_refGrid->getBlockInfoAll(info.level, n);
               infoNei.state = Compress;
               if (info.index[0] % 2 == 1 || info.index[1] % 2 == 1 || info.index[2] % 2 == 1)
               {
                  ary0.state = Leave;
                  (AMR::m_refGrid->getBlockInfoAll(info.level, info.Z)).state = Leave;
               }
            }
            info.state = ary0.state;
         }
      }

      LoadBalancer<TGrid> Balancer(*AMR::m_refGrid);

      // Refinement/compression of blocks
      /*************************************************/
      int r = 0;
      int c = 0;

      std::vector<int> m_com;
      std::vector<int> m_ref;
      std::vector<long long> n_com;
      std::vector<long long> n_ref;


      for (auto &info : I)
      {
         if (info.state == Refine)
         {
            m_ref.push_back(info.level);
            n_ref.push_back(info.Z);
         }
         else if (info.state == Compress)
         {
            m_com.push_back(info.level);
            n_com.push_back(info.Z);
         }
      }

      #pragma omp parallel
      {
         #pragma omp for
         for (size_t i = 0; i < m_ref.size(); i++)
         {
            MeshAdaptation_basic<TGrid,otherTGRID>::refine_1(m_ref[i], n_ref[i]);
            #pragma omp atomic
            r++;
         }
         #pragma omp for
         for (size_t i = 0; i < m_ref.size(); i++)
         {
            AMR::refine_2(m_ref[i], n_ref[i]);
         }
      }

      Balancer.PrepareCompression();

      #pragma omp parallel for
      for (size_t i = 0; i < m_com.size(); i++)
      {
         AMR::compress(m_com[i], n_com[i]);
         #pragma omp atomic
         c++;
      }

      AMR::m_refGrid->FillPos();
      Balancer.Balance_Diffusion();

      #if 0
      int temp[2] = {r, c};
      int result[2];
      MPI_Allreduce(&temp, &result, 2, MPI_INT, MPI_SUM, AMR::m_refGrid->getWorldComm());
      int rank;
      MPI_Comm_rank(AMR::m_refGrid->getWorldComm(), &rank);
      if (rank == 0)
      {
         std::cout << "==============================================================\n";
         std::cout << " refined:" << result[0] << "   compressed:" << result[1] << std::endl;
         std::cout << "==============================================================\n";
      }
      #endif

      if ( result[0] > 0 || result[1] > 0 || Balancer.movedBlocks)
      {
         AMR::m_refGrid->UpdateFluxCorrection = true;
         AMR::m_refGrid->UpdateGroups = true;
         AMR::m_refGrid->UpdateBlockInfoAll_States(false);
         auto it = AMR::m_refGrid->SynchronizerMPIs.begin();
         while (it != AMR::m_refGrid->SynchronizerMPIs.end())
         {
            (*it->second)._Setup(&(AMR::m_refGrid->getBlocksInfo())[0], (AMR::m_refGrid->getBlocksInfo()).size(), timestamp,true);
            it++;
         }
      }
      else
      {
         //AMR::m_refGrid->UpdateFluxCorrection = flag;
         //flag                            = false;
      }
      std::cout << std::flush;
   }
////////////////////////////////////////////////////////////////////////////////////////////////////
 protected:

   virtual void ValidStates() override
   {
      const std::array<int, 3> blocksPerDim = AMR::m_refGrid->getMaxBlocks();
      const int levelMin              = 0;
      const int levelMax              = AMR::m_refGrid->getlevelMax();
      const bool xperiodic            = AMR::labs[0].is_xperiodic();
      const bool yperiodic            = AMR::labs[0].is_yperiodic();
      const bool zperiodic            = AMR::labs[0].is_zperiodic();

      std::vector<BlockInfo> &I = AMR::m_refGrid->getBlocksInfo();

      for (size_t j = 0; j < I.size(); j++)
      {
         BlockInfo &info = I[j];
         if (info.state != Leave)
         {
            info.changed2 = true;
            (AMR::m_refGrid->getBlockInfoAll(info.level, info.Z)).changed2 = info.changed2;
         }
      }

      // 1.Change states of blocks next to finer resolution blocks
      // 2.Change states of blocks next to same resolution blocks
      // 3.Compress a block only if all blocks with the same parent need compression
      for (int m = levelMax - 1; m >= levelMin; m--)
      {
         // 1.
         for (size_t j = 0; j < I.size(); j++)
         {
            BlockInfo &info = I[j];
            if (info.level == m && info.state != Refine && info.level != levelMax - 1)
            {
               const int TwoPower = 1 << info.level;
               const bool xskin = info.index[0] == 0 || info.index[0] == blocksPerDim[0] * TwoPower - 1;
               const bool yskin = info.index[1] == 0 || info.index[1] == blocksPerDim[1] * TwoPower - 1;
               const bool zskin = info.index[2] == 0 || info.index[2] == blocksPerDim[2] * TwoPower - 1;
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

                  BlockInfo &infoNei = AMR::m_refGrid->getBlockInfoAll(info.level, info.Znei_(code[0], code[1], code[2]));
                  if (AMR::m_refGrid->Tree(infoNei).CheckFiner())
                  {
                     if (info.state == Compress)
                     {
                        info.state = Leave;
                        (AMR::m_refGrid->getBlockInfoAll(info.level, info.Z)).state = Leave;
                     }
                     // if (info.level == levelMax - 1) break;

                     int Bstep = 1;                                                    // face
                     if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 2)) Bstep = 3; // edge
                     else if ((abs(code[0]) + abs(code[1]) + abs(code[2]) == 3)) Bstep = 4; // corner

                     for (int B = 0; B <= 3; B += Bstep) // loop over blocks that make up face/edge/corner (respectively 4,2 or 1 blocks)
                     {
                        const int aux = (abs(code[0]) == 1) ? (B % 2) : (B / 2);
                        const int iNei = 2 * info.index[0] + max(code[0], 0) + code[0] + (B % 2) * max(0, 1 - abs(code[0]));
                        const int jNei = 2 * info.index[1] + max(code[1], 0) + code[1] + aux * max(0, 1 - abs(code[1]));
                        const int kNei = 2 * info.index[2] + max(code[2], 0) + code[2] + (B / 2) * max(0, 1 - abs(code[2]));
                        const long long zzz = AMR::m_refGrid->getZforward(m + 1, iNei, jNei, kNei);
                        BlockInfo &FinerNei = AMR::m_refGrid->getBlockInfoAll(m + 1, zzz);
                        State NeiState      = FinerNei.state;
                        if (NeiState == Refine)
                        {
                           info.state = Refine;
                           (AMR::m_refGrid->getBlockInfoAll(info.level, info.Z)).state = Refine;
                           info.changed2 = true;
                           (AMR::m_refGrid->getBlockInfoAll(info.level, info.Z)).changed2 = true;
                           break;
                        }
                     }
                  }
               }
            }
         }

         AMR::m_refGrid->UpdateBoundary();

         if (m == levelMin) break;
         // 2.
         for (size_t j = 0; j < I.size(); j++)
         {
            BlockInfo &info = I[j];
            if (info.level == m && info.state == Compress)
            {
               const int aux = 1 << info.level;
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

                  BlockInfo &infoNei = AMR::m_refGrid->getBlockInfoAll(info.level, info.Znei_(code[0], code[1], code[2]));
                  if (AMR::m_refGrid->Tree(infoNei).Exists()&& infoNei.state == Refine)
                  {
                     info.state = Leave;
                     (AMR::m_refGrid->getBlockInfoAll(info.level, info.Z)).state = Leave;
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
         const int m = info.level;
         bool found = false;
         for (int i = 2 * (info.index[0] / 2); i <= 2 * (info.index[0] / 2) + 1; i++)
         for (int j = 2 * (info.index[1] / 2); j <= 2 * (info.index[1] / 2) + 1; j++)
         for (int k = 2 * (info.index[2] / 2); k <= 2 * (info.index[2] / 2) + 1; k++)
         {
            const long long n = AMR::m_refGrid->getZforward(m, i, j, k);
            BlockInfo &infoNei = AMR::m_refGrid->getBlockInfoAll(m, n);
            if (AMR::m_refGrid->Tree(infoNei).Exists() == false|| infoNei.state != Compress)
            {
                found = true;
                if (info.state == Compress)
                {
                    info.state                                             = Leave;
                    (AMR::m_refGrid->getBlockInfoAll(info.level, info.Z)).state = Leave;
                }
                break;
            }
        }
        if (found)      
        for (int i = 2 * (info.index[0] / 2); i <= 2 * (info.index[0] / 2) + 1; i++)
        for (int j = 2 * (info.index[1] / 2); j <= 2 * (info.index[1] / 2) + 1; j++)
        for (int k = 2 * (info.index[2] / 2); k <= 2 * (info.index[2] / 2) + 1; k++)
        {
            const long long n = AMR::m_refGrid->getZforward(m, i, j, k);
            BlockInfo &infoNei = AMR::m_refGrid->getBlockInfoAll(m, n);
            if (AMR::m_refGrid->Tree(infoNei).Exists() && infoNei.state == Compress) infoNei.state = Leave;
        }
      }

      // 4.
      for (size_t jjj = 0; jjj < I.size(); jjj++)
      {
        BlockInfo &info = I[jjj];
        if (info.state == Compress)
        {
           const int m = info.level;
           bool first = true;
           for (int i = 2 * (info.index[0] / 2); i <= 2 * (info.index[0] / 2) + 1; i++)
           for (int j = 2 * (info.index[1] / 2); j <= 2 * (info.index[1] / 2) + 1; j++)
           for (int k = 2 * (info.index[2] / 2); k <= 2 * (info.index[2] / 2) + 1; k++)
           {
               const long long n = AMR::m_refGrid->getZforward(m, i, j, k);
               BlockInfo &infoNei = AMR::m_refGrid->getBlockInfoAll(m, n);                   
               if (!first) infoNei.state = Leave;
               first = false;
           }
           if (info.index[0] % 2 == 1 || info.index[1] % 2 == 1 || info.index[2] % 2 == 1)
           {
              info.state = Leave;
              (AMR::m_refGrid->getBlockInfoAll(info.level, info.Z)).state = Leave;
           }
        }
      }
   }

   virtual State TagLoadedBlock(TLab &Lab_, BlockInfo & info) override
   {
      return AMR::TagLoadedBlock(Lab_,info);
   }
};

} // namespace cubism
