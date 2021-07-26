#pragma once
#include "AMR_MeshAdaptation.h"
#include "BlockLabMPI.h"
#include "GridMPI.h"
#include "LoadBalancer.h"

namespace cubism
{

template <typename TGrid, typename TLab, typename otherTGRID = TGrid>
class MeshAdaptationMPI : public MeshAdaptation<TGrid,TLab>
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
   bool LabsPrepared;
   bool CallValidStates;
   LoadBalancer<TGrid> *Balancer;
   using AMR = MeshAdaptation<TGrid,TLab>;

 public:
   MeshAdaptationMPI(TGrid &grid, double Rtol, double Ctol): MeshAdaptation<TGrid,TLab>(grid,Rtol,Ctol)
   {
      bool tensorial = false;
      const int Gx = 1;
      const int Gy = 1;
      const int Gz = DIMENSION == 3? 1:0;
      StencilInfo stencil(-Gx, -Gy, -Gz, Gx + 1, Gy + 1, Gz + 1, tensorial, AMR::components);

      AMR::m_refGrid = &grid;
      AMR::s [0] = stencil.sx;
      AMR::e [0] = stencil.ex;
      AMR::s [1] = stencil.sy;
      AMR::e [1] = stencil.ey;
      AMR::s [2] = stencil.sz;
      AMR::e [2] = stencil.ez;
      AMR::Is[0] = stencil.sx;
      AMR::Ie[0] = stencil.ez;
      AMR::Is[1] = stencil.sy;
      AMR::Ie[1] = stencil.ey;
      AMR::Is[2] = stencil.sz;
      AMR::Ie[2] = stencil.ez;
      AMR::istensorial = stencil.tensorial;

      Balancer = new LoadBalancer<TGrid>(*AMR::m_refGrid);
      timestamp = 0;
      flag      = true;
      auto blockperDim     = AMR::m_refGrid->getMaxBlocks();
      StencilInfo Cstencil = stencil;
      Synch                = new SynchronizerMPIType(stencil, Cstencil, AMR::m_refGrid->getlevelMax(), TGrid::Block::sizeX,TGrid::Block::sizeY, TGrid::Block::sizeZ, blockperDim[0], blockperDim[1], blockperDim[2], &grid);
      Synch->_Setup(&(AMR::m_refGrid->getBlocksInfo())[0], (AMR::m_refGrid->getBlocksInfo()).size(), timestamp, true);
   }

   virtual ~MeshAdaptationMPI() { delete Synch; delete Balancer;}

   virtual void Tag(double t = 0) override
   {
      AMR::time = t;

      Synch->sync(sizeof(typename Block::element_type) / sizeof(Real),sizeof(Real) > 4 ? MPI_DOUBLE : MPI_FLOAT, timestamp);
      timestamp = (timestamp + 1) % 32768;

      const int nthreads = omp_get_max_threads();
      AMR::labs = new TLab[nthreads];
      for (int i = 0; i < nthreads; i++) AMR::labs[i].prepare(*AMR::m_refGrid, *Synch);

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
         MPI_Iallreduce(MPI_IN_PLACE, &tmp, 1, MPI_INT, MPI_SUM, AMR::m_refGrid->getWorldComm(),&Reduction_req);
      }

      MPI_Wait(&Reduction_req,MPI_STATUS_IGNORE);
      CallValidStates     = (tmp > 0);

      AMR::m_refGrid->boundary = halo;

      if (CallValidStates) AMR::ValidStates();
   }

   virtual void Adapt(double t = 0, bool verbosity = false, bool basic = false) override
   {
      AMR::basic_refinement = basic;
      if (LabsPrepared == false)
      {
         Synch->sync(sizeof(typename Block::element_type) / sizeof(Real),sizeof(Real) > 4 ? MPI_DOUBLE : MPI_FLOAT, timestamp);
         timestamp = (timestamp + 1) % 32768;
         const int nthreads = omp_get_max_threads();
         AMR::labs = new TLab[nthreads];
         for (int i = 0; i < nthreads; i++) AMR::labs[i].prepare(*AMR::m_refGrid, *Synch);
         //TODO: the line below means there's no computation & communication overlap here
         AMR::m_refGrid->boundary  = Synch->avail_halo();
         AMR::m_refGrid->UpdateBoundary();
      }

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
         else if (info.state == Compress
            && info.index[0]%2 == 0
            && info.index[1]%2 == 0
            && info.index[2]%2 == 0)
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

      Balancer->PrepareCompression();

      #pragma omp parallel for
      for (size_t i = 0; i < m_com.size(); i++)
      {
         AMR::compress(m_com[i], n_com[i]);
         #pragma omp atomic
         c++;
      }

      int temp[2] = {r, c};
      int result[2];
      MPI_Allreduce(&temp, &result, 2, MPI_INT, MPI_SUM, AMR::m_refGrid->getWorldComm());
      if (verbosity)
      {
         std::cout << "==============================================================\n";
         std::cout << " refined:" << result[0] << "   compressed:" << result[1] << std::endl;
         std::cout << "==============================================================\n";
         std::cout << std::flush;
      }
      AMR::m_refGrid->FillPos();     
      Balancer->Balance_Diffusion();

      delete[] AMR::labs;

      if ( result[0] > 0 || result[1] > 0 || Balancer->movedBlocks)
      {
         AMR::m_refGrid->UpdateFluxCorrection = true;
         AMR::m_refGrid->UpdateGroups = true;

         AMR::m_refGrid->UpdateBlockInfoAll_States(false);

         Synch->_Setup(&(AMR::m_refGrid->getBlocksInfo())[0], (AMR::m_refGrid->getBlocksInfo()).size(), timestamp, true);

         auto it = AMR::m_refGrid->SynchronizerMPIs.begin();
         while (it != AMR::m_refGrid->SynchronizerMPIs.end())
         {
            (*it->second)._Setup(&(AMR::m_refGrid->getBlocksInfo())[0], (AMR::m_refGrid->getBlocksInfo()).size(), timestamp);
            it++;
         }
      }
      LabsPrepared = false;
   }

 protected:
   void TagBlocksVector(std::vector<BlockInfo *> & I, bool & Reduction, MPI_Request & Reduction_req, int & tmp)
   {
      const int levelMax = AMR::m_refGrid->getlevelMax();
      #pragma omp parallel
      {
         const int tid = omp_get_thread_num();
         #pragma omp for schedule(dynamic, 1)
         for (size_t i = 0; i < I.size(); i++)
         {
            AMR::labs[tid].load(*I[i], AMR::time);

            BlockInfo &info = AMR::m_refGrid->getBlockInfoAll(I[i]->level, I[i]->Z);

            I[i]->state     = TagLoadedBlock(AMR::labs[tid],info);

            const bool maxLevel = (I[i]->state == Refine  ) && (I[i]->level == levelMax - 1);
            const bool minLevel = (I[i]->state == Compress) && (I[i]->level == 0);

            if (maxLevel || minLevel) I[i]->state = Leave;

            info.state = I[i]->state;
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
   }
   virtual State TagLoadedBlock(TLab &Lab_, BlockInfo & info) override
   {
      return AMR::TagLoadedBlock(Lab_,info);
   }
};

} // namespace cubism
