#pragma once
#include "Matrix3D.h"
#include "BlockInfo.h"
#include "Grid.h"
#include "BlockLab.h"
#include "GridMPI.h"
#include "BlockLabMPI.h"
#include "LoadBalancer.h"

#include <omp.h>
#include <cstring>
#include <string>
#include <algorithm>

namespace cubism
{


#define WENOWAVELET 5



template<typename TGrid,  typename TLab>
class MeshAdaptation
{

public:
    typedef typename TGrid::Block BlockType;   
    typedef typename TGrid::Block::ElementType ElementType;
    typedef SynchronizerMPI_AMR<Real> SynchronizerMPIType;
    typedef typename TGrid::BlockType Block;

protected:
    TGrid * m_refGrid;
    int s[3];  
    int e[3];
    bool istensorial;
    int Is[3];  
    int Ie[3];
    std::vector<int> components;
    double tolerance_for_refinement;
    double tolerance_for_compression;
    TLab * labs;
    double time;

    SynchronizerMPI_AMR<Real> * Synch;
    int timestamp;
    bool flag;
public:

    MeshAdaptation(TGrid & grid, double Rtol,double Ctol) 
    {
        bool tensorial = true;
        
        const int Gx = (WENOWAVELET == 3) ? 1:2;
        const int Gy = (WENOWAVELET == 3) ? 1:2;
        const int Gz = (WENOWAVELET == 3) ? 1:2;
        
        components.push_back(0); 
        components.push_back(1); 
        components.push_back(2); 
        components.push_back(3); 
        components.push_back(4); 
        components.push_back(5); 
        components.push_back(6); 
        components.push_back(7); 
    
        StencilInfo stencil(-Gx,-Gy,-Gz,Gx+1,Gy+1,Gz+1,tensorial,components);

        m_refGrid = &grid;
        
        s[0] = stencil.sx; e[0] = stencil.ex;
        s[1] = stencil.sy; e[1] = stencil.ey;
        s[2] = stencil.sz; e[2] = stencil.ez;
        istensorial = stencil.tensorial;
        
        Is[0] = stencil.sx; Ie[0] = stencil.ez;
        Is[1] = stencil.sy; Ie[1] = stencil.ey;
        Is[2] = stencil.sz; Ie[2] = stencil.ez;
    
        tolerance_for_refinement = Rtol;
        tolerance_for_compression = Ctol; 

        timestamp = 0;
        flag = true;
        
        auto blockperDim = m_refGrid->getMaxBlocks();          
        bool per [3] = {m_refGrid->xperiodic,m_refGrid->yperiodic,m_refGrid->zperiodic}; 
        StencilInfo Cstencil = stencil; 
        Synch = new SynchronizerMPIType(stencil, Cstencil, MPI_COMM_WORLD, per, 
                                        m_refGrid->getlevelMax(),
                                        TGrid::Block::sizeX,
                                        TGrid::Block::sizeY,
                                        TGrid::Block::sizeZ,
                                        blockperDim[0],
                                        blockperDim[1],
                                        blockperDim[2]);      

        Synch->_Setup(&(m_refGrid->getBlocksInfo())[0],
                       (m_refGrid->getBlocksInfo()).size(),
                        m_refGrid->getBlockInfoAll(),timestamp);
    }

    virtual
    ~MeshAdaptation()
    {
        delete Synch;
    }

    void AdaptTheMesh(double t = 0)
    {
        time = t;
        /*------------->*/Clock.start(10,"sync"); 
        Synch->sync(sizeof(typename Block::element_type)/sizeof(Real), sizeof(Real)>4 ? MPI_DOUBLE : MPI_FLOAT, timestamp);
        /*------------->*/Clock.finish(10);
        
        timestamp = (timestamp + 1) % 32768;

        vector<BlockInfo *> avail0, avail1;

        const int nthreads = omp_get_max_threads();
    
        labs = new TLab[nthreads];
        for (int i=0; i<nthreads; i++)
          labs[i].prepare(*m_refGrid, *Synch);
  
        bool CallValidStates = false;

        /*------------->*/Clock.start(1,"MeshAdaptation: inner blocks tagging");  
        avail0 = Synch->avail_inner(); 
        const int Ninner = avail0.size();
        
        #pragma omp parallel num_threads(nthreads)
        {
            int tid = omp_get_thread_num();
            TLab& mylab = labs[tid];

            #pragma omp for schedule(dynamic,1)
            for(int i=0; i<Ninner; i++)
            {
                BlockInfo & ary0 = * avail0[i];
                mylab.load(ary0, t);
                BlockInfo & info = m_refGrid->getBlockInfoAll(ary0.level,ary0.Z);
                ary0.state = TagLoadedBlock(labs[tid]);
                info.state = ary0.state;
                if (info.state != Leave) CallValidStates = true;  
            }
        }
        /*------------->*/Clock.finish(1);




        /*------------->*/Clock.start(2,"MeshAdaptation: waiting for inner blocks");
        avail1 = Synch->avail_halo();
        /*------------->*/Clock.finish(2);




        /*------------->*/Clock.start(3,"MeshAdaptation: outer block tagging");
        const int Nhalo = avail1.size();
        #pragma omp parallel num_threads(nthreads)
        {
            int tid = omp_get_thread_num();
            TLab& mylab = labs[tid];
    
            #pragma omp for schedule(dynamic,1)
            for(int i=0; i<Nhalo; i++)
            {
                BlockInfo & ary1 = * avail1[i];
                mylab.load(ary1, t);
                BlockInfo & info = m_refGrid->getBlockInfoAll(ary1.level,ary1.Z);
                ary1.state = TagLoadedBlock(labs[tid]);
                info.state = ary1.state;                          
                if (info.state != Leave) CallValidStates = true;
            }
        }
        /*------------->*/Clock.finish(3);




        /*------------->*/Clock.start(4,"MeshAdaptation: ValidStates");
        int tmp = CallValidStates ? 1:0;
        MPI_Allreduce(MPI_IN_PLACE,&tmp,1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        CallValidStates = (tmp == 1);
 
        if (CallValidStates)
         ValidStates();
        /*------------->*/Clock.finish(4);

        /*------------->*/Clock.start(5,"MeshAdaptation: PrepareCompression");
        LoadBalancer <TGrid> Balancer (*m_refGrid);

        Balancer.PrepareCompression();
        /*------------->*/Clock.finish(5);

        //Refinement/compression of blocks
        /*************************************************/
        /*------------->*/Clock.start(6,"MeshAdaptation: refinement and compression");
        int r=0;
        int c=0;
      
        std::vector <int> mn_com;
        std::vector <int> mn_ref;
    
        std::vector <BlockInfo> & I = m_refGrid->getBlocksInfo();   
    
        for ( auto & i: I)
        {
          BlockInfo & info =  m_refGrid->getBlockInfoAll(i.level,i.Z);                  
          if (info.state == Refine)
          {
            mn_ref.push_back(info.level);
            mn_ref.push_back(info.Z);
          }
          else if (info.state == Compress)
          {
            mn_com.push_back(info.level);
            mn_com.push_back(info.Z);
          }
        }
              
        #pragma omp parallel
        {
            #pragma omp for schedule(runtime)
            for (size_t i=0; i<mn_ref.size()/2; i++)
            {
              int m = mn_ref[2*i];
              int n = mn_ref[2*i+1];           
              refine_1(m,n);
              #pragma omp atomic
                r++;
            }        
            #pragma omp for schedule(runtime)
            for (size_t i=0; i<mn_ref.size()/2; i++)
            {
              int m = mn_ref[2*i];
              int n = mn_ref[2*i+1]; 
              refine_2(m,n);
            }   
    
            #pragma omp for schedule(runtime)
            for (size_t i=0; i<mn_com.size()/2; i++)
            {
              int m = mn_com[2*i];
              int n = mn_com[2*i+1];  
              compress(m,n);
              #pragma omp atomic
               c++;
            }
        }      
       /*************************************************/

        int temp[2] = {r,c};
        int result[2];
        MPI_Allreduce(&temp, &result, 2, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        if (rank ==0)
        {
            std::cout <<"==============================================================\n";
            std::cout << " refined:" << result[0] << "   compressed:"<< result[1] << std::endl;
            std::cout <<"==============================================================\n";
        }
        /*------------->*/Clock.finish(6);

        


        /*------------->*/Clock.start(7,"MeshAdaptation : UpdateBlockInfoAll_States");
        m_refGrid->FillPos();
        m_refGrid->UpdateBlockInfoAll_States(false);
        /*------------->*/Clock.finish(7);
   

        /*------------->*/Clock.start(8,"MeshAdaptation : Balance_Diffusion");
        Balancer.Balance_Diffusion();
        /*------------->*/Clock.finish(8);
   

        /*------------->*/Clock.start(7,"MeshAdaptation : UpdateBlockInfoAll_States");
        m_refGrid->FillPos();
        /*------------->*/Clock.finish(7);
        
        delete [] labs;

        /*------------->*/Clock.start(9,"MeshAdaptation : Setup");
        if (result[0] !=0 || result[1] != 0 || Balancer.movedBlocks)
        {
            m_refGrid->UpdateFluxCorrection = true;
            

            /*------------->*/Clock.start(7,"MeshAdaptation : UpdateBlockInfoAll_States");
            m_refGrid->UpdateBlockInfoAll_States(false);
            /*------------->*/Clock.finish(7);
            

            Synch->_Setup(&(m_refGrid->getBlocksInfo())[0],
                           (m_refGrid->getBlocksInfo()).size(),
                            m_refGrid->getBlockInfoAll(),timestamp);
            
            //typename std::map<StencilInfo,SynchronizerMPIType*>::iterator 
            auto it = m_refGrid->SynchronizerMPIs.begin();
            while (it != m_refGrid->SynchronizerMPIs.end())
            { 
                (*it->second)._Setup(&(m_refGrid->getBlocksInfo())[0],
                                      (m_refGrid->getBlocksInfo()).size(),
                                       m_refGrid->getBlockInfoAll(),timestamp);
                it++;
            }
        }
        else
        {
            m_refGrid->UpdateFluxCorrection = flag;
            flag = false;
        }
        /*------------->*/Clock.finish(9);
    }


protected:
  
    void refine_1(int level, int Z) 
    {
        int tid = omp_get_thread_num();
        
        BlockInfo & parent =  m_refGrid->getBlockInfoAll(level,Z);
        labs[tid].load(parent,time,true);   
    
        int p[3] = {parent.index[0],parent.index[1],parent.index[2]};
    
        assert(parent.ptrBlock != NULL);
        assert(level <= m_refGrid->getlevelMax()-1);
        
        BlockType * Blocks [8];
          
        for (int k=0; k<2; k++ )
        for (int j=0; j<2; j++ )
        for (int i=0; i<2; i++ )
        {      
            int nc = m_refGrid->getZforward(level+1,2*p[0]+i,2*p[1]+j,2*p[2]+k);      
            BlockInfo & Child = m_refGrid->getBlockInfoAll(level+1,nc);
            #pragma omp critical
            {
              m_refGrid->_alloc(level+1,nc);
            }
            Blocks [k*4 + j*2 + i] = (BlockType*) Child.ptrBlock;
        }
        RefineBlocks(Blocks,parent);  
    }

    void refine_2(int level, int Z) 
    {
        #pragma omp critical
        {
            m_refGrid->_dealloc(level,Z);
            m_refGrid->getBlockInfoAll(level,Z).state = Leave;
        }

        BlockInfo & parent =  m_refGrid->getBlockInfoAll(level,Z);
        parent.TreePos = CheckFiner;

        int p[3] = {parent.index[0],parent.index[1],parent.index[2]};
        for (int k=0; k<2; k++ )
        for (int j=0; j<2; j++ )
        for (int i=0; i<2; i++ )
        {      
            int nc = m_refGrid->getZforward(level+1,2*p[0]+i,2*p[1]+j,2*p[2]+k);
            BlockInfo & Child = m_refGrid->getBlockInfoAll(level+1,nc); 
            Child.TreePos = Exists;
        }
    }

    void compress(int level, int Z)
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);      
        assert (level > 0); 
    
        BlockInfo & info = m_refGrid->getBlockInfoAll(level,Z);
    
        BlockType * Blocks [8];
        for (int K=0; K<2; K++ )
        for (int J=0; J<2; J++ )
        for (int I=0; I<2; I++ )
        {
          int blk = K*4+J*2+I;
          int n   = m_refGrid->getZforward(level,info.index[0]+I,info.index[1]+J,info.index[2]+K); 
          Blocks[blk] = (BlockType *)(m_refGrid->getBlockInfoAll(level,n)).ptrBlock;
        }
   
        const int nx = BlockType::sizeX;
        const int ny = BlockType::sizeY;
        const int nz = BlockType::sizeZ;
        int offsetX [2] = {0,nx/2};
        int offsetY [2] = {0,ny/2};
        int offsetZ [2] = {0,nz/2};
        for (int K=0; K<2; K++ )
        for (int J=0; J<2; J++ )
        for (int I=0; I<2; I++ )
        {
            BlockType & b = *Blocks[K*4+J*2+I];
            for (int k=0; k<nz; k+=2 )
            for (int j=0; j<ny; j+=2 )
            for (int i=0; i<nx; i+=2 )
            {
                ElementType average =0.125*(b(i,j,k  )+b(i+1,j,k  )+b(i,j+1,k  )+b(i+1,j+1,k  )
                                           +b(i,j,k+1)+b(i+1,j,k+1)+b(i,j+1,k+1)+b(i+1,j+1,k+1));
                (*Blocks[0])(i/2+offsetX[I],j/2+offsetY[J]  ,k/2+offsetZ[K]) = average;
            }
        } 
        
        int np = m_refGrid->getZforward(level-1,info.index[0]/2,info.index[1]/2,info.index[2]/2);           
        BlockInfo & parent = m_refGrid->getBlockInfoAll(level-1,np);
        parent.myrank      = m_refGrid->rank();
        parent.ptrBlock    = info.ptrBlock;
        parent.TreePos     = Exists; 
        parent.h_gridpoint = parent.h;
        parent.changed     = true;

        #pragma omp critical
        {
          for (int K=0; K<2; K++ )
          for (int J=0; J<2; J++ )
          for (int I=0; I<2; I++ )
          {
            int n = m_refGrid->getZforward(level,info.index[0]+I,info.index[1]+J,info.index[2]+K); 
      
            if (I + J + K == 0 )
            {
                m_refGrid->FindBlockInfo(level,n,level - 1,np);
            } 
            else
            {
                m_refGrid->_dealloc(level,n);
            }
            m_refGrid->getBlockInfoAll(level,n).TreePos = CheckCoarser;
            m_refGrid->getBlockInfoAll(level,n).state = Leave;
          }
        }
    }


    void ValidStates()
    {
        static std::array <int,3> blocksPerDim = m_refGrid->getMaxBlocks();
        static const int levelMin = 0;
        static const int levelMax = m_refGrid->getlevelMax();
        static const bool xperiodic = labs[0].is_xperiodic();
        static const bool yperiodic = labs[0].is_yperiodic();
        static const bool zperiodic = labs[0].is_zperiodic();


        std::vector <BlockInfo> & I = m_refGrid->getBlocksInfo();
        for ( auto & b: I)
        {
            BlockInfo & info = m_refGrid->getBlockInfoAll(b.level,b.Z);
            info.changed = true;
            b.changed = true;
            if (info.state==Refine   && info.level ==levelMax-1)
            {
                info.state=Leave;
            } 
            if (info.state==Compress && info.level ==levelMin  )
            {
                info.state=Leave;
            }
        }

        
        /*------------->*/Clock.start(0,"MeshAdaptation: UpdateBoundary");
        m_refGrid->FillPos(false);
        m_refGrid->UpdateBoundary(true);
        /*------------->*/Clock.finish(0);


        for (int m=levelMax-1; m>=levelMin; m--)
        {
            //1.Change states of blocks next to finer resolution blocks
            //2.Change states of blocks next to same resolution blocks
            //3.Compress a block only if all blocks with the same parent need compression

      
            //1.
            for ( auto & b: I) if (b.level == m)
            {
                BlockInfo & info =  m_refGrid->getBlockInfoAll(m,b.Z);

                assert(b.TreePos == Exists);

                int TwoPower = 1<<info.level;
                const bool xskin = info.index[0]==0 || info.index[0]==blocksPerDim[0]*TwoPower-1;
                const bool yskin = info.index[1]==0 || info.index[1]==blocksPerDim[1]*TwoPower-1;
                const bool zskin = info.index[2]==0 || info.index[2]==blocksPerDim[2]*TwoPower-1;
                const int  xskip  = info.index[0]==0 ? -1 : 1;
                const int  yskip  = info.index[1]==0 ? -1 : 1;
                const int  zskip  = info.index[2]==0 ? -1 : 1;

                for(int icode=0; icode<27; icode++)
                {
                    if (icode == 1*1 + 3*1 + 9*1) continue;
                    const int code[3] = { icode%3-1, (icode/3)%3-1, (icode/9)%3-1};

                    if (!xperiodic && code[0] == xskip && xskin) continue;
                    if (!yperiodic && code[1] == yskip && yskin) continue;
                    if (!zperiodic && code[2] == zskip && zskin) continue;
                    
                    BlockInfo & infoNei = m_refGrid->getBlockInfoAll(info.level,info.Znei_(code[0],code[1],code[2]) );

                    if (infoNei.TreePos == CheckFiner && info.state!=Refine)
                    {
                        info.state=Leave;
                        info.changed = true;
                        b.changed = true;

                        int Bstep = 1; //face
                        if      ((abs(code[0])+abs(code[1])+abs(code[2])==2 )) Bstep = 3; //edge
                        else if ((abs(code[0])+abs(code[1])+abs(code[2])==3 )) Bstep = 4; //corner

                        for (int B = 0 ; B <= 3 ; B += Bstep) //loop over blocks that make up face/edge/corner (respectively 4,2 or 1 blocks)
                        {
                            const int aux = (abs(code[0])==1) ? (B%2) : (B/2) ;

                            int iNei = 2*info.index[0] + max(code[0],0) +code[0]  + (B%2)*max(0, 1 - abs(code[0]));
                            int jNei = 2*info.index[1] + max(code[1],0) +code[1]  +  aux *max(0, 1 - abs(code[1]));
                            int kNei = 2*info.index[2] + max(code[2],0) +code[2]  + (B/2)*max(0, 1 - abs(code[2]));
                            int zzz = m_refGrid->getZforward(m+1,iNei,jNei,kNei);
                            BlockInfo & FinerNei = m_refGrid->getBlockInfoAll(m+1,zzz);
                            State NeiState = FinerNei.state;
                            if (NeiState == Refine)
                            {
                                info.state=Refine;
                                info.changed = true;
                                b.state = Refine;
                                b.changed = true;
                                break;
                            }
                        }
                    }
                }
            }
            
            /*------------->*/Clock.start(0,"MeshAdaptation: UpdateBoundary");
            m_refGrid->FillPos(false);
            m_refGrid->UpdateBoundary(false);
            /*------------->*/Clock.finish(0);            

            //2.
            for ( auto & b: I) if (b.level == m)
            {
                BlockInfo & info =  m_refGrid->getBlockInfoAll(m,b.Z);
                
                if (info.state!=Compress) continue;

                int aux = 1<<info.level;
                const bool xskin = info.index[0]==0 || info.index[0]==blocksPerDim[0]*aux-1;
                const bool yskin = info.index[1]==0 || info.index[1]==blocksPerDim[1]*aux-1;
                const bool zskin = info.index[2]==0 || info.index[2]==blocksPerDim[2]*aux-1;
                const int xskip  = info.index[0]==0 ? -1 : 1;
                const int yskip  = info.index[1]==0 ? -1 : 1;
                const int zskip  = info.index[2]==0 ? -1 : 1;

                for(int icode=0; icode<27; icode++)
                {
                    if (icode == 1*1 + 3*1 + 9*1) continue;
                    const int code[3] = { icode%3-1, (icode/3)%3-1, (icode/9)%3-1};
                    if (!xperiodic && code[0] == xskip && xskin) continue;
                    if (!yperiodic && code[1] == yskip && yskin) continue;
                    if (!zperiodic && code[2] == zskip && zskin) continue;

                    BlockInfo & infoNei = m_refGrid->getBlockInfoAll(info.level,info.Znei_(code[0],code[1],code[2]) );

                    if (infoNei.TreePos == Exists && infoNei.state==Refine)
                    {
                        info.state=Leave;
                        info.changed = true;
                        b.changed = true;
                    }
                }
            }

            /*------------->*/Clock.start(0,"MeshAdaptation: UpdateBoundary");
            m_refGrid->FillPos(false);
            m_refGrid->UpdateBoundary(false);
            /*------------->*/Clock.finish(0);
         
            //3.
            for ( auto & b: I) if (b.level == m)
            {
              BlockInfo & info =  m_refGrid->getBlockInfoAll(m,b.Z);
              if (info.state==Compress)
                for (int i= 2*(info.index[0]/2); i <= 2*(info.index[0]/2)+1; i++)
                for (int j= 2*(info.index[1]/2); j <= 2*(info.index[1]/2)+1; j++)
                for (int k= 2*(info.index[2]/2); k <= 2*(info.index[2]/2)+1; k++)
                {
                    int n = m_refGrid->getZforward(m,i,j,k);
                    BlockInfo & infoNei = m_refGrid->getBlockInfoAll(m,n);
                    if (infoNei.TreePos != Exists || infoNei.state != Compress )
                    {
                      info.state = Leave;
                      info.changed = true;
                      b.changed = true;
                      break;
                    }
                }
            }

            //4.
            for ( auto & b: I) if (b.level == m)
            {
                if (b.state == Compress && (b.index[0]%2==1 || b.index[1]%2==1 || b.index[2]%2==1)  )
                {
                    BlockInfo & info =  m_refGrid->getBlockInfoAll(m,b.Z);
                    info.state = Leave;
                    b.state = Leave;                    
                }
            }
        }//m
    }



    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Virtual functions that can be overwritten by user
    ////////////////////////////////////////////////////////////////////////////////////////////////
    virtual 
    void RefineBlocks(BlockType * B[8], BlockInfo parent) 
    {
        int tid = omp_get_thread_num();

        const int nx = BlockType::sizeX;
        const int ny = BlockType::sizeY;
        const int nz = BlockType::sizeZ;

        int offsetX [2] = {0,nx/2};
        int offsetY [2] = {0,ny/2};
        int offsetZ [2] = {0,nz/2};
    
        TLab & Lab = labs[tid];

        for (int K=0; K<2; K++ )
        for (int J=0; J<2; J++ )
        for (int I=0; I<2; I++ )
        {
            BlockType & b = *B[K*4+J*2+I];
            b.clear();

            for (int k=0; k<nz; k+=2 )
            for (int j=0; j<ny; j+=2 )
            for (int i=0; i<nx; i+=2 )
            {
   
            #if 0 //simple linear 
    
                ElementType dudx = 0.5*( Lab(i/2+offsetX[I]+1,j/2+offsetY[J]  ,k/2+offsetZ[K]  )-Lab(i/2+offsetX[I]-1,j/2+offsetY[J]  ,k/2+offsetZ[K]  ));
                ElementType dudy = 0.5*( Lab(i/2+offsetX[I]  ,j/2+offsetY[J]+1,k/2+offsetZ[K]  )-Lab(i/2+offsetX[I]  ,j/2+offsetY[J]-1,k/2+offsetZ[K]  ));
                ElementType dudz = 0.5*( Lab(i/2+offsetX[I]  ,j/2+offsetY[J]  ,k/2+offsetZ[K]+1)-Lab(i/2+offsetX[I]  ,j/2+offsetY[J]  ,k/2+offsetZ[K]-1));
    
                b(i  ,j  ,k  ) = Lab( i   /2+offsetX[I], j   /2+offsetY[J]  ,k    /2+offsetZ[K] )+ (2*( i   %2)-1)*0.25*dudx + (2*( j   %2)-1)*0.25*dudy + (2*(k    %2)-1)*0.25*dudz; 
                b(i+1,j  ,k  ) = Lab((i+1)/2+offsetX[I], j   /2+offsetY[J]  ,k    /2+offsetZ[K] )+ (2*((i+1)%2)-1)*0.25*dudx + (2*( j   %2)-1)*0.25*dudy + (2*(k    %2)-1)*0.25*dudz; 
                b(i  ,j+1,k  ) = Lab( i   /2+offsetX[I],(j+1)/2+offsetY[J]  ,k    /2+offsetZ[K] )+ (2*( i   %2)-1)*0.25*dudx + (2*((j+1)%2)-1)*0.25*dudy + (2*(k    %2)-1)*0.25*dudz; 
                b(i+1,j+1,k  ) = Lab((i+1)/2+offsetX[I],(j+1)/2+offsetY[J]  ,k    /2+offsetZ[K] )+ (2*((i+1)%2)-1)*0.25*dudx + (2*((j+1)%2)-1)*0.25*dudy + (2*(k    %2)-1)*0.25*dudz; 
                b(i  ,j  ,k+1) = Lab( i   /2+offsetX[I], j   /2+offsetY[J]  ,(k+1)/2+offsetZ[K] )+ (2*( i   %2)-1)*0.25*dudx + (2*( j   %2)-1)*0.25*dudy + (2*((k+1)%2)-1)*0.25*dudz; 
                b(i+1,j  ,k+1) = Lab((i+1)/2+offsetX[I], j   /2+offsetY[J]  ,(k+1)/2+offsetZ[K] )+ (2*((i+1)%2)-1)*0.25*dudx + (2*( j   %2)-1)*0.25*dudy + (2*((k+1)%2)-1)*0.25*dudz; 
                b(i  ,j+1,k+1) = Lab( i   /2+offsetX[I],(j+1)/2+offsetY[J]  ,(k+1)/2+offsetZ[K] )+ (2*( i   %2)-1)*0.25*dudx + (2*((j+1)%2)-1)*0.25*dudy + (2*((k+1)%2)-1)*0.25*dudz; 
                b(i+1,j+1,k+1) = Lab((i+1)/2+offsetX[I],(j+1)/2+offsetY[J]  ,(k+1)/2+offsetZ[K] )+ (2*((i+1)%2)-1)*0.25*dudx + (2*((j+1)%2)-1)*0.25*dudy + (2*((k+1)%2)-1)*0.25*dudz; 
          
            #else //WENO

                
                const int Nweno = WENOWAVELET;
                ElementType El[Nweno][Nweno][Nweno];
                for (int i0= -Nweno/2 ; i0<= Nweno/2; i0++)
                for (int i1= -Nweno/2 ; i1<= Nweno/2; i1++)
                for (int i2= -Nweno/2 ; i2<= Nweno/2; i2++)
                    El[i0+Nweno/2][i1+Nweno/2][i2+Nweno/2] = Lab(i/2+offsetX[I] + i0,j/2+offsetY[J]+ i1 ,k/2+offsetZ[K]+ i2);      
                
                ElementType Lines [Nweno][Nweno][2];
                ElementType Planes[Nweno][4];
                ElementType Ref          [8]; 


                #if WENOWAVELET == 3  
                    for (int i2= -Nweno/2 ; i2<= Nweno/2; i2++)
                    for (int i1= -Nweno/2 ; i1<= Nweno/2; i1++)
                        Kernel_1D(El[0][i1+Nweno/2][i2+Nweno/2],
                                  El[1][i1+Nweno/2][i2+Nweno/2],
                                  El[2][i1+Nweno/2][i2+Nweno/2],Lines[i1+Nweno/2][i2+Nweno/2][0],Lines[i1+Nweno/2][i2+Nweno/2][1]);
                    for (int i2= -Nweno/2 ; i2<= Nweno/2; i2++)
                    {
                        Kernel_1D(Lines[0][i2+Nweno/2][0],
                                  Lines[1][i2+Nweno/2][0],
                                  Lines[2][i2+Nweno/2][0],
                                  Planes[i2+Nweno/2][0],
                                  Planes[i2+Nweno/2][1]);
              
                        Kernel_1D(Lines[0][i2+Nweno/2][1],
                                  Lines[1][i2+Nweno/2][1],
                                  Lines[2][i2+Nweno/2][1],
                                  Planes[i2+Nweno/2][2],
                                  Planes[i2+Nweno/2][3]);
                    }
                    Kernel_1D(Planes[0][0],Planes[1][0],Planes[2][0],Ref[0],Ref[1]);
                    Kernel_1D(Planes[0][1],Planes[1][1],Planes[2][1],Ref[2],Ref[3]);
                    Kernel_1D(Planes[0][2],Planes[1][2],Planes[2][2],Ref[4],Ref[5]);
                    Kernel_1D(Planes[0][3],Planes[1][3],Planes[2][3],Ref[6],Ref[7]);
                
                #else
                    
                    for (int i2= -Nweno/2 ; i2<= Nweno/2; i2++)
                    for (int i1= -Nweno/2 ; i1<= Nweno/2; i1++)
                        Kernel_1D(El[0][i1+Nweno/2][i2+Nweno/2],
                                  El[1][i1+Nweno/2][i2+Nweno/2],
                                  El[2][i1+Nweno/2][i2+Nweno/2],
                                  El[3][i1+Nweno/2][i2+Nweno/2],
                                  El[4][i1+Nweno/2][i2+Nweno/2],
                                  Lines[i1+Nweno/2][i2+Nweno/2][0],
                                  Lines[i1+Nweno/2][i2+Nweno/2][1]);
                   
    
                    for (int i2= -Nweno/2 ; i2<= Nweno/2; i2++)
                    {
                        Kernel_1D(Lines[0][i2+Nweno/2][0],
                                  Lines[1][i2+Nweno/2][0],
                                  Lines[2][i2+Nweno/2][0],
                                  Lines[3][i2+Nweno/2][0],
                                  Lines[4][i2+Nweno/2][0],
                                  Planes[i2+Nweno/2][0],
                                  Planes[i2+Nweno/2][1]);
              
                        Kernel_1D(Lines[0][i2+Nweno/2][1],
                                  Lines[1][i2+Nweno/2][1],
                                  Lines[2][i2+Nweno/2][1],
                                  Lines[3][i2+Nweno/2][1],
                                  Lines[4][i2+Nweno/2][1],
                                  Planes[i2+Nweno/2][2],
                                  Planes[i2+Nweno/2][3]);
                    }
                    Kernel_1D(Planes[0][0],Planes[1][0],Planes[2][0],Planes[3][0],Planes[4][0],Ref[0],Ref[1]);
                    Kernel_1D(Planes[0][1],Planes[1][1],Planes[2][1],Planes[3][1],Planes[4][1],Ref[2],Ref[3]);
                    Kernel_1D(Planes[0][2],Planes[1][2],Planes[2][2],Planes[3][2],Planes[4][2],Ref[4],Ref[5]);
                    Kernel_1D(Planes[0][3],Planes[1][3],Planes[2][3],Planes[3][3],Planes[4][3],Ref[6],Ref[7]);
                
                #endif
    
                b(i  ,j  ,k  ) = Ref[0];
                b(i  ,j  ,k+1) = Ref[1];
                b(i  ,j+1,k  ) = Ref[2];
                b(i  ,j+1,k+1) = Ref[3];
                b(i+1,j  ,k  ) = Ref[4];
                b(i+1,j  ,k+1) = Ref[5];
                b(i+1,j+1,k  ) = Ref[6];
                b(i+1,j+1,k+1) = Ref[7];
            #endif         
            }         
        }
    }

    virtual 
    void WENOWavelets3(double cm, double c , double cp, double & left, double & right)
    {
        double b1 = (c-cm)*(c-cm);
        double b2 = (c-cp)*(c-cp);
        double w1 = (1e-6 + b2)*(1e-6 + b2); //yes, 2 goes to 1 and 1 goes to 2
        double w2 = (1e-6 + b1)*(1e-6 + b1);
        double aux = 1.0 / (w1+w2);
        w1 *= aux; w2 *= aux;
        double g1,g2;
        g1 = 0.75*c + 0.25*cm;
        g2 = 1.25*c - 0.25*cp;
        left = g1*w1+g2*w2;
        g1 = 1.25*c - 0.25*cm;
        g2 = 0.75*c + 0.25*cp;
        right = g1*w1+g2*w2;
    }


    virtual 
    void WENOWavelets5(double cm2, double cm , double c, double cp, double cp2, double & left, double & right)
    {
        static const double k  = 13.0/12.0;
        static const double d1 = 0.625;
        static const double d2 = 0.1875;
        static const double d3 = 0.1875;

        double b1 = k * pow(cm - 2.0 * c   + cp ,2) + 0.25*pow(cm-cp,2);
        double b2 = k * pow(c  - 2.0 * cp  + cp2,2) + 0.25*pow(3.0*c-4.0*cp+cp2,2);
        double b3 = k * pow(c  - 2.0 * cm  + cm2,2) + 0.25*pow(3.0*c-4.0*cm+cm2,2);

        double w1 = d1 / pow(1e-6 + b1,2);
        double w2 = d2 / pow(1e-6 + b2,2);
        double w3 = d3 / pow(1e-6 + b3,2);
        double aux = 1.0 / (w1+w2+w3);

        double g1,g2,g3;

        g1 = 0.125*cm +       c  - 0.125*cp ;
        g2 = 1.375*c  - 0.5 * cp + 0.125*cp2;
        g3 = 0.625*c  + 0.5 * cm - 0.125*cm2;
        left = aux*(g1*w1+g2*w2+g3*w3);
    
        g1 = -0.125*cm +       c  + 0.125*cp ;
        g2 = 0.625*c  + 0.5 * cp - 0.125*cp2;
        g3 = 1.375*c  - 0.5 * cm + 0.125*cm2;
        right = aux*(g1*w1+g2*w2+g3*w3);   
    }







    #if WENOWAVELET == 3
    virtual 
    void Kernel_1D(ElementType E0,ElementType E1,ElementType E2, ElementType & left, ElementType & right)
    {
        left .dummy = E1.dummy;// - 0.125*(E2.dummy-E0.dummy);
        right.dummy = E1.dummy;// + 0.125*(E2.dummy-E0.dummy);
        WENOWavelets3(E0.alpha1rho1,E1.alpha1rho1,E2.alpha1rho1,left.alpha1rho1,right.alpha1rho1);
        WENOWavelets3(E0.alpha2rho2,E1.alpha2rho2,E2.alpha2rho2,left.alpha2rho2,right.alpha2rho2);
        WENOWavelets3(E0.ru        ,E1.ru        ,E2.ru        ,left.ru        ,right.ru        );
        WENOWavelets3(E0.rv        ,E1.rv        ,E2.rv        ,left.rv        ,right.rv        );
        WENOWavelets3(E0.rw        ,E1.rw        ,E2.rw        ,left.rw        ,right.rw        );
        WENOWavelets3(E0.alpha2    ,E1.alpha2    ,E2.alpha2    ,left.alpha2    ,right.alpha2    );
        WENOWavelets3(E0.energy    ,E1.energy    ,E2.energy    ,left.energy    ,right.energy    );
  

        //clipping
        if  ( left.alpha2 < 0.0 || right.alpha2 < 0.0 || left.alpha2 > 1.0 || right.alpha2 > 1.0)
        {
            left .alpha2 = E1.alpha2;
            right.alpha2 = E1.alpha2;
        } 
    
        if  ( left.alpha1rho1 < 0.0 || right.alpha1rho1 < 0.0 || left.alpha1rho1 > 1.0 || right.alpha1rho1 > 1.0)
        {
            left .alpha1rho1 = E1.alpha1rho1;
            right.alpha1rho1 = E1.alpha1rho1;
        } 
    
        if  ( left.alpha2rho2 < 0.0 || right.alpha2rho2 < 0.0 || left.alpha2rho2 > 1.0 || right.alpha2rho2 > 1.0)
        {
            left .alpha2rho2 = E1.alpha2rho2;
            right.alpha2rho2 = E1.alpha2rho2;
        } 
        
        if  ( left.energy < 0.0 || right.energy < 0.0 )
        {
            left .energy = E1.energy;
            right.energy = E1.energy;
        } 
    }
    #else
    virtual 
    void Kernel_1D(ElementType E0,ElementType E1,ElementType E2,ElementType E3,ElementType E4,ElementType & left, ElementType & right)
    {
        left .dummy = E2.dummy;// - 0.125*(E2.dummy-E0.dummy);
        right.dummy = E2.dummy;// + 0.125*(E2.dummy-E0.dummy);
        

        WENOWavelets5(E0.alpha1rho1,E1.alpha1rho1,E2.alpha1rho1,E3.alpha1rho1,E4.alpha1rho1,left.alpha1rho1,right.alpha1rho1);
        WENOWavelets5(E0.alpha2rho2,E1.alpha2rho2,E2.alpha2rho2,E3.alpha2rho2,E4.alpha2rho2,left.alpha2rho2,right.alpha2rho2);
        WENOWavelets5(E0.ru        ,E1.ru        ,E2.ru        ,E3.ru        ,E4.ru        ,left.ru        ,right.ru        );
        WENOWavelets5(E0.rv        ,E1.rv        ,E2.rv        ,E3.rv        ,E4.rv        ,left.rv        ,right.rv        );
        WENOWavelets5(E0.rw        ,E1.rw        ,E2.rw        ,E3.rw        ,E4.rw        ,left.rw        ,right.rw        );
        WENOWavelets5(E0.alpha2    ,E1.alpha2    ,E2.alpha2    ,E3.alpha2    ,E4.alpha2    ,left.alpha2    ,right.alpha2    );
        WENOWavelets5(E0.energy    ,E1.energy    ,E2.energy    ,E3.energy    ,E4.energy    ,left.energy    ,right.energy    );
  
        //clipping
        if  ( left.alpha2 < 0.0 || right.alpha2 < 0.0 || left.alpha2 > 1.0 || right.alpha2 > 1.0)
        {
            left .alpha2 = E2.alpha2;
            right.alpha2 = E2.alpha2;
        } 
    
        if  ( left.alpha1rho1 < 0.0 || right.alpha1rho1 < 0.0 || left.alpha1rho1 > 1.0 || right.alpha1rho1 > 1.0)
        {
            left .alpha1rho1 = E2.alpha1rho1;
            right.alpha1rho1 = E2.alpha1rho1;
        } 
    
        if  ( left.alpha2rho2 < 0.0 || right.alpha2rho2 < 0.0 || left.alpha2rho2 > 1.0 || right.alpha2rho2 > 1.0)
        {
            left .alpha2rho2 = E2.alpha2rho2;
            right.alpha2rho2 = E2.alpha2rho2;
        } 
        
        if  ( left.energy < 0.0 || right.energy < 0.0 )
        {
            left .energy = E2.energy;
            right.energy = E2.energy;
        } 
    }
    #endif                         
    //Tag block loaded in Lab for refinement/compression; user can write own function
    virtual 
    State TagLoadedBlock(TLab & Lab_)
    {
        static const int nx = BlockType::sizeX;
        static const int ny = BlockType::sizeY;
        static const int nz = BlockType::sizeZ;
            
        double L1   = 0.0;
        double Linf = 0.0;
      
        for (int k=0; k<nz; k++ )
        for (int j=0; j<ny; j++ )
        for (int i=0; i<nx; i++ )
        {
          double dudx = 0.5*( Lab_(i+1,j  ,k  ).energy-Lab_(i-1,j  ,k  ).energy);
          double dudy = 0.5*( Lab_(i  ,j+1,k  ).energy-Lab_(i  ,j-1,k  ).energy);
          double dudz = 0.5*( Lab_(i  ,j  ,k+1).energy-Lab_(i  ,j  ,k-1).energy);  
          
          double gradMag = sqrt(dudx*dudx+dudy*dudy+dudz*dudz);
          gradMag /= ( 1e-6 + Lab_(i,j,k).energy); 
    
          #if 1
            dudx = 0.5*( Lab_(i+1,j  ,k  ).alpha2-Lab_(i-1,j  ,k  ).alpha2);
            dudy = 0.5*( Lab_(i  ,j+1,k  ).alpha2-Lab_(i  ,j-1,k  ).alpha2);
            dudz = 0.5*( Lab_(i  ,j  ,k+1).alpha2-Lab_(i  ,j  ,k-1).alpha2);         
            double gradMag1 = sqrt(dudx*dudx+dudy*dudy+dudz*dudz);
            gradMag1 /= ( 1e-6 + Lab_(i,j,k).alpha2); 
            gradMag = max(gradMag,gradMag1);
          #endif
    
          L1 += gradMag;
          Linf = max(Linf,gradMag);       
        }
        L1 /= (nz*ny*nx);     
    
        if      (Linf > tolerance_for_refinement) return Refine;
        else if (Linf < tolerance_for_compression) return Compress;
        else return Leave;
    }

};


        

}//namespace AMR_CUBISM
