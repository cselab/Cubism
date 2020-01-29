#pragma once
#include "Matrix3D.h"
#include "BlockInfo.h"
#include "GridMPI.h"
#include "BlockLabMPI.h"

#include <omp.h>
#include <cstring>
#include <string>
#include <algorithm>




#include <chrono>
typedef std::chrono::high_resolution_clock Clock;


#ifdef __bgq__
#include <builtins.h>
#define memcpy2(a,b,c)  __bcopy((b),(a),(c))
#else
#define memcpy2(a,b,c)  memcpy((a),(b),(c))
#endif


namespace cubism//AMR_CUBISM
{

template<typename TMeshAdaptation, typename TLab>
class MeshAdaptationMPI: public TMeshAdaptation
{

public:
  typedef typename TMeshAdaptation::GridType TGrid;
  typedef typename TMeshAdaptation::BlockType Block;
  typedef typename TMeshAdaptation::BlockType BlockType;
  typedef typename TMeshAdaptation::ElementType ElementType;

  typedef SynchronizerMPI_AMR<Real> SynchronizerMPIType;

   


  TLab * LabMPI;

  MeshAdaptationMPI(TGrid & grid, double Rtol,double Ctol): TMeshAdaptation(grid, Rtol, Ctol)
  {

  }


  ~MeshAdaptationMPI()
  {

  }



#if 1



  virtual void AdaptTheMesh(double t = 0, bool reload = true) override 
  {

    SynchronizerMPI_AMR<Real> * Synch = sync();

    vector<BlockInfo> avail0, avail1;

    const int nthreads = omp_get_max_threads();
    
    LabMPI = new TLab[nthreads];
    for (int i=0; i<nthreads; i++)
      LabMPI[i].prepare((*TMeshAdaptation::m_refGrid), *Synch);


    MPI_Barrier(MPI_COMM_WORLD); //is it necessary?? 






    static int rounds = -1;
    static int one_less = 1;
    if (rounds == -1)
    {
      char *s = getenv("MYROUNDS");
      if (s != NULL)
          rounds = atoi(s);
      else
          rounds = 0;
       char *s2 = getenv("USEMAXTHREADS");
      if (s2 != NULL)
          one_less = !atoi(s2);
    }
  
   


    ///////////////////////////////////////////////////////////////////////
    //avail0 = Synch.avail_inner(); //uncomment this to go back to original
    auto tmpCrap =  Synch->avail_inner(); //this crap is needed for now.../*mike*/
    avail0 = (*TMeshAdaptation::m_refGrid).getBlocksInfo(); /*mike*/
    ///////////////////////////////////////////////////////////////////////



    const int Ninner = avail0.size();
    BlockInfo * ary0 = &avail0.front();
  


    int nthreads_first;
    if (one_less)
        nthreads_first = nthreads-1;
    else
        nthreads_first = nthreads;

    if (nthreads_first == 0) nthreads_first = 1;

    int Ninner_first = (nthreads_first)*rounds;
    if (Ninner_first > Ninner) Ninner_first = Ninner;
    int Ninner_rest = Ninner - Ninner_first;






#pragma omp parallel num_threads(nthreads_first)
    {
        int tid = omp_get_thread_num();
        TLab& mylab = LabMPI[tid];

#pragma omp for schedule(dynamic,1)
        for(int i=0; i<Ninner_first; i++)
        {
            assert(false && "BlockProcessor_MPI assertion 2");
            //mylab.load(ary0[i], t);
            //rhs(mylab, ary0[i], *(typename TGrid::BlockType*)ary0[i].ptrBlock);
        }
    }

    avail1 = Synch->avail_halo();
    const int Nhalo = avail1.size();
    BlockInfo * ary1 = nullptr; //&avail1.front(); //nullptr




#pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        TLab& mylab = LabMPI[tid];

#pragma omp for schedule(dynamic,1)
        for(int i=-Ninner_rest; i<Nhalo; i++)
        {
            if (i < 0)
            {
                int ii = i + Ninner;
                
                mylab.load(ary0[ii], t);

                BlockInfo & info = TMeshAdaptation::m_refGrid->getBlockInfoAll(ary0[ii].level,ary0[ii].Z);
                
                ary0[ii].state = Leave;
                //if (ary0[ii].level == 0 && ary0[ii].Z ==3) ary0[ii].state = Refine;
                //if (ary0[ii].level == 0 && ary0[ii].Z ==5) ary0[ii].state = Refine;
                

                ary0[ii].state = TagLoadedBlock__(LabMPI[tid]);
                info.state = ary0[ii].state;                          
 
            }
            else
            {
                assert(false && "BlockProcessor_MPI assertion 1");
            //    mylab.load(ary1[i], t);
            //    rhs(mylab, ary1[i], *(typename TGrid::BlockType*)ary1[i].ptrBlock);
            }
        }
    }

   
    MPI_Barrier(MPI_COMM_WORLD);








   ValidStates();





    int r=0;
    int c=0;
  
    std::vector <int> mn_com;
    std::vector <int> mn_ref;

    std::vector <BlockInfo> & I = TMeshAdaptation::m_refGrid->getBlocksInfo();   

    for ( auto & i: I)
    {
      BlockInfo & info =  TMeshAdaptation::m_refGrid->getBlockInfoAll(i.level,i.Z);                  
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
           
    //Does not work for some reason #pragma omp parallel for 
    for (size_t i=0; i<mn_com.size()/2; i++)
    {
      int m = mn_com[2*i];
      int n = mn_com[2*i+1];               
      TMeshAdaptation::compress(m,n);
      #pragma omp atomic
        c++;
    }

    //Does not work for some reason #pragma omp parallel for 
    for (size_t i=0; i<mn_ref.size()/2; i++)
    {
      int m = mn_ref[2*i];
      int n = mn_ref[2*i+1]; 
      refine(m,n);
      #pragma omp atomic
        r++;
    }


    int temp[2] = {r,c};
    int result[2];
    MPI_Reduce(&temp, &result, 2, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if (rank ==0)
    {     
      std::cout <<"==============================================================\n";
      std::cout << " refined:" << result[0] << "   compressed:"<< result[1] << std::endl;
      std::cout <<"==============================================================\n";
    }



    TMeshAdaptation::m_refGrid->FillPos();
    TMeshAdaptation::m_refGrid->UpdateBlockInfoAll();


    delete [] LabMPI;
    delete Synch;

}





  //Tag block loaded in Lab for refinement/compression; user can write own function
  virtual State TagLoadedBlock__(TLab & Lab_) 
  {
    const int nx = BlockType::sizeX;
    const int ny = BlockType::sizeY;
    const int nz = BlockType::sizeZ;
        
    double L1   = 0.0;
    double Linf = 0.0;
  
    for (int k=0; k<nz; k++ )
    for (int j=0; j<ny; j++ )
    for (int i=0; i<nx; i++ )
    {


      double dudx = 0.5*( Lab_(i+1,j  ,k  ).energy-Lab_(i-1,j  ,k  ).energy);
      double dudy = 0.5*( Lab_(i  ,j+1,k  ).energy-Lab_(i  ,j-1,k  ).energy);
      double dudz = 0.5*( Lab_(i  ,j  ,k+1).energy-Lab_(i  ,j  ,k-1).energy);  
      
      #if 0
        if (i==0)
          dudx = ( Lab_(i+1,j  ,k  ).energy-Lab_(i  ,j  ,k  ).energy);
        else if (i==nx-1)
          dudx = ( Lab_(i  ,j  ,k  ).energy-Lab_(i-1,j  ,k  ).energy);
        if (j==0)
          dudy = ( Lab_(i  ,j+1,k  ).energy-Lab_(i  ,j  ,k  ).energy);
        else if (j==ny-1)
          dudy = ( Lab_(i  ,j  ,k  ).energy-Lab_(i  ,j-1,k  ).energy);
        if (k==0)
          dudz = ( Lab_(i  ,j  ,k+1).energy-Lab_(i  ,j  ,k  ).energy);
        else if (k==nz-1)
          dudz = ( Lab_(i  ,j  ,k  ).energy-Lab_(i  ,j  ,k-1).energy);
      #endif


      double gradMag = sqrt(dudx*dudx+dudy*dudy+dudz*dudz);
      gradMag /= ( 1e-6 + Lab_(i,j,k).energy); 

      #if 0
        dudx = 0.5*( Lab(i+1,j  ,k  ).alpha2-Lab(i-1,j  ,k  ).alpha2);
        dudy = 0.5*( Lab(i  ,j+1,k  ).alpha2-Lab(i  ,j-1,k  ).alpha2);
        dudz = 0.5*( Lab(i  ,j  ,k+1).alpha2-Lab(i  ,j  ,k-1).alpha2);         
        double gradMag1 = sqrt(dudx*dudx+dudy*dudy+dudz*dudz);
        gradMag1 /= ( 1e-4 + Lab(i,j,k).alpha2); 
        gradMag = max(gradMag,gradMag1);
      #endif


      L1 += gradMag;
      Linf = max(Linf,gradMag);       
    }
    L1 /= (nz*ny*nx);    




    if      (Linf > TMeshAdaptation::tolerance_for_refinement) return Refine;
    else if (Linf < TMeshAdaptation::tolerance_for_compression) return Compress;
    else return Leave;
  }









protected:
  SynchronizerMPIType * sync()
  {
    auto blockperDim = TMeshAdaptation::m_refGrid->getMaxBlocks();     
    
    TLab dummy;
    bool per [3] = {dummy.is_xperiodic(),dummy.is_yperiodic(),dummy.is_zperiodic()}; 

    StencilInfo stencil(TMeshAdaptation::s[0],TMeshAdaptation::s[1],TMeshAdaptation::s[2],
                        TMeshAdaptation::e[0],TMeshAdaptation::e[1],TMeshAdaptation::e[2],TMeshAdaptation::istensorial,TMeshAdaptation::components);

    StencilInfo Cstencil = stencil;

    SynchronizerMPIType * queryresult = NULL;
        
    queryresult = new SynchronizerMPIType(stencil, Cstencil, MPI_COMM_WORLD, per, TMeshAdaptation::m_refGrid->getlevelMax(),
                                          TGrid::Block::sizeX,
                                          TGrid::Block::sizeY,
                                          TGrid::Block::sizeZ,
                                          blockperDim[0],
                                          blockperDim[1],
                                          blockperDim[2],
                                          TMeshAdaptation::m_refGrid->getBlocksInfo(),TMeshAdaptation::m_refGrid->getBlockInfoAll());      
    int timestamp = 0;


    queryresult->sync(sizeof(typename Block::element_type)/sizeof(Real), sizeof(Real)>4 ? MPI_DOUBLE : MPI_FLOAT, timestamp);
    return queryresult;
  }





 virtual void refine(int level, int Z) override
  {
    int tid = omp_get_thread_num();
      
    BlockInfo & parent =  TMeshAdaptation::m_refGrid->getBlockInfoAll(level,Z);

    int p[3] = {parent.index[0],parent.index[1],parent.index[2]};

    assert(parent.ptrBlock != NULL);
    assert(level <= TMeshAdaptation::m_refGrid->getlevelMax()-1);


    LabMPI[tid].load(parent);   
      
    parent.TreePos = CheckFiner;
    int nChild = TMeshAdaptation::m_refGrid->getZchild(level,parent.index[0],parent.index[1],parent.index[2]);



    BlockType * Blocks [8];
        
    for (int k=0; k<2; k++ )
    for (int j=0; j<2; j++ )
    for (int i=0; i<2; i++ )
    {      
      int nc = TMeshAdaptation::m_refGrid->getZforward(level+1,2*p[0]+i,2*p[1]+j,2*p[2]+k);
             
      BlockInfo & Child = TMeshAdaptation::m_refGrid->getBlockInfoAll(level+1,nc); 

      Child.TreePos = Exists;
      Child.myrank = TMeshAdaptation::m_refGrid->rank();

      if (nc !=nChild)
      {
        assert(!(i==0 && j==0 && k==0));
        #pragma omp critical
        {
          TMeshAdaptation::m_refGrid->_alloc(level+1,nc);
        }
      }
      else
      {
        assert(i==0 && j==0 && k==0);
        //std::swap(Child.ptrBlock, parent.ptrBlock);
        Child.ptrBlock = parent.ptrBlock;
        Child.h_gridpoint = Child.h;
        parent.ptrBlock = nullptr;
      }
      Blocks [k*4 + j*2 + i] = (BlockType*) Child.ptrBlock;
    }

    //TMeshAdaptation::
    RefineBlocks(Blocks);  
    parent.myrank = -1;
  }












 virtual void RefineBlocks(BlockType * B[8]) override
  {
    int tid = omp_get_thread_num();
      
    const int nx = BlockType::sizeX;
    const int ny = BlockType::sizeY;
    const int nz = BlockType::sizeZ;
      
    int offsetX [2] = {0,nx/2};
    int offsetY [2] = {0,ny/2};
    int offsetZ [2] = {0,nz/2};


    TLab & Lab = LabMPI[tid];


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
         #if 0



            ElementType dudx = 0.5*( Lab(i/2+offsetX[I]+1,j/2+offsetY[J]  ,k/2+offsetZ[K]  )-Lab(i/2+offsetX[I]-1,j/2+offsetY[J]  ,k/2+offsetZ[K]  ));
            ElementType dudy = 0.5*( Lab(i/2+offsetX[I]  ,j/2+offsetY[J]+1,k/2+offsetZ[K]  )-Lab(i/2+offsetX[I]  ,j/2+offsetY[J]-1,k/2+offsetZ[K]  ));
            ElementType dudz = 0.5*( Lab(i/2+offsetX[I]  ,j/2+offsetY[J]  ,k/2+offsetZ[K]+1)-Lab(i/2+offsetX[I]  ,j/2+offsetY[J]  ,k/2+offsetZ[K]-1));

            

            if (i==0)
              dudx = ( Lab(i/2+offsetX[I]+1,j/2+offsetY[J]  ,k/2+offsetZ[K]  )-Lab(i/2+offsetX[I],j/2+offsetY[J]  ,k/2+offsetZ[K]  ));
            else if (i==nx-2)
              dudx = ( Lab(i/2+offsetX[I],j/2+offsetY[J]  ,k/2+offsetZ[K]  )-Lab(i/2+offsetX[I]-1,j/2+offsetY[J]  ,k/2+offsetZ[K]  ));
     
            if (j==0)
              dudy = ( Lab(i/2+offsetX[I],j/2+offsetY[J]+1  ,k/2+offsetZ[K]  )-Lab(i/2+offsetX[I],j/2+offsetY[J]  ,k/2+offsetZ[K]  ));
            else if (j==ny-2)
              dudy = ( Lab(i/2+offsetX[I],j/2+offsetY[J]  ,k/2+offsetZ[K]  )-Lab(i/2+offsetX[I],j/2+offsetY[J]-1  ,k/2+offsetZ[K]  ));
     
            if (k==0)
              dudz = ( Lab(i/2+offsetX[I],j/2+offsetY[J]  ,k/2+offsetZ[K]+1)-Lab(i/2+offsetX[I],j/2+offsetY[J]  ,k/2+offsetZ[K]  ));
            else if (k==nz-2)
              dudz = ( Lab(i/2+offsetX[I],j/2+offsetY[J]  ,k/2+offsetZ[K]  )-Lab(i/2+offsetX[I],j/2+offsetY[J]  ,k/2+offsetZ[K]-1));
     



            b(i  ,j  ,k  ) = Lab( i   /2+offsetX[I], j   /2+offsetY[J]  ,k    /2+offsetZ[K] )+ (2*( i   %2)-1)*0.25*dudx + (2*( j   %2)-1)*0.25*dudy + (2*(k    %2)-1)*0.25*dudz; 
            b(i+1,j  ,k  ) = Lab((i+1)/2+offsetX[I], j   /2+offsetY[J]  ,k    /2+offsetZ[K] )+ (2*((i+1)%2)-1)*0.25*dudx + (2*( j   %2)-1)*0.25*dudy + (2*(k    %2)-1)*0.25*dudz; 
            b(i  ,j+1,k  ) = Lab( i   /2+offsetX[I],(j+1)/2+offsetY[J]  ,k    /2+offsetZ[K] )+ (2*( i   %2)-1)*0.25*dudx + (2*((j+1)%2)-1)*0.25*dudy + (2*(k    %2)-1)*0.25*dudz; 
            b(i+1,j+1,k  ) = Lab((i+1)/2+offsetX[I],(j+1)/2+offsetY[J]  ,k    /2+offsetZ[K] )+ (2*((i+1)%2)-1)*0.25*dudx + (2*((j+1)%2)-1)*0.25*dudy + (2*(k    %2)-1)*0.25*dudz; 
            b(i  ,j  ,k+1) = Lab( i   /2+offsetX[I], j   /2+offsetY[J]  ,(k+1)/2+offsetZ[K] )+ (2*( i   %2)-1)*0.25*dudx + (2*( j   %2)-1)*0.25*dudy + (2*((k+1)%2)-1)*0.25*dudz; 
            b(i+1,j  ,k+1) = Lab((i+1)/2+offsetX[I], j   /2+offsetY[J]  ,(k+1)/2+offsetZ[K] )+ (2*((i+1)%2)-1)*0.25*dudx + (2*( j   %2)-1)*0.25*dudy + (2*((k+1)%2)-1)*0.25*dudz; 
            b(i  ,j+1,k+1) = Lab( i   /2+offsetX[I],(j+1)/2+offsetY[J]  ,(k+1)/2+offsetZ[K] )+ (2*( i   %2)-1)*0.25*dudx + (2*((j+1)%2)-1)*0.25*dudy + (2*((k+1)%2)-1)*0.25*dudz; 
            b(i+1,j+1,k+1) = Lab((i+1)/2+offsetX[I],(j+1)/2+offsetY[J]  ,(k+1)/2+offsetZ[K] )+ (2*((i+1)%2)-1)*0.25*dudx + (2*((j+1)%2)-1)*0.25*dudy + (2*((k+1)%2)-1)*0.25*dudz; 
         #else
          const int Nweno = 3;
          ElementType El[Nweno][Nweno][Nweno];
          for (int i0= -Nweno/2 ; i0<= Nweno/2; i0++)
          for (int i1= -Nweno/2 ; i1<= Nweno/2; i1++)
          for (int i2= -Nweno/2 ; i2<= Nweno/2; i2++)
          //            El[i0+Nweno/2][i1+Nweno/2][i2+Nweno/2] = LabOMP[tid](i/2+offsetX[I] + i0,j/2+offsetY[J]+ i1 ,k/2+offsetZ[K]+ i2);      
          El[i0+Nweno/2][i1+Nweno/2][i2+Nweno/2] = Lab(i/2+offsetX[I] + i0,j/2+offsetY[J]+ i1 ,k/2+offsetZ[K]+ i2);      


          ElementType Lines [Nweno][Nweno][2];
          ElementType Planes[Nweno][4];
          ElementType Ref          [8]; 


      for (int i2= -Nweno/2 ; i2<= Nweno/2; i2++)
      {
        for (int i1= -Nweno/2 ; i1<= Nweno/2; i1++)
            {
              TMeshAdaptation::Kernel_1D(El[0][i1+Nweno/2][i2+Nweno/2],
                      El[1][i1+Nweno/2][i2+Nweno/2],
                      El[2][i1+Nweno/2][i2+Nweno/2],
                      Lines[i1+Nweno/2][i2+Nweno/2][0],
                      Lines[i1+Nweno/2][i2+Nweno/2][1]);
            } 
      }

      




          for (int i2= -Nweno/2 ; i2<= Nweno/2; i2++)
          {
          TMeshAdaptation::Kernel_1D(Lines[0][i2+Nweno/2][0],
                      Lines[1][i2+Nweno/2][0],
                    Lines[2][i2+Nweno/2][0],
                    Planes[i2+Nweno/2][0],
                    Planes[i2+Nweno/2][1]);

          TMeshAdaptation::Kernel_1D(Lines[0][i2+Nweno/2][1],
                    Lines[1][i2+Nweno/2][1],
                    Lines[2][i2+Nweno/2][1],
                    Planes[i2+Nweno/2][2],
                    Planes[i2+Nweno/2][3]);
          }

        

      TMeshAdaptation::Kernel_1D(Planes[0][0],
                    Planes[1][0],
                  Planes[2][0],
                  Ref[0],
                  Ref[1]);
      
          TMeshAdaptation::Kernel_1D(Planes[0][1],
                    Planes[1][1],
                  Planes[2][1],
                  Ref[2],
                  Ref[3]);
      
          TMeshAdaptation::Kernel_1D(Planes[0][2],
                    Planes[1][2],
                  Planes[2][2],
                  Ref[4],
                  Ref[5]);
         
          TMeshAdaptation::Kernel_1D(Planes[0][3],
                    Planes[1][3],
                  Planes[2][3],
                  Ref[6],
                  Ref[7]);
      
            b(i  ,j  ,k  ) = Ref[0];
            b(i  ,j  ,k+1) = Ref[1];
            b(i  ,j+1,k  ) = Ref[2];
            b(i  ,j+1,k+1) = Ref[3];
            b(i+1,j  ,k  ) = Ref[4];
            b(i+1,j  ,k+1) = Ref[5];
            b(i+1,j+1,k  ) = Ref[6];
            b(i+1,j+1,k+1) = Ref[7];

            //b(i,j,k) = Lab(i/2+offsetX[I],j/2+offsetY[J]  ,k/2+offsetZ[K] )+ (2*(i%2)-1)*0.25*dudx + (2*(j%2)-1)*0.25*dudy + (2*(k%2)-1)*0.25*dudz; 
          #endif

            


            for (int kk=0; kk<2; kk++)
              for (int jj=0; jj<2; jj++)
                for (int ii=0; ii<2; ii++)
                {
                  assert  (!isnan(b(i+ii,j+jj,k+kk).alpha1rho1));
                  assert  (!isnan(b(i+ii,j+jj,k+kk).alpha2rho2));
                  assert  (!isnan(b(i+ii,j+jj,k+kk).ru        ));
                  assert  (!isnan(b(i+ii,j+jj,k+kk).rv        ));
                  assert  (!isnan(b(i+ii,j+jj,k+kk).rw        ));
                  assert  (!isnan(b(i+ii,j+jj,k+kk).alpha2    ));
                  assert  (!isnan(b(i+ii,j+jj,k+kk).energy    ));
                  assert  (!isnan(b(i+ii,j+jj,k+kk).dummy     ));
                  assert  (abs(b(i+ii,j+jj,k+kk).alpha1rho1) < 1e20 );
                  assert  (abs(b(i+ii,j+jj,k+kk).alpha2rho2) < 1e20 );
                  assert  (abs(b(i+ii,j+jj,k+kk).ru        ) < 1e20 );
                  assert  (abs(b(i+ii,j+jj,k+kk).rv        ) < 1e20 );
                  assert  (abs(b(i+ii,j+jj,k+kk).rw        ) < 1e20 );
                  assert  (abs(b(i+ii,j+jj,k+kk).alpha2    ) < 1e20 );
                  assert  (abs(b(i+ii,j+jj,k+kk).energy    ) < 1e20 );
                  assert  (abs(b(i+ii,j+jj,k+kk).dummy     ) < 1e20 );
                
                }














        }         
      }
  }











#endif













  virtual void ValidStates() override
  {
    std::vector <BlockInfo> & I = TMeshAdaptation::m_refGrid->getBlocksInfo();
    std::array <int,3> blocksPerDim = TMeshAdaptation::m_refGrid->getMaxBlocks();

    int levelMin = 0;
    int levelMax = TMeshAdaptation::m_refGrid->getlevelMax();


 
    TMeshAdaptation::m_refGrid->FillPos();
    TMeshAdaptation::m_refGrid->UpdateBlockInfoAll_States();
    for ( auto & b: I)
    {
        BlockInfo & info = TMeshAdaptation::m_refGrid->getBlockInfoAll(b.level,b.Z);
        if (info.state==Refine   && info.level ==levelMax-1) info.state=Leave;
        if (info.state==Compress && info.level ==levelMin  ) info.state=Leave;
    }

    TMeshAdaptation::m_refGrid->FillPos();
    TMeshAdaptation::m_refGrid->UpdateBlockInfoAll_States();

    const bool xperiodic = LabMPI[0].is_xperiodic();
    const bool yperiodic = LabMPI[0].is_yperiodic();
    const bool zperiodic = LabMPI[0].is_zperiodic();

    for (int m=levelMax-1; m>=levelMin; m--)
    { 
      //1.Change states of blocks next to finer resolution blocks
      //2.Change states of blocks next to same resolution blocks
      //3.Compress a block only if all blocks with the same parent need compression

  
      //1.
      for ( auto & b: I) if (b.level == m)
      {
        BlockInfo & info =  TMeshAdaptation::m_refGrid->getBlockInfoAll(m,b.Z);
      
        assert(b.TreePos == Exists);
        
        int TwoPower = pow(2,info.level);
        const bool xskin = info.index[0]==0 || info.index[0]==blocksPerDim[0]*TwoPower-1;
        const bool yskin = info.index[1]==0 || info.index[1]==blocksPerDim[1]*TwoPower-1;
        const bool zskin = info.index[2]==0 || info.index[2]==blocksPerDim[2]*TwoPower-1;
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

          BlockInfo & infoNei = TMeshAdaptation::m_refGrid->getBlockInfoAll(info.level_(),info.Znei_(code[0],code[1],code[2]) );
            
          if (infoNei.TreePos == CheckFiner && info.state!=Refine)
          {
            info.state=Leave;
            int Bstep = 1; //face
            if      ((abs(code[0])+abs(code[1])+abs(code[2])==2 )) Bstep = 3; //edge
            else if ((abs(code[0])+abs(code[1])+abs(code[2])==3 )) Bstep = 4; //corner
      
            for (int B = 0 ; B <= 3 ; B += Bstep) //loop over blocks that make up face/edge/corner (respectively 4,2 or 1 blocks)
            {
              const int aux = (abs(code[0])==1) ? (B%2) : (B/2) ;
                            
              int iNei = 2*info.index[0] + max(code[0],0) +code[0]  + (B%2)*max(0, 1 - abs(code[0]));
              int jNei = 2*info.index[1] + max(code[1],0) +code[1]  +  aux *max(0, 1 - abs(code[1]));
              int kNei = 2*info.index[2] + max(code[2],0) +code[2]  + (B/2)*max(0, 1 - abs(code[2]));
              int zzz = TMeshAdaptation::m_refGrid->getZforward(m+1,iNei,jNei,kNei);
              BlockInfo & FinerNei = TMeshAdaptation::m_refGrid->getBlockInfoAll(m+1,zzz);
              State NeiState = FinerNei.state;
              if (NeiState == Refine)
              {
                info.state=Refine;
                break;
              }
            }
          }
        }
      }
  
    TMeshAdaptation::m_refGrid->FillPos();
    TMeshAdaptation::m_refGrid->UpdateBlockInfoAll_States();



      //2.
      for ( auto & b: I) if (b.level == m)
      {
        BlockInfo & info =  TMeshAdaptation::m_refGrid->getBlockInfoAll(m,b.Z);
        int aux = pow(2,info.level);
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
          BlockInfo & infoNei = TMeshAdaptation::m_refGrid->getBlockInfoAll(info.level_(),info.Znei_(code[0],code[1],code[2]) );
          
          if (infoNei.TreePos == Exists && infoNei.state==Refine && info.state==Compress)
            info.state=Leave;
        }
      }


    TMeshAdaptation::m_refGrid->FillPos();
    TMeshAdaptation::m_refGrid->UpdateBlockInfoAll_States();


      //3.
      for ( auto & b: I) if (b.level == m)
      {
        BlockInfo & info =  TMeshAdaptation::m_refGrid->getBlockInfoAll(m,b.Z);       
        if (info.state==Compress)
          for (int i= 2*(info.index[0]/2); i <= 2*(info.index[0]/2)+1; i++)
          for (int j= 2*(info.index[1]/2); j <= 2*(info.index[1]/2)+1; j++)
          for (int k= 2*(info.index[2]/2); k <= 2*(info.index[2]/2)+1; k++)
          {
            int n = TMeshAdaptation::m_refGrid->getZforward(m,i,j,k);
            BlockInfo & infoNei = TMeshAdaptation::m_refGrid->getBlockInfoAll(m,n);
            if (infoNei.TreePos != Exists || infoNei.state != Compress )
            {
              info.state = Leave;
              break;
            }                       
          }
      }
      
    TMeshAdaptation::m_refGrid->FillPos();
    TMeshAdaptation::m_refGrid->UpdateBlockInfoAll_States();


      //4.
      for ( auto & b: I) if (b.level == m)
      {
        BlockInfo & info =  TMeshAdaptation::m_refGrid->getBlockInfoAll(m,b.Z);
        int nBlock = TMeshAdaptation::m_refGrid->getZforward(m, 2*(info.index[0]/2),2*(info.index[1]/2),2*(info.index[2]/2) ); 
        if (b.Z != nBlock)
        {
          if (info.state==Compress)  info.state = Leave;  
        }
      } 

    TMeshAdaptation::m_refGrid->FillPos();
    TMeshAdaptation::m_refGrid->UpdateBlockInfoAll_States();


    }//m
  }




};
        

}//namespace AMR_CUBISM
