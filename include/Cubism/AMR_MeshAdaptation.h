#pragma once
#include "Matrix3D.h"
#include "BlockInfo.h"
#include "Grid.h"
#include "BlockLab.h"

#include <omp.h>
#include <cstring>
#include <string>
#include <algorithm>


#ifdef __bgq__
#include <builtins.h>
#define memcpy2(a,b,c)  __bcopy((b),(a),(c))
#else
#define memcpy2(a,b,c)  memcpy((a),(b),(c))
#endif


namespace cubism
{
template<typename TGrid,  typename TLab>
class MeshAdaptation
{

public:
  typedef TGrid GridType;
  typedef TLab LabType;
  typedef typename TGrid::Block BlockType;   
  typedef typename TGrid::Block::ElementType ElementType;

protected:
  TGrid * m_refGrid;
  int s[3];  
  int e[3];
  bool istensorial;
  int Is[3];  
  int Ie[3];
  double tolerance_for_refinement;
  double tolerance_for_compression;
  int nthreads;
  TLab * LabOMP;
  std::vector<int> components;
  //TLab Lab;

public:

  MeshAdaptation(TGrid & grid, double Rtol,double Ctol)
  {
    bool tensorial = true;
    const int Gx = 1;
    const int Gy = 1;
    const int Gz = 1;

    components.push_back(0); 
    components.push_back(1); 
    components.push_back(2); 
    components.push_back(3); 
    components.push_back(4); 
    components.push_back(5); //use energy as criterion
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

    //Lab.prepare((*m_refGrid),s,e,istensorial,Is,Ie);

    nthreads = omp_get_max_threads();
    
    LabOMP = new TLab[nthreads];
    for (int i=0; i<nthreads; i++)
      LabOMP[i].prepare((*m_refGrid),s,e,istensorial,Is,Ie);
  }


///////////////////////////////////////////////////////

  ~MeshAdaptation()
  {
    delete[] LabOMP;
  }


  virtual void AdaptTheMesh(double t = 0, bool reload = true)
  {
    if (reload)
    for (int i=0; i<nthreads; i++)
      LabOMP[i].prepare((*m_refGrid),s,e,istensorial,Is,Ie);


    TagBlocks();           
            
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
           
    //Does not work for some reason #pragma omp parallel for 
    for (size_t i=0; i<mn_com.size()/2; i++)
    {
      int m = mn_com[2*i];
      int n = mn_com[2*i+1];               
      compress(m,n);
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



    m_refGrid->FillPos();
    m_refGrid->UpdateBlockInfoAll();
  }



protected:
  
  virtual void refine(int level, int Z)
  {
    int tid = omp_get_thread_num();
      
    BlockInfo & parent =  m_refGrid->getBlockInfoAll(level,Z);

    assert(parent.ptrBlock != NULL);
    assert(level <= m_refGrid->getlevelMax()-1);

    LabOMP[tid].load(parent);
    LabOMP[tid].post_load(parent);  
      
    parent.TreePos = CheckFiner;
    int nChild = m_refGrid->getZchild(level,parent.index[0],parent.index[1],parent.index[2]);


    int p[3] = {parent.index[0],parent.index[1],parent.index[2]};
    BlockType * Blocks [8];
        
    for (int k=0; k<2; k++ )
    for (int j=0; j<2; j++ )
    for (int i=0; i<2; i++ )
    {      
      int nc = m_refGrid->getZforward(level+1,2*p[0]+i,2*p[1]+j,2*p[2]+k);
             
      BlockInfo & Child = m_refGrid->getBlockInfoAll(level+1,nc); 

      Child.TreePos = Exists;
      Child.myrank = m_refGrid->rank();

      if (nc !=nChild)
      {
        assert(!(i==0 && j==0 && k==0));
        #pragma omp critical
        {
        	m_refGrid->_alloc(level+1,nc);
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

    RefineBlocks(Blocks);  
    parent.myrank = -1;
  }




  virtual void compress(int level, int Z)
  {
    //assert (Z%8 ==0);
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
      (m_refGrid->getBlockInfoAll(level,n)).myrank = -1;     
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

    int np = m_refGrid->getZforward(level-1,info.index_(0)/2,info.index_(1)/2,info.index_(2)/2);           
    BlockInfo & parent = m_refGrid->getBlockInfoAll(level-1,np);
    parent.myrank =m_refGrid->rank();

    #pragma omp critical
    {
        for (int K=0; K<2; K++ )
        for (int J=0; J<2; J++ )
        for (int I=0; I<2; I++ )
        {
          if (I + J + K == 0 ) continue;
          int n = m_refGrid->getZforward(level,info.index[0]+I,info.index[1]+J,info.index[2]+K); 
          m_refGrid->_dealloc(level,n);
        }
    }
    parent.ptrBlock = info.ptrBlock;
    parent.TreePos = Exists; 
    parent.h_gridpoint = parent.h; 
    

    for (int K=0; K<2; K++ )
    for (int J=0; J<2; J++ )
    for (int I=0; I<2; I++ )
    {
      int n = m_refGrid->getZforward(level,info.index[0]+I,info.index[1]+J,info.index[2]+K); 
      m_refGrid->getBlockInfoAll(level,n).TreePos = CheckCoarser;
      m_refGrid->getBlockInfoAll(level,n).ptrBlock = NULL;
    }
  }

    

  virtual void WENOWavelets3(double cm, double c , double cp, double & left, double & right)
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


  virtual void Kernel_1D(ElementType E0,ElementType E1,ElementType E2, ElementType & left, ElementType & right)
  {
  	left .dummy = E1.dummy - 0.125*(E2.dummy-E0.dummy);
  	right.dummy = E1.dummy + 0.125*(E2.dummy-E0.dummy);
  	WENOWavelets3(E0.alpha1rho1,E1.alpha1rho1,E2.alpha1rho1,left.alpha1rho1,right.alpha1rho1);
  	WENOWavelets3(E0.alpha2rho2,E1.alpha2rho2,E2.alpha2rho2,left.alpha2rho2,right.alpha2rho2);
  	WENOWavelets3(E0.ru        ,E1.ru        ,E2.ru        ,left.ru        ,right.ru        );
  	WENOWavelets3(E0.rv        ,E1.rv        ,E2.rv        ,left.rv        ,right.rv        );
  	WENOWavelets3(E0.rw        ,E1.rw        ,E2.rw        ,left.rw        ,right.rw        );
  	WENOWavelets3(E0.alpha2    ,E1.alpha2    ,E2.alpha2    ,left.alpha2    ,right.alpha2    );
  	WENOWavelets3(E0.energy    ,E1.energy    ,E2.energy    ,left.energy    ,right.energy    );
  }
        				 



  virtual void RefineBlocks(BlockType * B[8])
  {
    int tid = omp_get_thread_num();
      
    const int nx = BlockType::sizeX;
    const int ny = BlockType::sizeY;
    const int nz = BlockType::sizeZ;
      
    int offsetX [2] = {0,nx/2};
    int offsetY [2] = {0,ny/2};
    int offsetZ [2] = {0,nz/2};

    TLab & Lab = LabOMP[tid];


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
          //        		El[i0+Nweno/2][i1+Nweno/2][i2+Nweno/2] = LabOMP[tid](i/2+offsetX[I] + i0,j/2+offsetY[J]+ i1 ,k/2+offsetZ[K]+ i2);      
          El[i0+Nweno/2][i1+Nweno/2][i2+Nweno/2] = Lab(i/2+offsetX[I] + i0,j/2+offsetY[J]+ i1 ,k/2+offsetZ[K]+ i2);      


        	ElementType Lines [Nweno][Nweno][2];
    			ElementType Planes[Nweno][4];
    			ElementType Ref          [8]; 


			for (int i2= -Nweno/2 ; i2<= Nweno/2; i2++)
			{
				for (int i1= -Nweno/2 ; i1<= Nweno/2; i1++)
        		{
        			Kernel_1D(El[0][i1+Nweno/2][i2+Nweno/2],
        				      El[1][i1+Nweno/2][i2+Nweno/2],
        				      El[2][i1+Nweno/2][i2+Nweno/2],
        				      Lines[i1+Nweno/2][i2+Nweno/2][0],
        				      Lines[i1+Nweno/2][i2+Nweno/2][1]);
        		}	
			}

    	




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

    		

 			Kernel_1D(Planes[0][0],
      		          Planes[1][0],
       			      Planes[2][0],
       			      Ref[0],
       			      Ref[1]);
      
	        Kernel_1D(Planes[0][1],
      		          Planes[1][1],
       			      Planes[2][1],
       			      Ref[2],
       			      Ref[3]);
      
         	Kernel_1D(Planes[0][2],
      		          Planes[1][2],
       			      Planes[2][2],
       			      Ref[4],
       			      Ref[5]);
         
         	Kernel_1D(Planes[0][3],
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

            assert  (!isnan(b(i,j,k).alpha1rho1));
            assert  (!isnan(b(i,j,k).alpha2rho2));
            assert  (!isnan(b(i,j,k).ru        ));
            assert  (!isnan(b(i,j,k).rv        ));
            assert  (!isnan(b(i,j,k).rw        ));
            assert  (!isnan(b(i,j,k).alpha2    ));
            assert  (!isnan(b(i,j,k).energy    ));
            assert  (!isnan(b(i,j,k).dummy     ));

            assert  (abs(b(i,j,k).alpha1rho1) < 1e20 );
            assert  (abs(b(i,j,k).alpha2rho2) < 1e20 );
            assert  (abs(b(i,j,k).ru        ) < 1e20 );
            assert  (abs(b(i,j,k).rv        ) < 1e20 );
            assert  (abs(b(i,j,k).rw        ) < 1e20 );
            assert  (abs(b(i,j,k).alpha2    ) < 1e20 );
            assert  (abs(b(i,j,k).energy    ) < 1e20 );
            assert  (abs(b(i,j,k).dummy     ) < 1e20 );
        }         
      }
  }



  virtual void TagBlocks()
  {
    std::vector <BlockInfo> & B = m_refGrid->getBlocksInfo();          
  
    #pragma omp parallel num_threads(nthreads)
    {
      int tid = omp_get_thread_num();
      TLab& mylab = LabOMP[tid];
 
      #pragma omp for schedule(dynamic,1)
      for(size_t i=0; i<B.size(); i++)
      {
        mylab.load(B[i]);    
        BlockInfo & info = m_refGrid->getBlockInfoAll(B[i].level,B[i].Z);
        B[i].state = TagLoadedBlock(LabOMP[tid]);
        info.state = B[i].state;        	
      }
    }

    ValidStates();      
  }

  


  //Tag block loaded in Lab for refinement/compression; user can write own function
  virtual State TagLoadedBlock(TLab & Lab_)
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

    if      (Linf > tolerance_for_refinement) return Refine;
    else if (Linf < tolerance_for_compression) return Compress;
    else return Leave;
  }







  virtual void ValidStates()
  {
    std::vector <BlockInfo> & I = m_refGrid->getBlocksInfo();
    std::array <int,3> blocksPerDim = m_refGrid->getMaxBlocks();

    int levelMin = 0;
    int levelMax = m_refGrid->getlevelMax();

    for ( auto & b: I)
    {
        BlockInfo & info =  m_refGrid->getBlockInfoAll(b.level,b.Z);
        if (info.state==Refine   && info.level ==levelMax-1) info.state=Leave;
        if (info.state==Compress && info.level ==levelMin  ) info.state=Leave;
    }

    const bool xperiodic = LabOMP[0].is_xperiodic();
    const bool yperiodic = LabOMP[0].is_yperiodic();
    const bool zperiodic = LabOMP[0].is_zperiodic();

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

          BlockInfo & infoNei = m_refGrid->getBlockInfoAll(info.level_(),info.Znei_(code[0],code[1],code[2]) );
            
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
              int zzz = m_refGrid->getZforward(m+1,iNei,jNei,kNei);
              BlockInfo & FinerNei = m_refGrid->getBlockInfoAll(m+1,zzz);
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
  
      //2.
      for ( auto & b: I) if (b.level == m)
      {
        BlockInfo & info =  m_refGrid->getBlockInfoAll(m,b.Z);
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
          BlockInfo & infoNei = m_refGrid->getBlockInfoAll(info.level_(),info.Znei_(code[0],code[1],code[2]) );
          
          if (infoNei.TreePos == Exists && infoNei.state==Refine && info.state==Compress)
            info.state=Leave;
        }
      }


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
              break;
            }                       
          }
      }
      


      //4.
      for ( auto & b: I) if (b.level == m)
      {
        BlockInfo & info =  m_refGrid->getBlockInfoAll(m,b.Z);
        int nBlock = m_refGrid->getZforward(m, 2*(info.index[0]/2),2*(info.index[1]/2),2*(info.index[2]/2) ); 
        if (b.Z != nBlock)
        {
          if (info.state==Compress)  info.state = Leave;  
        }
      } 

    }//m
  }

};


        

}//namespace AMR_CUBISM
