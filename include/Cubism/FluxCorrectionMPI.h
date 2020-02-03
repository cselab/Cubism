#pragma once

#include "FluxCorrection.h"
//#include <omp.h>

#ifdef __bgq__
#include <builtins.h>
#define memcpy2(a,b,c)  __bcopy((b),(a),(c))
#else
#define memcpy2(a,b,c)  memcpy((a),(b),(c))
#endif


namespace cubism //AMR_CUBISM
{


template< typename TFluxCorrection , typename TGrid>
class FluxCorrectionMPI: public TFluxCorrection
{
  public:
    typedef typename TFluxCorrection::ElementType ElementType;
    typedef typename TFluxCorrection::Real Real;
    typedef typename TFluxCorrection::BlockType BlockType;
    typedef typename TFluxCorrection::ElementTypeBlock ElementTypeBlock;
    typedef BlockCase <BlockType, allocator> Case;
 
  protected:

    struct face
    {
      BlockInfo * infos [2];
      int icode [2];
      int offset;
      //infos[0] : Fine block
      //infos[1] : Coarse block
      face (BlockInfo & i0,BlockInfo & i1, int a_icode0, int a_icode1)
      {
        infos[0] = &i0;
        infos[1] = &i1;
        icode[0] = a_icode0;
        icode[1] = a_icode1;
      }     
      bool operator<(const face & other) const 
      {
        if (infos[0]->Z == other.infos[0]->Z) 
        {
          return (icode[0] < other.icode[0]);
        }
        else
        {
          return (infos[0]->Z < other.infos[0]->Z);
        }      
      }
    };

    int rank,size;
    std::vector< std::vector<Real> > send_buffer;
    std::vector< std::vector<Real> > recv_buffer;
    std::vector<std::vector<face>> send_faces;
    std::vector<std::vector<face>> recv_faces;


  public:

    virtual void prepare(TGrid & grid) override
    {
      MPI_Comm_size(MPI_COMM_WORLD,&size);
      MPI_Comm_rank(MPI_COMM_WORLD,&rank);

      send_buffer.resize(size);
      recv_buffer.resize(size);
      send_faces.resize(size);
      recv_faces.resize(size);


      std::vector<int> send_buffer_size(size,0);
      std::vector<int> recv_buffer_size(size,0);


      TFluxCorrection::prepare(grid);

      std::vector<BlockInfo> & BLOCKS = (*TFluxCorrection::m_refGrid).getBlocksInfo();
      
      const int NC = 8;
      
      int blocksize[3];
      blocksize[0]=BlockType::sizeX;
      blocksize[1]=BlockType::sizeY;
      blocksize[2]=BlockType::sizeZ;

      //1.Define faces
      for (auto & info: BLOCKS)
      {
        int aux = pow(2,info.level);

        const bool xskin = info.index[0]==0 || info.index[0]==TFluxCorrection::blocksPerDim[0]*aux-1;
        const bool yskin = info.index[1]==0 || info.index[1]==TFluxCorrection::blocksPerDim[1]*aux-1;
        const bool zskin = info.index[2]==0 || info.index[2]==TFluxCorrection::blocksPerDim[2]*aux-1;

        const int xskip  = info.index[0]==0 ? -1 : 1;
        const int yskip  = info.index[1]==0 ? -1 : 1;
        const int zskip  = info.index[2]==0 ? -1 : 1;
       
        for(int icode=0; icode<27; icode++)
        {
          if (icode == 1*1 + 3*1 + 9*1) continue;
          
          const int code[3] = { icode%3-1, (icode/3)%3-1, (icode/9)%3-1};
    
          if (abs(code[0])+abs(code[1])+abs(code[2])>1) continue;
               
          BlockInfo infoNei = (*TFluxCorrection::m_refGrid).getBlockInfoAll(info.level,info.Znei_(code[0],code[1],code[2]));

          if (!TFluxCorrection::xperiodic && code[0] == xskip && xskin) continue;
          if (!TFluxCorrection::yperiodic && code[1] == yskip && yskin) continue;
          if (!TFluxCorrection::zperiodic && code[2] == zskip && zskin) continue; 


          int L[3];
          L[0] = (code[0] == 0) ? blocksize[0]/2 : 1;
          L[1] = (code[1] == 0) ? blocksize[1]/2 : 1;
          L[2] = (code[2] == 0) ? blocksize[2]/2 : 1;
          int V = L[0]*L[1]*L[2]; 

          if (infoNei.TreePos == CheckCoarser)
          {
            int nCoarse = (*TFluxCorrection::m_refGrid).getZforward(infoNei.level-1,infoNei.index[0]/2,infoNei.index[1]/2,infoNei.index[2]/2);
            BlockInfo & infoNeiCoarser = (*TFluxCorrection::m_refGrid).getBlockInfoAll(infoNei.level-1,nCoarse);
            if (infoNeiCoarser.myrank != rank)
            {
              int code2[3] = {-code[0],-code[1],-code[2]};
              int icode2 = (code2[0]+1) + (code2[1]+1)*3 + (code2[2]+1)*9;
              send_faces[infoNeiCoarser.myrank].push_back( face(info,infoNeiCoarser,icode,icode2) );            
              send_buffer_size[infoNeiCoarser.myrank] += V;
            }
          }
          else if (infoNei.TreePos == CheckFiner)
          {
            int Bstep = 1; //face
            for (int B = 0 ; B <= 3 ; B += Bstep) //loop over blocks that make up face
            {
              const int temp = (abs(code[0])==1) ? (B%2) : (B/2) ;
              int nFine = (*TFluxCorrection::m_refGrid).getZforward(infoNei.level+1,2*info.index[0] + max(code[0],0) +code[0]  + (B%2)*max(0, 1 - abs(code[0])),
                                                      2*info.index[1] + max(code[1],0) +code[1]  + temp *max(0, 1 - abs(code[1])),
                                                      2*info.index[2] + max(code[2],0) +code[2]  + (B/2)*max(0, 1 - abs(code[2])));
              BlockInfo & infoNeiFiner = (*TFluxCorrection::m_refGrid).getBlockInfoAll(infoNei.level+1,nFine);
              if (infoNeiFiner.myrank != rank)
              {
                int icode2 = (-code[0]+1) + (-code[1]+1)*3 + (-code[2]+1)*9;
                recv_faces[infoNeiFiner.myrank].push_back( face(infoNeiFiner,info,icode2,icode) );
                recv_buffer_size[infoNeiFiner.myrank] += V;
              }
            }
          }
        }//icode = 0,...,26
      }


      //2.Sort faces 
      for (int r=0; r<size; r++)
      {
        std::sort (send_faces[r].begin(), send_faces[r].end());
        std::sort (recv_faces[r].begin(), recv_faces[r].end());
      }

     
      //3.Define map
      for (int r=0; r<size; r++)
      {
        send_buffer[r].resize(send_buffer_size[r]*NC);
        recv_buffer[r].resize(recv_buffer_size[r]*NC);

        int offset = 0;
        for (int k = 0; k < (int)recv_faces[r].size(); k++)
        {
          face & f = recv_faces[r][k];

          const int code[3] = { f.icode[1]%3-1, (f.icode[1]/3)%3-1, (f.icode[1]/9)%3-1};

          int L[3];
          L[0] = (code[0] == 0) ? blocksize[0]/2 : 1;
          L[1] = (code[1] == 0) ? blocksize[1]/2 : 1;
          L[2] = (code[2] == 0) ? blocksize[2]/2 : 1;
          int V=L[0]*L[1]*L[2];

          f.offset = offset;

          offset += V*NC;        
        }
      }
    }


    virtual void FillBlockCases() override
    {
      //This assumes that the BlockCases have been filled by the user somehow... 
      std::vector<BlockInfo> & B = (*TFluxCorrection::m_refGrid).getBlocksInfo();
         
    
      for (auto & info: B)
      {
        int aux = pow(2,info.level);

        const bool xskin = info.index[0]==0 || info.index[0]==TFluxCorrection::blocksPerDim[0]*aux-1;
        const bool yskin = info.index[1]==0 || info.index[1]==TFluxCorrection::blocksPerDim[1]*aux-1;
        const bool zskin = info.index[2]==0 || info.index[2]==TFluxCorrection::blocksPerDim[2]*aux-1;

        const int xskip  = info.index[0]==0 ? -1 : 1;
        const int yskip  = info.index[1]==0 ? -1 : 1;
        const int zskip  = info.index[2]==0 ? -1 : 1;
              
        for(int icode=0; icode<27; icode++)
        {
          if (icode == 1*1 + 3*1 + 9*1) continue;
          const int code[3] = { icode%3-1, (icode/3)%3-1, (icode/9)%3-1};
    
          if (abs(code[0])+abs(code[1])+abs(code[2])>1) continue;
     
          BlockInfo infoNei = (*TFluxCorrection::m_refGrid).getBlockInfoAll(info.level,info.Znei_(code[0],code[1],code[2]));

          if (!TFluxCorrection::xperiodic && code[0] == xskip && xskin) continue;
          if (!TFluxCorrection::yperiodic && code[1] == yskip && yskin) continue;
          if (!TFluxCorrection::zperiodic && code[2] == zskip && zskin) continue; 

          if (infoNei.TreePos == CheckFiner)
          {
            FillCase(info,code);
          }        
        }//icode = 0,...,26       
      }

      //1.Pack send data
      for (int r=0; r<size; r++)
      {

        int displacement = 0;
        for (int k=0; k<(int)send_faces[r].size(); k++)
        {
          face & f = send_faces[r][k];

          BlockInfo & info = *(f.infos[0]); 

          auto search = TFluxCorrection::MapOfCases.find({info.level,info.Z});
          assert(  search != TFluxCorrection::MapOfCases.end());

          Case & FineCase = (*search->second);
        

          int icode = f.icode[0];
          const int code[3] = { icode%3-1, (icode/3)%3-1, (icode/9)%3-1};
          int myFace    = abs( code[0]) * max(0, code[0]) + abs( code[1]) * (max(0, code[1])+2) + abs( code[2]) * (max(0, code[2])+4);
          std::vector< ElementType > & FineFace = FineCase.m_pData[myFace];


          int d   = myFace / 2 ;       
          int d1  = max((d+1)%3,(d+2)%3);
          int d2  = min((d+1)%3,(d+2)%3);
          int N1F = FineCase.m_vSize[d1];
          int N2F = FineCase.m_vSize[d2];        
          int N1 = N1F;
          int N2 = N2F; 


          assert (N1==N2);
          assert (N1==_BLOCKSIZE_);    
     
          for (int i1 = 0 ; i1 < N1; i1 +=2)
          for (int i2 = 0 ; i2 < N2; i2 +=2)
          {
            ElementType avg = 0.25*((FineFace[i2+i1*N2]+FineFace[i2+1+i1*N2])+(FineFace[i2+(i1+1)*N2]+FineFace[i2+1+(i1+1)*N2])); 
          
            send_buffer[r][displacement  ] = avg.alpha1rho1;
            send_buffer[r][displacement+1] = avg.alpha2rho2;
            send_buffer[r][displacement+2] = avg.ru;        
            send_buffer[r][displacement+3] = avg.rv;        
            send_buffer[r][displacement+4] = avg.rw;        
            send_buffer[r][displacement+5] = avg.energy;    
            send_buffer[r][displacement+6] = avg.alpha2;    
            send_buffer[r][displacement+7] = avg.dummy;     
            displacement += 8;
          } 
        }
      }


      std::vector <MPI_Request> send_requests(size);
      std::vector <MPI_Request> recv_requests(size);
        
      for (int r = 0 ; r < size; r ++ )
      {
          MPI_Irecv(&recv_buffer[r][0], recv_buffer[r].size(), MPI_DOUBLE, r, 123456 , MPI_COMM_WORLD, &recv_requests[r]);
          MPI_Isend(&send_buffer[r][0], send_buffer[r].size(), MPI_DOUBLE, r, 123456 , MPI_COMM_WORLD, &send_requests[r]);
      }
      MPI_Waitall(size, &recv_requests[0], MPI_STATUSES_IGNORE);
      MPI_Waitall(size, &send_requests[0], MPI_STATUSES_IGNORE);


      for (int r = 0 ; r < size; r ++ )
      {
        for (int index = 0; index < (int)recv_faces[r].size(); index++)
        {
          face & f = recv_faces[r][index];
          FillCase_2(f) ;
        }
      }

      TFluxCorrection::Correct();
    }



    void FillCase_2(face F)
    {


      BlockInfo info = *F.infos[1];
      int icode = F.icode[1];
      const int code[3] = { icode%3-1, (icode/3)%3-1, (icode/9)%3-1};



      int myFace    = abs( code[0]) * max(0, code[0]) + abs( code[1]) * (max(0, code[1])+2) + abs( code[2]) * (max(0, code[2])+4);
      

      std::array <int,2> temp = {info.level,info.Z};           
      auto search = TFluxCorrection::MapOfCases.find(temp);


      assert(search != TFluxCorrection::MapOfCases.end());


      Case & CoarseCase = (*search->second); 
      std::vector< ElementType > & CoarseFace = CoarseCase.m_pData[myFace];


      for (int B = 0 ; B <= 3 ; B ++) //loop over fine blocks that make up coarse face
      {
        const int aux = (abs(code[0])==1) ? (B%2) : (B/2) ;

        int Z = (*TFluxCorrection::m_refGrid).getZforward(info.level+1, 2*info.index[0] + max(code[0],0) +code[0]  + (B%2)*max(0, 1 - abs(code[0])),
                                                                        2*info.index[1] + max(code[1],0) +code[1]  +  aux *max(0, 1 - abs(code[1])),
                                                                        2*info.index[2] + max(code[2],0) +code[2]  + (B/2)*max(0, 1 - abs(code[2])));
        
        if ( Z != F.infos[0]->Z ) continue;

     
        int d   = myFace / 2 ;       
        int d1  = max((d+1)%3,(d+2)%3);
        int d2  = min((d+1)%3,(d+2)%3);
        int N1F = CoarseCase.m_vSize[d1];
        int N2F = CoarseCase.m_vSize[d2];        
        int N1 = N1F;
        int N2 = N2F;


        int base = 0 ; //(B%2)*(N1/2)+ (B/2)*(N2/2)*N1;
        if      (B==1) base = (N2/2)+ (0   )*N2; 
        else if (B==2) base = (0   )+ (N1/2)*N2; 
        else if (B==3) base = (N2/2)+ (N1/2)*N2;

      
       
        double coef = 1.0 / info.h; 

        int r =F.infos[0]->myrank;
        int dis = 0;
        for (int i1 = 0 ; i1 < N1; i1 +=2)
        for (int i2 = 0 ; i2 < N2; i2 +=2)
        {
          CoarseFace[base + (i2/2)+ (i1/2)   *N2] *= coef;  

          CoarseFace[base + (i2/2)+ (i1/2)   *N2].alpha1rho1 += (coef)* recv_buffer[r][F.offset + dis  ];
          CoarseFace[base + (i2/2)+ (i1/2)   *N2].alpha2rho2 += (coef)* recv_buffer[r][F.offset + dis+1];
          CoarseFace[base + (i2/2)+ (i1/2)   *N2].ru         += (coef)* recv_buffer[r][F.offset + dis+2];
          CoarseFace[base + (i2/2)+ (i1/2)   *N2].rv         += (coef)* recv_buffer[r][F.offset + dis+3];
          CoarseFace[base + (i2/2)+ (i1/2)   *N2].rw         += (coef)* recv_buffer[r][F.offset + dis+4];
          CoarseFace[base + (i2/2)+ (i1/2)   *N2].energy     += (coef)* recv_buffer[r][F.offset + dis+5];
          CoarseFace[base + (i2/2)+ (i1/2)   *N2].alpha2     += (coef)* recv_buffer[r][F.offset + dis+6];
          CoarseFace[base + (i2/2)+ (i1/2)   *N2].dummy      += (coef)* recv_buffer[r][F.offset + dis+7];
          dis += 8; 
        }
      }
    }

    virtual void FillCase(BlockInfo info, const int * const code) override
    {
      int myFace    = abs( code[0]) * max(0, code[0]) + abs( code[1]) * (max(0, code[1])+2) + abs( code[2]) * (max(0, code[2])+4);
      int otherFace = abs(-code[0]) * max(0,-code[0]) + abs(-code[1]) * (max(0,-code[1])+2) + abs(-code[2]) * (max(0,-code[2])+4);
    

      std::array <int,2> temp = {info.level,info.Z};           
      auto search = TFluxCorrection::MapOfCases.find(temp);

      assert(myFace / 2 == otherFace / 2);
      assert(search != TFluxCorrection::MapOfCases.end());

      Case & CoarseCase = (*search->second);
  
      assert (CoarseCase.Z     == info.Z    );
      assert (CoarseCase.level == info.level);

      std::vector< ElementType > & CoarseFace = CoarseCase.m_pData[myFace];


      for (int B = 0 ; B <= 3 ; B ++) //loop over fine blocks that make up coarse face
      {
        const int aux = (abs(code[0])==1) ? (B%2) : (B/2) ;

        int Z = (*TFluxCorrection::m_refGrid).getZforward(info.level+1, 2*info.index[0] + max(code[0],0) +code[0]  + (B%2)*max(0, 1 - abs(code[0])),
                                                       2*info.index[1] + max(code[1],0) +code[1]  +  aux *max(0, 1 - abs(code[1])),
                                                       2*info.index[2] + max(code[2],0) +code[2]  + (B/2)*max(0, 1 - abs(code[2])));
        
        if ( (*TFluxCorrection::m_refGrid).getBlockInfoAll(info.level+1,Z).myrank != rank ) continue;

        auto search1 = TFluxCorrection::MapOfCases.find({info.level+1,Z});
        assert(  search1 != TFluxCorrection::MapOfCases.end());

        Case & FineCase = (*search1->second);
        std::vector< ElementType > & FineFace = FineCase.m_pData[otherFace];

        int d   = myFace / 2 ;       
        int d1  = max((d+1)%3,(d+2)%3);
        int d2  = min((d+1)%3,(d+2)%3);
        int N1F = FineCase.m_vSize[d1];
        int N2F = FineCase.m_vSize[d2];        
        int N1 = N1F;
        int N2 = N2F;

        assert(N1F == (int)CoarseCase.m_vSize[d1]);
        assert(N2F == (int)CoarseCase.m_vSize[d2]);


        int base = 0 ; //(B%2)*(N1/2)+ (B/2)*(N2/2)*N1;
        if      (B==1) base = (N2/2)+ (0   )*N2; 
        else if (B==2) base = (0   )+ (N1/2)*N2; 
        else if (B==3) base = (N2/2)+ (N1/2)*N2;

        assert(FineFace.size() == CoarseFace.size());

       
        double coef = 1.0 / info.h; 

        for (int i1 = 0 ; i1 < N1; i1 +=2)
        for (int i2 = 0 ; i2 < N2; i2 +=2)
        {
          CoarseFace[base + (i2/2)+ (i1/2)   *N2] *= coef;  
          CoarseFace[base + (i2/2)+ (i1/2)   *N2] += (0.25*coef)*((FineFace[i2+i1*N2]+FineFace[i2+1+i1*N2])+(FineFace[i2+(i1+1)*N2]+FineFace[i2+1+(i1+1)*N2])); 
        }
      }
    }
};



}//namespace AMR_CUBISM
