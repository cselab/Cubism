#pragma once


#include "Grid.h"
#include <map>
//#include <omp.h>



#ifdef __bgq__
#include <builtins.h>
#define memcpy2(a,b,c)  __bcopy((b),(a),(c))
#else
#define memcpy2(a,b,c)  memcpy((a),(b),(c))
#endif




namespace cubism //AMR_CUBISM
{



template<typename TBlock, template<typename X> class allocator = std::allocator, typename ElementTypeT = typename TBlock::ElementType>
class BlockCase
{

public:
    typedef ElementTypeT ElementType;
    typedef TBlock BlockType;

    std::vector < std::vector< ElementType>  > m_pData;
    unsigned int m_vSize[3]; 
    bool storedFace[6];
    int level,Z;
   
    BlockCase(bool a_storedFace[6], unsigned int nSizeX, unsigned int nSizeY, unsigned int nSizeZ)
    {
      m_vSize[0] = nSizeX;
      m_vSize[1] = nSizeY;
      m_vSize[2] = nSizeZ;

      storedFace[0] = a_storedFace[0];
      storedFace[1] = a_storedFace[1];
      storedFace[2] = a_storedFace[2];
      storedFace[3] = a_storedFace[3];
      storedFace[4] = a_storedFace[4];
      storedFace[5] = a_storedFace[5];

      m_pData.resize(6); 

      for (int d=0; d<3; d++)
      {
        int d1 = (d+1)%3;
        int d2 = (d+2)%3; 
        //assume everything is initialized to 0!!!!!!!!
        if (storedFace[2*d  ]) m_pData[2*d].resize(m_vSize[d1]*m_vSize[d2]);
        if (storedFace[2*d+1])m_pData[2*d+1].resize(m_vSize[d1]*m_vSize[d2]);
      }
    }

    void SetupMetaData(int m, int n)
    {
      level = m;
      Z = n;
    }

    int GetLevel()
    {
      return level;
    }
    int GetZ()
    {
      return Z;
    }

    ~BlockCase(){}
};





template<typename TBlock, typename TLab, template<typename X> class allocator = std::allocator,typename ElementTypeT = typename TBlock::ElementType>
class FluxCorrection
{
  public:
    typedef ElementTypeT ElementType;
    typedef typename ElementTypeT::RealType Real;
    typedef TBlock BlockType;
    typedef typename BlockType::ElementType ElementTypeBlock;
    typedef BlockCase <BlockType, allocator> Case;
 
  protected:
    std::map<std::array<int,2>, Case * > MapOfCases;
    Grid<BlockType, allocator>* m_refGrid;
    std::vector < Case > Cases;
    TLab temp_Lab; //needed only to call functions is_xperiodic(),is_yperiodic(),is_zperiodic()


  public:
    void prepare(Grid<BlockType,allocator>& grid)
    {
      Cases.clear();
      MapOfCases.clear();

      m_refGrid = &grid;
      std::vector<BlockInfo> & B = (*m_refGrid).getBlocksInfo();

      std::array <int,3> blocksPerDim = (*m_refGrid).getMaxBlocks();

      const bool xperiodic = temp_Lab.is_xperiodic(); 
      const bool yperiodic = temp_Lab.is_yperiodic();
      const bool zperiodic = temp_Lab.is_zperiodic();
         
      for (auto & info: B)
      {
        int aux = pow(2,info.level);

        const bool xskin = info.index[0]==0 || info.index[0]==blocksPerDim[0]*aux-1;
        const bool yskin = info.index[1]==0 || info.index[1]==blocksPerDim[1]*aux-1;
        const bool zskin = info.index[2]==0 || info.index[2]==blocksPerDim[2]*aux-1;

        const int xskip  = info.index[0]==0 ? -1 : 1;
        const int yskip  = info.index[1]==0 ? -1 : 1;
        const int zskip  = info.index[2]==0 ? -1 : 1;
       
        bool storeFace[6] = {false,false,false,false,false,false};
        bool stored = false;

        for(int icode=0; icode<27; icode++)
        {
          if (icode == 1*1 + 3*1 + 9*1) continue;
          
          const int code[3] = { icode%3-1, (icode/3)%3-1, (icode/9)%3-1};
    
          if (abs(code[0])+abs(code[1])+abs(code[2])>1) continue;
     
          BlockInfo infoNei = (*m_refGrid).getBlockInfoAll(info.level,info.Znei_(code[0],code[1],code[2]));

          if (!xperiodic && code[0] == xskip && xskin) continue;
          if (!yperiodic && code[1] == yskip && yskin) continue;
          if (!zperiodic && code[2] == zskip && zskin) continue; 

          if (infoNei.TreePos != Exists)
          {
            storeFace[ abs(code[0]) * max(0,code[0]) + abs(code[1]) * (max(0,code[1])+2) + abs(code[2]) * (max(0,code[2])+4)   ] = true;
            stored = true;
          }    
        }//icode = 0,...,26       

        if (stored)
        {
          Cases.push_back(Case(storeFace,BlockType::sizeX,BlockType::sizeY,BlockType::sizeZ));
          Cases.back().SetupMetaData(info.level,info.Z);
        }
      }

      for (size_t i = 0; i < Cases.size() ; i ++ )
      {
        MapOfCases.insert(  std::pair<std::array <int,2>,Case *>  (   {Cases[i].level,Cases[i].Z}  , &Cases[i] ) );
      }
    }




    Case * GetCase(int level, int Z)
    {   	   
    	std::array<int,2> tmp = {level,Z}; 

      auto search = MapOfCases.find(tmp);
    
      if (search == MapOfCases.end() )
      {
      	return nullptr;
      } 

		    assert ( (*search->second).level == level );
        assert ( (*search->second).Z     == Z     );
        return (search->second);
    }






    void FillBlockCases()
    {
      //This assumes that the BlockCases have been filled by the user somehow... 
      std::vector<BlockInfo> & B = (*m_refGrid).getBlocksInfo();
      std::array <int,3> blocksPerDim = (*m_refGrid).getMaxBlocks();

      const bool xperiodic = temp_Lab.is_xperiodic(); 
      const bool yperiodic = temp_Lab.is_yperiodic();
      const bool zperiodic = temp_Lab.is_zperiodic();
      
    
      for (auto & info: B)
      {
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
    
          if (abs(code[0])+abs(code[1])+abs(code[2])>1) continue;
     
          BlockInfo infoNei = (*m_refGrid).getBlockInfoAll(info.level,info.Znei_(code[0],code[1],code[2]));

          if (!xperiodic && code[0] == xskip && xskin) continue;
          if (!yperiodic && code[1] == yskip && yskin) continue;
          if (!zperiodic && code[2] == zskip && zskin) continue; 

          if (infoNei.TreePos == CheckFiner)
          {
            FillCase(info,code);
          }        
        }//icode = 0,...,26       
      }


      Correct();
    }







    void FillCase(BlockInfo info, const int * const code)
    {
      int myFace    = abs( code[0]) * max(0, code[0]) + abs( code[1]) * (max(0, code[1])+2) + abs( code[2]) * (max(0, code[2])+4);
      int otherFace = abs(-code[0]) * max(0,-code[0]) + abs(-code[1]) * (max(0,-code[1])+2) + abs(-code[2]) * (max(0,-code[2])+4);
    

      std::array <int,2> temp = {info.level,info.Z};           
      auto search = MapOfCases.find(temp);

      assert(myFace / 2 == otherFace / 2);
      assert(search != MapOfCases.end());

      Case & CoarseCase = (*search->second);
  
      assert (CoarseCase.Z     == info.Z    );
      assert (CoarseCase.level == info.level);

      std::vector< ElementType > & CoarseFace = CoarseCase.m_pData[myFace];


      for (int B = 0 ; B <= 3 ; B ++) //loop over fine blocks that make up coarse face
      {
        const int aux = (abs(code[0])==1) ? (B%2) : (B/2) ;

        int Z = (*m_refGrid).getZforward(info.level+1, 2*info.index[0] + max(code[0],0) +code[0]  + (B%2)*max(0, 1 - abs(code[0])),
                                                       2*info.index[1] + max(code[1],0) +code[1]  +  aux *max(0, 1 - abs(code[1])),
                                                       2*info.index[2] + max(code[2],0) +code[2]  + (B/2)*max(0, 1 - abs(code[2])));
        

        auto search1 = MapOfCases.find({info.level+1,Z});
        assert(  search1 != MapOfCases.end());

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





    void Correct()
    {
      //This assumes that the BlockCases have been filled by the user somehow... 
      std::vector<BlockInfo>  B = (*m_refGrid).getBlocksInfo();

      std::array <int,3> blocksPerDim = (*m_refGrid).getMaxBlocks();

      const bool xperiodic = temp_Lab.is_xperiodic(); 
      const bool yperiodic = temp_Lab.is_yperiodic();
      const bool zperiodic = temp_Lab.is_zperiodic();
      
    
      for (auto & info: B)
      {
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
    
          if (abs(code[0])+abs(code[1])+abs(code[2])>1) continue;
     
          BlockInfo infoNei = (*m_refGrid).getBlockInfoAll(info.level,info.Znei_(code[0],code[1],code[2]));

          //mike: this should also check if children/parent of this block belong to the grid (MPI version)
          //if (!(*m_refGrid).avail(info.index[0] + code[0], info.index[1] + code[1], info.index[2] + code[2],info.level)) continue;
    
          if (!xperiodic && code[0] == xskip && xskin) continue;
          if (!yperiodic && code[1] == yskip && yskin) continue;
          if (!zperiodic && code[2] == zskip && zskin) continue; 


          if (infoNei.TreePos == CheckFiner)
          {
            int myFace    = abs( code[0]) * max(0, code[0]) + abs( code[1]) * (max(0, code[1])+2) + abs( code[2]) * (max(0, code[2])+4);
            std::array <int,2> temp = {info.level,info.Z};     
            auto search = MapOfCases.find(temp);
            assert(search != MapOfCases.end());
            Case & CoarseCase = (*search->second);
            std::vector< ElementType > & CoarseFace = CoarseCase.m_pData[myFace];

                  

            int d  = myFace / 2 ;
            int d1 = max((d+1)%3,(d+2)%3);
            int d2 = min((d+1)%3,(d+2)%3);
            int N1 = CoarseCase.m_vSize[d1];
            int N2 = CoarseCase.m_vSize[d2];

            assert (N1 == _BLOCKSIZE_);
            assert (N2 == _BLOCKSIZE_);

            BlockType & block = *(BlockType *)info.ptrBlock;


            //WARNING: tmp indices are tmp[z][y][x][Flow Quantity]!
            if (d==0) 
            {
              if (myFace %2 ==0)
              {
                for (int i1 = 0 ; i1 < N1; i1 +=1)
                for (int i2 = 0 ; i2 < N2; i2 +=1)               
                {

                  block.tmp[i1][i2][0][0] += CoarseFace[i2+i1*N2].alpha1rho1;
                  block.tmp[i1][i2][0][1] += CoarseFace[i2+i1*N2].alpha2rho2;
                  block.tmp[i1][i2][0][2] += CoarseFace[i2+i1*N2].ru;
                  block.tmp[i1][i2][0][3] += CoarseFace[i2+i1*N2].rv;
                  block.tmp[i1][i2][0][4] += CoarseFace[i2+i1*N2].rw;
                  block.tmp[i1][i2][0][5] += CoarseFace[i2+i1*N2].energy;
                  block.tmp[i1][i2][0][6] += CoarseFace[i2+i1*N2].alpha2;
                  block.tmp[i1][i2][0][7] += CoarseFace[i2+i1*N2].dummy;


                }
              }
              else
              {
                for (int i1 = 0 ; i1 < N1; i1 +=1)
                for (int i2 = 0 ; i2 < N2; i2 +=1)
                {
                  block.tmp[i1][i2][TBlock::sizeX-1][0] += CoarseFace[i2+i1*N2].alpha1rho1;
                  block.tmp[i1][i2][TBlock::sizeX-1][1] += CoarseFace[i2+i1*N2].alpha2rho2;
                  block.tmp[i1][i2][TBlock::sizeX-1][2] += CoarseFace[i2+i1*N2].ru;
                  block.tmp[i1][i2][TBlock::sizeX-1][3] += CoarseFace[i2+i1*N2].rv;
                  block.tmp[i1][i2][TBlock::sizeX-1][4] += CoarseFace[i2+i1*N2].rw;
                  block.tmp[i1][i2][TBlock::sizeX-1][5] += CoarseFace[i2+i1*N2].energy;
                  block.tmp[i1][i2][TBlock::sizeX-1][6] += CoarseFace[i2+i1*N2].alpha2;
                  block.tmp[i1][i2][TBlock::sizeX-1][7] += CoarseFace[i2+i1*N2].dummy;
           


                }
              }
            }
            else if (d==1)
            {
              if (myFace %2 ==0)
              {
                for (int i1 = 0 ; i1 < N1; i1 +=1)
                for (int i2 = 0 ; i2 < N2; i2 +=1)
                {
                  block.tmp[i1][0][i2][0] += CoarseFace[i2+i1*N2].alpha1rho1;
                  block.tmp[i1][0][i2][1] += CoarseFace[i2+i1*N2].alpha2rho2;
                  block.tmp[i1][0][i2][2] += CoarseFace[i2+i1*N2].ru;
                  block.tmp[i1][0][i2][3] += CoarseFace[i2+i1*N2].rv;
                  block.tmp[i1][0][i2][4] += CoarseFace[i2+i1*N2].rw;
                  block.tmp[i1][0][i2][5] += CoarseFace[i2+i1*N2].energy;
                  block.tmp[i1][0][i2][6] += CoarseFace[i2+i1*N2].alpha2;
                  block.tmp[i1][0][i2][7] += CoarseFace[i2+i1*N2].dummy;
              	
                }
              }
              else
              {
                for (int i1 = 0 ; i1 < N1; i1 +=1)
                for (int i2 = 0 ; i2 < N2; i2 +=1)
                {
                  block.tmp[i1][TBlock::sizeY-1][i2][0] += CoarseFace[i2+i1*N2].alpha1rho1;
                  block.tmp[i1][TBlock::sizeY-1][i2][1] += CoarseFace[i2+i1*N2].alpha2rho2;
                  block.tmp[i1][TBlock::sizeY-1][i2][2] += CoarseFace[i2+i1*N2].ru;
                  block.tmp[i1][TBlock::sizeY-1][i2][3] += CoarseFace[i2+i1*N2].rv;
                  block.tmp[i1][TBlock::sizeY-1][i2][4] += CoarseFace[i2+i1*N2].rw;
                  block.tmp[i1][TBlock::sizeY-1][i2][5] += CoarseFace[i2+i1*N2].energy;
                  block.tmp[i1][TBlock::sizeY-1][i2][6] += CoarseFace[i2+i1*N2].alpha2;
                  block.tmp[i1][TBlock::sizeY-1][i2][7] += CoarseFace[i2+i1*N2].dummy;
                

                }
              }
            }
            else
            {
              if (myFace %2 ==0)
              {
                for (int i1 = 0 ; i1 < N1; i1 +=1)
                for (int i2 = 0 ; i2 < N2; i2 +=1)
                {
                  block.tmp[0][i1][i2][0] += CoarseFace[i2+i1*N2].alpha1rho1;
                  block.tmp[0][i1][i2][1] += CoarseFace[i2+i1*N2].alpha2rho2;
                  block.tmp[0][i1][i2][2] += CoarseFace[i2+i1*N2].ru;
                  block.tmp[0][i1][i2][3] += CoarseFace[i2+i1*N2].rv;
                  block.tmp[0][i1][i2][4] += CoarseFace[i2+i1*N2].rw;
                  block.tmp[0][i1][i2][5] += CoarseFace[i2+i1*N2].energy;
                  block.tmp[0][i1][i2][6] += CoarseFace[i2+i1*N2].alpha2;
                  block.tmp[0][i1][i2][7] += CoarseFace[i2+i1*N2].dummy;
                }
              }
              else
              {
                for (int i1 = 0 ; i1 < N1; i1 +=1)
                for (int i2 = 0 ; i2 < N2; i2 +=1)
                {         
                  block.tmp[TBlock::sizeZ-1][i1][i2][0] += CoarseFace[i2+i1*N2].alpha1rho1;
                  block.tmp[TBlock::sizeZ-1][i1][i2][1] += CoarseFace[i2+i1*N2].alpha2rho2;
                  block.tmp[TBlock::sizeZ-1][i1][i2][2] += CoarseFace[i2+i1*N2].ru;
                  block.tmp[TBlock::sizeZ-1][i1][i2][3] += CoarseFace[i2+i1*N2].rv;
                  block.tmp[TBlock::sizeZ-1][i1][i2][4] += CoarseFace[i2+i1*N2].rw;
                  block.tmp[TBlock::sizeZ-1][i1][i2][5] += CoarseFace[i2+i1*N2].energy;
                  block.tmp[TBlock::sizeZ-1][i1][i2][6] += CoarseFace[i2+i1*N2].alpha2;
                  block.tmp[TBlock::sizeZ-1][i1][i2][7] += CoarseFace[i2+i1*N2].dummy;
          

                }
              }

            }     
          }    
        }//icode = 0,...,26       
      }
    }


};



}//namespace AMR_CUBISM