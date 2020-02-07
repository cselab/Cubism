#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>
#include <array>
#include <algorithm>

#ifdef CUBISM_USE_NUMA
#include <numa.h>
#include <omp.h>
#endif

#include "BlockInfo.h"
#include "MeshMap.h"




#define HACK





namespace cubism //AMR_CUBISM
{

template <typename Block, template<typename X> class allocator=std::allocator>
class Grid
{  

protected:
    std::vector <BlockInfo> m_vInfo; //meta-data for blocks that belong to this rank
    std::vector <Block * > m_blocks; //pointers to blocks that belong to this rank
    std::vector <std::vector<BlockInfo> > BlockInfoAll; 

    const int NX;         //Total # of blocks for level 0 in X-direction  
    const int NY;         //Total # of blocks for level 0 in Y-direction
    const int NZ;         //Total # of blocks for level 0 in Z-direction
    const double maxextent;        //Maximum domain extent
    const int levelMax;   //Maximum refinement level allowed
    const int levelStart; //Initial refinement level      
    const bool xperiodic;
    const bool yperiodic;
    const bool zperiodic;


    int **** Zholder;

public:
   
    int N;                     //Current number of blocks
    typedef Block BlockType;
    typedef typename Block::RealType Real;  //Block MUST provide `RealType`.
    
    SpaceFillingCurve Zcurve;

    void _alloc() //called in class constructor
    {
        int m=levelStart;
        int TwoPower = pow(2,m);
        for (int n=0; n<NX*NY*NZ*pow(TwoPower,3); n++) 
        {
            _alloc(m,n);
        }
        FillPos();
    }

    void _alloc(int m, int n) //called whenever the grid is refined
    {
        allocator <Block> alloc;
        BlockInfoAll[m][n].ptrBlock = alloc.allocate(1);
        //BlockInfoAll[m][n].ptrBlock = (Block*)calloc(1,sizeof(Block)); 
       
        #ifdef HACK
            BlockInfoAll[m][n].h_gridpoint = BlockInfoAll[m][n].h;
        #endif

        //m_blocks.push_back((Block * )BlockInfoAll[m][n].ptrBlock);
        //m_vInfo. push_back(BlockInfoAll[m][n]);

        assert(BlockInfoAll[m][n].ptrBlock != NULL);
        N++;        
    }

    virtual void _deallocAll() //called in class destructor
    {
        m_blocks.clear();
        m_vInfo.clear();
        for (int m=0; m<levelMax; m++)
        {
            for (int n=0; n<NX*NY*NZ*pow(pow(2,m),3); n++)
            {
                if (BlockInfoAll[m][n].TreePos==Exists) 
                {
                    allocator <Block> alloc;
                    //alloc.deallocate((Block*)BlockInfoAll[m][n].ptrBlock,1);
                }
            }
        }    
    }



    void _dealloc(int m, int n) //called whenever the grid is compressed
    {
        N --;        
        m_blocks.clear();
        m_vInfo.clear();      
        allocator <Block> alloc;
        alloc.deallocate((Block*)BlockInfoAll[m][n].ptrBlock,1);         
    }

    virtual void FillPos()
    {
        m_blocks.clear();
        m_vInfo.clear();
        for (int m=0; m<levelMax; m++)
        {
            for (int n=0; n<NX*NY*NZ*pow(pow(2,m),3); n++)
            {
                if (BlockInfoAll[m][n].TreePos == Exists) 
                {
                    m_vInfo.push_back(BlockInfoAll[m][n]);
                    m_blocks.push_back((Block*)BlockInfoAll[m][n].ptrBlock);
                }
            }

        }   
    }

   
    #ifdef HACK //empty functions just to make the code compile with stretched meshes 
        std::vector<MeshMap<Block>*> m_mesh_maps;

        Grid(const MeshMap<Block>* const mapX, const MeshMap<Block>* const mapY, const MeshMap<Block>* const mapZ, const int _NX, const int _NY=1, const int _NZ=1) :m_blocks(NULL),maxextent(-1.0),N(_NX*_NY*_NZ),NX(_NX), NY(_NY), NZ(_NZ)
        {
            std::cout <<"Grid was constructed using MeshMap in an AMR setting. Are you sure?\n";
            assert(false);
            abort();
        }
    
        double getH() const
        {
            //std::vector<BlockInfo> vInfo = this->getBlocksInfo();
            //BlockInfo info = vInfo[0];
            return -1.0;//info.h_gridpoint;
        }
    
        inline MeshMap<Block>& getMeshMap(const int i)
        {
            assert(false);
            abort();
            assert(i>=0 && i<3);
            return *m_mesh_maps[i];
        }
        inline const MeshMap<Block>& getMeshMap(const int i) const
        {
            assert(false);
            abort();
            assert(i>=0 && i<3);
            return *m_mesh_maps[i];
        }

        void setup(const unsigned int nX, const unsigned int nY, const unsigned int nZ)
        {
            assert(false && "You called Grid::setup() in an AMR solver. Do you really need that?\n");
            abort();
        }
    
        virtual int getBlocksPerDimension(int idim) const
        {
            assert(false && "You called Grid::getBlocksPerDimension() in an AMR solver. Do you really need that?\n");
            abort();
        }
    #endif


    Grid(const unsigned int _NX, 
         const unsigned int _NY = 1, 
         const unsigned int _NZ = 1, 
         const double _maxextent = 1,
         const unsigned int _levelStart = 0,
         const unsigned int _levelMax = 1,     
         const bool AllocateBlocks = true,
         const bool a_xperiodic = true,
         const bool a_yperiodic = true,
         const bool a_zperiodic = true ):
          NX(_NX), NY(_NY), NZ(_NZ), maxextent(_maxextent),levelMax(_levelMax), levelStart(_levelStart), xperiodic(a_xperiodic),yperiodic(a_yperiodic),zperiodic(a_zperiodic),Zcurve(NX,NY,NZ)    
    {
        N = 0 ;
        int blocksize[3] = {Block::sizeX,Block::sizeY,Block::sizeZ};
        int Bmin[3] = {NX, NY, NZ};
        double h0 = (maxextent / std::max(NX*Block::sizeX, std::max(NY*Block::sizeY, NZ*Block::sizeZ)));



             //We loop over all levels m=0,...,levelMax-1 and all blocks found in each level. All blockInfos are initialized here.       
        BlockInfoAll.resize(levelMax);
        
        Zholder = new int *** [levelMax];

        for (int m=0; m<levelMax; m++)
        {
            int TwoPower  = pow(2,m);
            const unsigned int Ntot = NX*NY*NZ*pow(TwoPower,3);
            
            BlockInfoAll[m].resize(Ntot);
            
            double h = h0 / TwoPower;

            double origin[3];

            Zholder[m] = new int ** [NX * TwoPower];  
            for (int i=0; i<NX * TwoPower; i++)
            {
                Zholder[m][i] = new int * [NY * TwoPower];
                for (int j=0; j<NY * TwoPower; j++)
                    Zholder[m][i][j] = new int [NZ * TwoPower];
            }


            for (int i=0; i<NX * TwoPower; i++)
            for (int j=0; j<NY * TwoPower; j++)
            for (int k=0; k<NZ * TwoPower; k++)
            {
                int n = Zcurve.forward(m,i,j,k);

                Zholder[m][i][j][k] = n;
                
                int IJK[3] = {i,j,k};
                origin[0]  = i*blocksize[0]*h;
                origin[1]  = j*blocksize[1]*h;
                origin[2]  = k*blocksize[2]*h;

                TreePosition TreePos;
                if      (m==levelStart) TreePos = Exists;
                else if (m <levelStart) TreePos = CheckFiner;
                else                    TreePos = CheckCoarser;
                
                int rank = (m==levelStart) ? 0 : -1;
                BlockInfoAll[m][n].setup(m,h,origin,Bmin,IJK,rank,TreePos); //Ranks are initialized in GridMPI constructor
            }
        }

        if (AllocateBlocks) _alloc();
    }
  
    virtual ~Grid() {_deallocAll();}

    virtual bool avail(int ix, int iy, int iz, int m) const { return true; }
  
    virtual int rank() const { return 0; }
  

    int getZforward(const int level,const int i, const int j, const int k) const 
    {
        int TwoPower = pow(2,level);
        int ix = (i+TwoPower*NX) % (NX*TwoPower);
        int iy = (j+TwoPower*NY) % (NY*TwoPower);
        int iz = (k+TwoPower*NZ) % (NZ*TwoPower);

        //return Zholder[level][ix][iy][iz];
        return   Zcurve.forward(level,ix,iy,iz);
    }

    int getZchild(int level,int i, int j, int k)
    {
        return Zcurve.child(level,i,j,k);
    }
   
    virtual Block& operator()(int ix, int iy, int iz, int m) const
    { 
        int n = getZforward(m,ix,iy,iz);
        return  * (Block * ) BlockInfoAll[m][n].ptrBlock;  
    }

    virtual std::array<int,3> getMaxBlocks() const
    {
        return {NX,NY,NZ};
    }

    inline int getlevelMax()
    {
        return levelMax;
    }

    inline int getlevelMax() const 
    {
        return levelMax;
    }

    inline BlockInfo & getBlockInfoAll(int m, int n) 
    {
        return BlockInfoAll[m][n];
    }

    virtual BlockInfo getBlockInfoAll(int m,int n) const
    {
        return BlockInfoAll[m][n];
    }


    inline std::vector<std::vector<BlockInfo>> & getBlockInfoAll() 
    {
        return BlockInfoAll;
    }

    virtual std::vector<std::vector<BlockInfo>> getBlockInfoAll() const
    {
        return BlockInfoAll;
    }


    inline std::vector < Block * > & GetBlocks()
    {
    	return  m_blocks;
    }

    inline const std::vector < Block * > & GetBlocks() const 
    {
        return  m_blocks;
    }   

    virtual std::vector<BlockInfo>& getBlocksInfo()
    {
        return m_vInfo;
    }

    virtual const std::vector<BlockInfo>& getBlocksInfo() const 
    {
        return m_vInfo;
    }
};

}//namespace AMR_CUBISM