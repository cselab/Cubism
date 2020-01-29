#pragma once

#include "Matrix3D.h"
#include "Grid.h"

// #include <omp.h>
#include <cstring>
#include <string>

#ifdef __bgq__
#include <builtins.h>
#define memcpy2(a,b,c)  __bcopy((b),(a),(c))
#else
#define memcpy2(a,b,c)  memcpy((a),(b),(c))
#endif






//for debug only
#include<math.h>
#include <mpi.h>


namespace cubism //AMR_CUBISM
{

/*
   Working copy of Block + Ghosts.
   Data of original block is copied (!) here. So when changing something in
   the lab we are not changing the original data.
   Works for (inner) blocks on the same rank. The blocks may have a different resolution; however
   two adjacent blocks are assumed to differ by at most one level of resolution (no adjacent blocks
   with grid spacing h and h/4 are allowed). Refinement ratio is (of course) 2.   
*/


template<typename TBlock,template<typename X> class allocator = std::allocator,typename ElementTypeT = typename TBlock::ElementType>
class BlockLab
{

  public:
    typedef ElementTypeT ElementType;
    typedef typename ElementTypeT::RealType Real;//Element type MUST provide `RealType`.

  protected:
    typedef TBlock BlockType;
    typedef typename BlockType::ElementType ElementTypeBlock;
    
    enum eBlockLab_State{eMRAGBlockLab_Prepared, eMRAGBlockLab_Loaded, eMRAGBlockLab_Uninitialized};
    eBlockLab_State m_state;

    Matrix3D<ElementType, true, allocator> * m_cacheBlock; //This is filled by the Blocklab
    int m_stencilStart[3], m_stencilEnd[3];
    bool istensorial;

    const Grid<BlockType, allocator>* m_refGrid;
    int NX, NY, NZ;
  


    //Extra stuff for AMR:
    Matrix3D<ElementType, true, allocator> * m_CoarsenedBlock; //coarsened version of given block
    int m_InterpStencilStart[3], m_InterpStencilEnd[3]; //stencil used for refinement (assumed tensorial) 
    bool coarsened; //will be true if block has at least one coarser neighbor 
       
    virtual void _apply_bc(const BlockInfo& info, const Real t=0, bool coarse = false) { }

    template<typename T>
    void _release(T *& t)
    {
      if (t != NULL)
      {
        allocator<T>().destroy(t);
        allocator<T>().deallocate(t,1);
      }
      t = NULL;
    }

  public:

    BlockLab(): m_state(eMRAGBlockLab_Uninitialized), m_cacheBlock(nullptr), m_refGrid(nullptr), m_CoarsenedBlock(nullptr)   
    {
      m_stencilStart      [0] = m_stencilStart      [1] = m_stencilStart      [2] = 0;
      m_stencilEnd        [0] = m_stencilEnd        [1] = m_stencilEnd        [2] = 0;
      m_InterpStencilStart[0] = m_InterpStencilStart[1] = m_InterpStencilStart[2] = 0;
      m_InterpStencilEnd  [0] = m_InterpStencilEnd  [1] = m_InterpStencilEnd  [2] = 0;
    }

    virtual std::string name() const { return "BlockLab"; }
    virtual bool is_xperiodic() { return true; }
    virtual bool is_yperiodic() { return true; }
    virtual bool is_zperiodic() { return true; }

    virtual ~BlockLab()
    {
      _release(m_cacheBlock);
      _release(m_CoarsenedBlock);
    }

    template <int dim>
    int getActualSize() const
    {
      assert(dim>=0 && dim<3);
      return m_cacheBlock->getSize()[dim];
    }

    inline ElementType * getBuffer() const { return &m_cacheBlock->LinAccess(0);}

    // rasthofer May 2016: required for non-relecting time-dependent boundary conditions
    //virtual void apply_bc_update(const BlockInfo& info, const Real dt=0, const Real a=0, const Real b=0) { }

    void prepare(Grid<BlockType,allocator>& grid, int  startX, int  endX, int  startY, int  endY, int  startZ, int  endZ, const bool _istensorial,int IstartX = -1, int IendX = 2, int IstartY = -1, int IendY = 2,int IstartZ = -1, int IendZ = 2)
    {
      const int ss[3] = {startX, startY, startZ};
      const int se[3] = {endX  , endY  , endZ  };
      const int Iss[3] = {IstartX, IstartY, IstartZ};
      const int Ise[3] = {IendX  , IendY  , IendZ  };
      prepare(grid, ss, se, _istensorial, Iss, Ise);
    }



    /**
     * Prepare the extended block.
     * @param collection    Collection of blocks in the grid (e.g. result of Grid::getBlockCollection()).
     * @param boundaryInfo  Info on the boundaries of the grid (e.g. result of Grid::getBoundaryInfo()).
     * @param stencil_start Maximal stencil used for computations at lower boundary.
     *                      Defines how many ghosts we will get in extended block.
     * @param stencil_end   Maximal stencil used for computations at lower boundary.
     *                      Defines how many ghosts we will get in extended block.
     */

    void prepare(Grid<BlockType,allocator>& grid, const int stencil_start[3],const int stencil_end  [3], const bool _istensorial,const int Istencil_start[3],const int Istencil_end  [3])
    {
      istensorial = true; //_istensorial;

      m_refGrid = &grid;

      assert(stencil_start[0]>= -BlockType::sizeX);
      assert(stencil_start[1]>= -BlockType::sizeY);
      assert(stencil_start[2]>= -BlockType::sizeZ);
      assert(stencil_end[0] < BlockType::sizeX*2);
      assert(stencil_end[1] < BlockType::sizeY*2);
      assert(stencil_end[2] < BlockType::sizeZ*2);

      m_stencilStart[0] = stencil_start[0];
      m_stencilStart[1] = stencil_start[1];
      m_stencilStart[2] = stencil_start[2];

      m_stencilEnd[0] = stencil_end[0];
      m_stencilEnd[1] = stencil_end[1];
      m_stencilEnd[2] = stencil_end[2];

      assert(m_stencilStart[0]<=m_stencilEnd[0]);
      assert(m_stencilStart[1]<=m_stencilEnd[1]);
      assert(m_stencilStart[2]<=m_stencilEnd[2]);

      if (m_cacheBlock == NULL ||
         (int) m_cacheBlock->getSize()[0] != (int)BlockType::sizeX + m_stencilEnd[0] - m_stencilStart[0] -1 ||
         (int) m_cacheBlock->getSize()[1] != (int)BlockType::sizeY + m_stencilEnd[1] - m_stencilStart[1] -1 ||
         (int) m_cacheBlock->getSize()[2] != (int)BlockType::sizeZ + m_stencilEnd[2] - m_stencilStart[2] -1 )
      {
        if (m_cacheBlock != NULL)
          _release(m_cacheBlock);

        m_cacheBlock = allocator< Matrix3D<ElementType,  true, allocator> >().allocate(1);

        allocator< Matrix3D<ElementType,  true, allocator> >().construct(m_cacheBlock);

        m_cacheBlock->_Setup(BlockType::sizeX + m_stencilEnd[0] - m_stencilStart[0] -1,
                             BlockType::sizeY + m_stencilEnd[1] - m_stencilStart[1] -1,
                             BlockType::sizeZ + m_stencilEnd[2] - m_stencilStart[2] -1);

      }

      coarsened = false;
      m_InterpStencilStart[0] = Istencil_start[0];
      m_InterpStencilStart[1] = Istencil_start[1];
      m_InterpStencilStart[2] = Istencil_start[2];

      m_InterpStencilEnd[0] = Istencil_end[0];
      m_InterpStencilEnd[1] = Istencil_end[1];
      m_InterpStencilEnd[2] = Istencil_end[2];

      assert(m_InterpStencilStart[0]<=m_InterpStencilEnd[0] &&  m_InterpStencilEnd[0] == 2);
      assert(m_InterpStencilStart[1]<=m_InterpStencilEnd[1] &&  m_InterpStencilEnd[1] == 2);
      assert(m_InterpStencilStart[2]<=m_InterpStencilEnd[2] &&  m_InterpStencilEnd[2] == 2);


      const int e[3] = {(m_stencilEnd[0])/2 + 1 + m_InterpStencilEnd[0] -1,
                        (m_stencilEnd[1])/2 + 1 + m_InterpStencilEnd[1] -1,
                        (m_stencilEnd[2])/2 + 1 + m_InterpStencilEnd[2] -1};

      const int s[3] = {(m_stencilStart[0]-1)/2+ m_InterpStencilStart[0],
                        (m_stencilStart[1]-1)/2+ m_InterpStencilStart[1],
                        (m_stencilStart[2]-1)/2+ m_InterpStencilStart[2]};

       
      if (m_CoarsenedBlock == NULL ||
          (int) m_CoarsenedBlock->getSize()[0] != (int)BlockType::sizeX/2 + e[0] - s[0] -1 ||
          (int) m_CoarsenedBlock->getSize()[1] != (int)BlockType::sizeY/2 + e[1] - s[1] -1 ||
          (int) m_CoarsenedBlock->getSize()[2] != (int)BlockType::sizeZ/2 + e[2] - s[2] -1 )
      {
          if (m_CoarsenedBlock != NULL)
            _release(m_CoarsenedBlock);

          m_CoarsenedBlock = allocator< Matrix3D<ElementType,  true, allocator> >().allocate(1);

          allocator< Matrix3D<ElementType,  true, allocator> >().construct(m_CoarsenedBlock);
      
          m_CoarsenedBlock->_Setup(BlockType::sizeX/2 + e[0] - s[0] -1,
                                   BlockType::sizeY/2 + e[1] - s[1] -1,
                                   BlockType::sizeZ/2 + e[2] - s[2] -1);
      }

      m_state = eMRAGBlockLab_Prepared;   
    }


    /**
     * Load a block (incl. ghosts for it).
     * This is not called internally but by the BlockProcessing-class. Hence a new version of BlockLab,
     * can just overwrite it and through template-passing to BlockProcessing, the right version will be
     * called.
     * @param info  Reference to info of block to be loaded.
     */
    void load(BlockInfo info, const Real t=0, const bool applybc=true)
    {
      const Grid<BlockType,allocator>& grid = *m_refGrid;
      const int nX = BlockType::sizeX;
      const int nY = BlockType::sizeY;
      const int nZ = BlockType::sizeZ;
    
      //0. couple of checks
      //1. load the block into the cache
      //2. put the ghosts into the cache

      //0.
      assert(m_state == eMRAGBlockLab_Prepared || m_state==eMRAGBlockLab_Loaded);
      assert(m_cacheBlock != NULL);
      assert(info.TreePos == Exists);
      *m_cacheBlock     =  NAN;
      *m_CoarsenedBlock =  NAN;

      //std::cout << name() << "\n";
       
      //double t0 = omp_get_wtime();
      //1.
    {
        assert(sizeof(ElementType) == sizeof(typename BlockType::ElementType));

        BlockType& block = *(BlockType *)info.ptrBlock;

        ElementTypeBlock * ptrSource = &block(0);

        #if 0 // original
          for(int iz=0; iz<nZ; iz++)
          for(int iy=0; iy<nY; iy++)
          {
            ElementType * ptrDestination = &m_cacheBlock->Access(0-m_stencilStart[0], iy-m_stencilStart[1], iz-m_stencilStart[2]);
    
            //for(int ix=0; ix<nX; ix++, ptrSource++, ptrDestination++)
            //  *ptrDestination = (ElementType)*ptrSource;
            memcpy2((char *)ptrDestination, (char *)ptrSource, sizeof(ElementType)*nX);
            ptrSource+= nX;
          }
        #else
          const int nbytes = sizeof(ElementType)*nX;            
          #if 1 // not bad
            const int _iz0 = -m_stencilStart[2];
            const int _iz1 = _iz0 + nZ;
            const int _iy0 = -m_stencilStart[1];
            const int _iy1 = _iy0 + nY;
          
            const int m_vSize0 = m_cacheBlock->getSize(0); //m_vSize[0];
            const int m_nElemsPerSlice = m_cacheBlock->getNumberOfElementsPerSlice(); //m_nElementsPerSlice;
        
            const int my_ix = 0-m_stencilStart[0];
            for(int iz=_iz0; iz<_iz1; iz++)
            {
              const int my_izx = iz*m_nElemsPerSlice + my_ix;
              for(int iy=_iy0; iy<_iy1; iy+=4)
              {
                ElementType * ptrDestination0 = &m_cacheBlock->LinAccess(my_izx + (iy+0)*m_vSize0);
                ElementType * ptrDestination1 = &m_cacheBlock->LinAccess(my_izx + (iy+1)*m_vSize0);
                ElementType * ptrDestination2 = &m_cacheBlock->LinAccess(my_izx + (iy+2)*m_vSize0);
                ElementType * ptrDestination3 = &m_cacheBlock->LinAccess(my_izx + (iy+3)*m_vSize0);
        
                memcpy2((char *)ptrDestination0, (char *)(ptrSource+0*nX), nbytes);
                memcpy2((char *)ptrDestination1, (char *)(ptrSource+1*nX), nbytes);
                memcpy2((char *)ptrDestination2, (char *)(ptrSource+2*nX), nbytes);
                memcpy2((char *)ptrDestination3, (char *)(ptrSource+3*nX), nbytes);
                ptrSource+= 4*nX;
              }
            }
          #else
            #if 1 // not bad either
              const int _iz0 = -m_stencilStart[2];
              const int _iz1 = _iz0 + nZ;
              const int _iy0 = -m_stencilStart[1];
              const int _iy1 = _iy0 + nY;
              for(int iz=_iz0; iz<_iz1; iz++)
              for(int iy=_iy0; iy<_iy1; iy++)
            #else
              for(int iz=-m_stencilStart[2]; iz<nZ-m_stencilStart[2]; iz++)
              for(int iy=-m_stencilStart[1]; iy<nY-m_stencilStart[1]; iy++)
            #endif
              {
                ElementType * ptrDestination = &m_cacheBlock->Access(0-m_stencilStart[0], iy, iz);
                //for(int ix=0; ix<nX; ix++, ptrSource++, ptrDestination++)
                // *ptrDestination = (ElementType)*ptrSource;
                memcpy2((char *)ptrDestination, (char *)ptrSource, nbytes);
                //for (int ix = 0; ix < nX; ix++)  ptrDestination[ix] = ptrSource[ix];
                ptrSource+= nX;
              }
          #endif
        #endif
    }
      //double t1 = omp_get_wtime();
      
      std::array <int,3> blocksPerDim = grid.getMaxBlocks();
      int aux = pow(2,info.level);
      NX = blocksPerDim[0]*aux;
      NY = blocksPerDim[1]*aux;
      NZ = blocksPerDim[2]*aux;    

      //2.
      {
        const bool xperiodic = is_xperiodic();
        const bool yperiodic = is_yperiodic();
        const bool zperiodic = is_zperiodic();

        const bool xskin = info.index[0]==0 || info.index[0]==blocksPerDim[0]*aux-1;
        const bool yskin = info.index[1]==0 || info.index[1]==blocksPerDim[1]*aux-1;
        const bool zskin = info.index[2]==0 || info.index[2]==blocksPerDim[2]*aux-1;

        const int xskip  = info.index[0]==0 ? -1 : 1;
        const int yskip  = info.index[1]==0 ? -1 : 1;
        const int zskip  = info.index[2]==0 ? -1 : 1;

        coarsened = false;
        for(int icode=0; icode<27; icode++)
        {
            if (icode == 1*1 + 3*1 + 9*1) continue;
            const int code[3] = { icode%3-1, (icode/3)%3-1, (icode/9)%3-1};
           
            if (!xperiodic && code[0] == xskip && xskin)                  continue;
            if (!yperiodic && code[1] == yskip && yskin)                  continue;
            if (!zperiodic && code[2] == zskip && zskin)                  continue;                
  
            BlockInfo infoNei = grid.getBlockInfoAll(info.level,info.Znei_(code[0],code[1],code[2]));
            if (infoNei.TreePos == CheckCoarser)
            {
            	coarsened = true;
            	break;
            } 
        }//icode = 0,...,26  



        for(int icode=0; icode<27; icode++)
        {
          if (icode == 1*1 + 3*1 + 9*1) continue;
          const int code[3] = { icode%3-1, (icode/3)%3-1, (icode/9)%3-1};
    
          //mike: get neighbor on same level of resolution
          BlockInfo infoNei = grid.getBlockInfoAll(info.level,info.Znei_(code[0],code[1],code[2]));

          if (!xperiodic && code[0] == xskip && xskin) continue;
          if (!yperiodic && code[1] == yskip && yskin) continue;
          if (!zperiodic && code[2] == zskip && zskin) continue; 

          if (infoNei.TreePos == Exists && coarsened)
          {
          	FillCoarseVersion(info,code);
          }   
          else if (infoNei.TreePos == CheckCoarser)
            CoarseFineExchange(info,code);

          if (!istensorial && abs(code[0])+abs(code[1])+abs(code[2])>1) continue;
        
          //mike : s and e correspond to start and end of this lab's cells that are filled by neighbors
          const int s[3] = { code[0]<1? (code[0]<0 ? m_stencilStart[0]:0 ):nX,
                             code[1]<1? (code[1]<0 ? m_stencilStart[1]:0 ):nY,
                             code[2]<1? (code[2]<0 ? m_stencilStart[2]:0 ):nZ};

          const int e[3] = { code[0]<1? (code[0]<0 ? 0                :nX):nX+m_stencilEnd[0]-1,
                             code[1]<1? (code[1]<0 ? 0                :nY):nY+m_stencilEnd[1]-1,
                             code[2]<1? (code[2]<0 ? 0                :nZ):nZ+m_stencilEnd[2]-1};
          
          if (infoNei.TreePos == Exists )
            SameLevelExchange   (info,code,s,e,m_InterpStencilStart,m_InterpStencilEnd);
          else if (infoNei.TreePos == CheckFiner)  
            FineToCoarseExchange(info,code,s,e);      
        }//icode = 0,...,26       

        m_state = eMRAGBlockLab_Loaded;

      }//2.
      //double t2 = omp_get_wtime();
      //if (applybc) _apply_bc(info, t);
      //double t3 = omp_get_wtime();   
      //printf("load: %5.10e %5.10e %5.10e %5.10e\n", t1-t0, t2-t1, t3-t2, t3-t0);
    }






    void post_load(BlockInfo info, const Real t=0,bool applybc = true)
    {
        const int nX = BlockType::sizeX;
        const int nY = BlockType::sizeY;
        const int nZ = BlockType::sizeZ;

        if (coarsened)
        {
          const int offset[3] = {(m_stencilStart[0]-1)/2+ m_InterpStencilStart[0],(m_stencilStart[1]-1)/2+ m_InterpStencilStart[1],(m_stencilStart[2]-1)/2+ m_InterpStencilStart[2]};
                                          

          for (int k = 0 ; k < nZ/2 ; k++)           
          for (int j = 0 ; j < nY/2 ; j++)
          for (int i = 0 ; i < nX/2 ; i++)
          {
            //if  ( i > -m_InterpStencilStart[0] && i <nX/2 -  m_InterpStencilEnd[0]  
            //   && j > -m_InterpStencilStart[1] && j <nY/2 -  m_InterpStencilEnd[1]  
            //   && k > -m_InterpStencilStart[2] && k <nZ/2 -  m_InterpStencilEnd[2] ) continue;

            const int ix = 2*i  - m_stencilStart[0];
            const int iy = 2*j  - m_stencilStart[1];
            const int iz = 2*k  - m_stencilStart[2];
            ElementType &coarseElement =  m_CoarsenedBlock->Access(i-offset[0],j-offset[1],k-offset[2]);
            coarseElement = AverageDown(m_cacheBlock->Read(ix  ,iy  ,iz  ),
                                        m_cacheBlock->Read(ix+1,iy  ,iz  ),
                                        m_cacheBlock->Read(ix  ,iy+1,iz  ),
                                        m_cacheBlock->Read(ix+1,iy+1,iz  ),
                                        m_cacheBlock->Read(ix  ,iy  ,iz+1),
                                        m_cacheBlock->Read(ix+1,iy  ,iz+1),
                                        m_cacheBlock->Read(ix  ,iy+1,iz+1),
                                        m_cacheBlock->Read(ix+1,iy+1,iz+1)); 

          }

          if (applybc) _apply_bc(info, t, true); //apply BC to coarse block
          CoarseFineInterpolation(info);
        }
        if (applybc) _apply_bc(info, t);
	}






  
    void SameLevelExchange(BlockInfo info, const int * const code , const int * const s, const int * const e, int * sC= NULL ,  int * eC= NULL)
    {
      const Grid<BlockType,allocator>& grid = *m_refGrid;
    
      if (!grid.avail(info.index[0] + code[0], info.index[1] + code[1], info.index[2] + code[2],info.level)) return;
  
      const int nX = BlockType::sizeX;
      const int nY = BlockType::sizeY;
      const int nZ = BlockType::sizeZ;
      
      BlockType& b = grid(info.index_(0) + code[0], info.index_(1) + code[1], info.index_(2) + code[2],info.level_());

      #if 1
        const int bytes = (e[0]-s[0])*sizeof(ElementType);
        if (!bytes) return;

        const int m_vSize0 = m_cacheBlock->getSize(0); 
        const int m_nElemsPerSlice = m_cacheBlock->getNumberOfElementsPerSlice(); 
        const int my_ix = s[0]-m_stencilStart[0];

        for(int iz=s[2]; iz<e[2]; iz++)
        {
          const int my_izx = (iz-m_stencilStart[2])*m_nElemsPerSlice + my_ix;
          #if 0
            for(int iy=s[1]; iy<e[1]; iy++)
            {
              #if 1   // ...
                //char * ptrDest = (char*)&m_cacheBlock->Access(s[0]-m_stencilStart[0], iy-m_stencilStart[1], iz-m_stencilStart[2]);
                char * ptrDest = (char*)&m_cacheBlock->LinAccess(my_izx + (iy-m_stencilStart[1])*m_vSize0);
                const char * ptrSrc = (const char*)&b(s[0] - code[0]*BlockType::sizeX, iy - code[1]*BlockType::sizeY, iz - code[2]*BlockType::sizeZ);
                memcpy2((char *)ptrDest, (char *)ptrSrc, bytes);
              #else
                for(int ix=s[0]; ix<e[0]; ix++)
                  m_cacheBlock->Access(ix-m_stencilStart[0], iy-m_stencilStart[1], iz-m_stencilStart[2]) = (ElementType)b(ix - code[0]*BlockType::sizeX, iy - code[1]*BlockType::sizeY, iz - code[2]*BlockType::sizeZ);
              #endif
            }
          #else
            if ((e[1]-s[1]) % 4 != 0)
            {
              for(int iy=s[1]; iy<e[1]; iy++)
              {
                char * ptrDest = (char*)&m_cacheBlock->LinAccess(my_izx + (iy-m_stencilStart[1])*m_vSize0);
                const char * ptrSrc = (const char*)&b(s[0] - code[0]*nX, iy - code[1]*nY, iz - code[2]*nZ);
                const int cpybytes = (e[0]-s[0])*sizeof(ElementType);
                memcpy2((char *)ptrDest, (char *)ptrSrc, cpybytes);
              }
            }
            else
            {
              for(int iy=s[1]; iy<e[1]; iy+=4)
              {
                char * ptrDest0 = (char*)&m_cacheBlock->LinAccess(my_izx +(iy+0-m_stencilStart[1])*m_vSize0);
                char * ptrDest1 = (char*)&m_cacheBlock->LinAccess(my_izx +(iy+1-m_stencilStart[1])*m_vSize0);
                char * ptrDest2 = (char*)&m_cacheBlock->LinAccess(my_izx +(iy+2-m_stencilStart[1])*m_vSize0);
                char * ptrDest3 = (char*)&m_cacheBlock->LinAccess(my_izx +(iy+3-m_stencilStart[1])*m_vSize0);
     
                const char * ptrSrc0 = (const char*)&b(s[0] - code[0]*nX,iy + 0 - code[1]*nY,iz - code[2]*nZ);
                const char * ptrSrc1 = (const char*)&b(s[0] - code[0]*nX,iy + 1 - code[1]*nY,iz - code[2]*nZ);
                const char * ptrSrc2 = (const char*)&b(s[0] - code[0]*nX,iy + 2 - code[1]*nY,iz - code[2]*nZ);
                const char * ptrSrc3 = (const char*)&b(s[0] - code[0]*nX,iy + 3 - code[1]*nY,iz - code[2]*nZ);
     
                memcpy2((char *)ptrDest0, (char *)ptrSrc0, bytes);
                memcpy2((char *)ptrDest1, (char *)ptrSrc1, bytes);
                memcpy2((char *)ptrDest2, (char *)ptrSrc2, bytes);
                memcpy2((char *)ptrDest3, (char *)ptrSrc3, bytes);
              }
            }
          #endif
        }
      #else
        const int off_x = - code[0]*nX + m_stencilStart[0];
        const int off_y = - code[1]*nY + m_stencilStart[1];
        const int off_z = - code[2]*nZ + m_stencilStart[2];
        const int nbytes = (e[0]-s[0])*sizeof(ElementType);
        #if 1
          const int _iz0 = s[2] -m_stencilStart[2];
          const int _iz1 = e[2] -m_stencilStart[2];
          const int _iy0 = s[1] -m_stencilStart[1];
          const int _iy1 = e[1] -m_stencilStart[1];
          for(int iz=_iz0; iz<_iz1; iz++)
          for(int iy=_iy0; iy<_iy1; iy++)    
        #else
          for(int iz=s[2]-m_stencilStart[2]; iz<e[2]-m_stencilStart[2]; iz++)
          for(int iy=s[1]-m_stencilStart[1]; iy<e[1]-m_stencilStart[1]; iy++)
        #endif
          {
            #if 1
              char * ptrDest = (char*)&m_cacheBlock->Access(s[0]-m_stencilStart[0], iy, iz);
              const char * ptrSrc = (const char*)&b(0 + off_x, iy + off_y, iz + off_z);
              memcpy2(ptrDest, ptrSrc, nbytes);
            #else
              for(int ix=s[0]-m_stencilStart[0]; ix<e[0]-m_stencilStart[0]; ix++)
                m_cacheBlock->Access(ix, iy, iz) = (ElementType)b(ix + off_x , iy + off_y, iz + off_z);
            #endif
          }
      #endif
    }


    ElementType AverageDown(const ElementType & e0, const ElementType & e1, const ElementType & e2, const ElementType & e3, const ElementType & e4, const ElementType & e5, const ElementType & e6, const ElementType & e7)
    {
      ElementType retval = 0.125*( (e0+e1) + (e2+e3) + (e4+e5) + (e6+e7) );
      return retval;
    }


    void FineToCoarseExchange(BlockInfo info, const int * const code, const int * const s, const int * const e)
    {
      //Take averaged-down values from finer neighbors (low level stuff) 

      const Grid<BlockType,allocator>& grid = *m_refGrid;
      const int nX = BlockType::sizeX;
      const int nY = BlockType::sizeY;
      const int nZ = BlockType::sizeZ;
        

      const int bytes =   ( abs(code[0])*(e[0]-s[0]) + (1-abs(code[0]))*((e[0]-s[0])/2) ) *sizeof(ElementType);
      if (!bytes) return;
         
      const int m_vSize0         = m_cacheBlock->getSize(0);
      const int m_nElemsPerSlice = m_cacheBlock->getNumberOfElementsPerSlice();    
      const int yStep = (code[1] == 0) ? 2:1;
      const int zStep = (code[2] == 0) ? 2:1;
           
      int Bstep = 1; //face
      if      ((abs(code[0])+abs(code[1])+abs(code[2])==2 )) Bstep = 3; //edge
      else if ((abs(code[0])+abs(code[1])+abs(code[2])==3 )) Bstep = 4; //corner

      /*
        A corner has one finer block.
        An edge has two finer blocks, corresponding to B=0 and B=3. The block B=0 is the one closer to the origin (0,0,0).

        A face has four finer blocks. They are numbered as follows, depending on whether the face lies on the xy- , yz- or xz- plane

        y                                  z                                  z                                         
        ^                                  ^                                  ^                                         
        |                                  |                                  |                                         
        |                                  |                                  |                                         
        |_________________                 |_________________                 |_________________                 
        |        |        |                |        |        |                |        |        |                
        |    2   |   3    |                |    2   |   3    |                |    2   |   3    |                
        |________|________|                |________|________|                |________|________|                
        |        |        |                |        |        |                |        |        |                
        |    0   |    1   |                |    0   |    1   |                |    0   |    1   |                
        |________|________|------------->x |________|________|------------->x |________|________|------------->y 

      */

      for (int B = 0 ; B <= 3 ; B += Bstep) //loop over blocks that make up face/edge/corner (respectively 4,2 or 1 blocks)
      {
        const int aux = (abs(code[0])==1) ? (B%2) : (B/2) ;
  

        if (!grid.avail(2*info.index[0] + max(code[0],0) +code[0]  + (B%2)*max(0, 1 - abs(code[0]))  ,
                        2*info.index[1] + max(code[1],0) +code[1]  +  aux *max(0, 1 - abs(code[1]))  , 
                        2*info.index[2] + max(code[2],0) +code[2]  + (B/2)*max(0, 1 - abs(code[2]))  ,info.level+1)) continue;
  


        BlockType& b   = grid(2*info.index[0] + max(code[0],0) +code[0]  + (B%2)*max(0, 1 - abs(code[0]))  ,
                              2*info.index[1] + max(code[1],0) +code[1]  +  aux *max(0, 1 - abs(code[1]))  , 
                              2*info.index[2] + max(code[2],0) +code[2]  + (B/2)*max(0, 1 - abs(code[2]))  ,info.level+1); //get one of those blocks
                       
        const int my_ix =  abs(code[0])*(s[0]-m_stencilStart[0]) + (1-abs(code[0]) )*(  s[0]  -m_stencilStart[0] + (B%2)*(e[0]-s[0])/2);                                                
        for(int iz=s[2]; iz<e[2]; iz+= zStep)
        {                               
          const int my_izx = ( abs(code[2])*(iz-m_stencilStart[2]) + (1-abs(code[2]) )*(iz/2-m_stencilStart[2] + (B/2)*(e[2]-s[2])/2)  )*m_nElemsPerSlice + my_ix;
          #if 1
            for(int iy=s[1]; iy<e[1]; iy+=yStep)
            {
              #if 1   
                char * ptrDest = (char*)&m_cacheBlock->LinAccess(my_izx + ( abs(code[1])*(iy-m_stencilStart[1]) + (1-abs(code[1]) )*(iy/2-m_stencilStart[1] + aux*(e[1]-s[1])/2)  )*m_vSize0);

                const int XX =  s[0] - code[0]*nX  + min(0,code[0])*  (e[0]-s[0]);
                const int YY = (abs(code[1]) == 1) ? 2*(iy- code[1]*nY) + min(0,code[1])*nY : iy ;
                const int ZZ = (abs(code[2]) == 1) ? 2*(iz- code[2]*nZ) + min(0,code[2])*nZ : iz ;                                      

                const ElementType * ptrSrc_0 = (const ElementType *)&b( XX, YY  , ZZ   );
                const ElementType * ptrSrc_1 = (const ElementType *)&b( XX, YY  , ZZ +1);
                const ElementType * ptrSrc_2 = (const ElementType *)&b( XX, YY+1, ZZ   );
                const ElementType * ptrSrc_3 = (const ElementType *)&b( XX, YY+1, ZZ +1);
 
                //average down elements of block b to send to coarser neighbor
                ElementType * ptrSend = new ElementType[bytes / sizeof (ElementType)];                                   
                for (int ee=0; ee< ( abs(code[0])*(e[0]-s[0]) + (1-abs(code[0]))*((e[0]-s[0])/2) ); ee++)
                {
                  //ptrSend[ee] = 0.125* ( (* (ptrSrc_0 + 2*ee   ) +  * (ptrSrc_1 + 2*ee   ) ) +   
                  //                       (* (ptrSrc_2 + 2*ee   ) +  * (ptrSrc_3 + 2*ee   ) ) +   
                  //                       (* (ptrSrc_0 + 2*ee +1) +  * (ptrSrc_1 + 2*ee +1) ) + 
                  //                       (* (ptrSrc_2 + 2*ee +1) +  * (ptrSrc_3 + 2*ee +1) ) );
                  ptrSend[ee] = AverageDown(* (ptrSrc_0 + 2*ee   ),* (ptrSrc_1 + 2*ee   ),* (ptrSrc_2 + 2*ee   ),* (ptrSrc_3 + 2*ee   ),* (ptrSrc_0 + 2*ee +1),* (ptrSrc_1 + 2*ee +1),* (ptrSrc_2 + 2*ee +1),* (ptrSrc_3 + 2*ee +1));
                } 
                memcpy2((char *)ptrDest, (char *)ptrSend, bytes);
                delete [] ptrSend;                                   
              #else
                //mike: this does not work as it is (not modified from uniform case)
                for(int ix=s[0]; ix<e[0]; ix++)
                  m_cacheBlock->Access(ix-m_stencilStart[0], iy-m_stencilStart[1], iz-m_stencilStart[2]) =(ElementType)b(ix - code[0]*BlockType::sizeX, iy - code[1]*BlockType::sizeY, iz - code[2]*BlockType::sizeZ);
              #endif
            }    
          #else //"vectorized"
            if (  ((e[1]-s[1])/yStep) % 4 != 0 )
            {
              for(int iy=s[1]; iy<e[1]; iy+=yStep)
              {
                char * ptrDest = (char*)&m_cacheBlock->LinAccess(my_izx + ( abs(code[1])*(iy-m_stencilStart[1]) + (1-abs(code[1]) )*(iy/2-m_stencilStart[1] + aux*(e[1]-s[1])/2)  )*m_vSize0);

                const int XX =  s[0] - code[0]*nX  + min(0,code[0])*  (e[0]-s[0]);
                const int YY = (abs(code[1]) == 1) ? 2*(iy- code[1]*nY) + min(0,code[1])*nY : iy ;
                const int ZZ = (abs(code[2]) == 1) ? 2*(iz- code[2]*nZ) + min(0,code[2])*nZ : iz ;                                      

                const ElementType * ptrSrc_0 = (const ElementType *)&b( XX, YY  , ZZ   );
                const ElementType * ptrSrc_1 = (const ElementType *)&b( XX, YY  , ZZ +1);
                const ElementType * ptrSrc_2 = (const ElementType *)&b( XX, YY+1, ZZ   );
                const ElementType * ptrSrc_3 = (const ElementType *)&b( XX, YY+1, ZZ +1);
   
                //average down elements of block b to send to coarser neighbor
                ElementType * ptrSend = new ElementType[bytes / sizeof (ElementType)];                                   
                for (int ee=0; ee< ( abs(code[0])*(e[0]-s[0]) + (1-abs(code[0]))*((e[0]-s[0])/2) ); ee++)
                {
                  //ptrSend[ee] = 0.125* ( (* (ptrSrc_0 + 2*ee   ) +  * (ptrSrc_1 + 2*ee   ) ) +   
                  //                       (* (ptrSrc_2 + 2*ee   ) +  * (ptrSrc_3 + 2*ee   ) ) +   
                  //                       (* (ptrSrc_0 + 2*ee +1) +  * (ptrSrc_1 + 2*ee +1) ) + 
                  //                       (* (ptrSrc_2 + 2*ee +1) +  * (ptrSrc_3 + 2*ee +1) ) );                     
                  ptrSend[ee] = AverageDown(* (ptrSrc_0 + 2*ee   ),* (ptrSrc_1 + 2*ee   ),* (ptrSrc_2 + 2*ee   ),* (ptrSrc_3 + 2*ee   ),* (ptrSrc_0 + 2*ee +1),* (ptrSrc_1 + 2*ee +1),* (ptrSrc_2 + 2*ee +1),* (ptrSrc_3 + 2*ee +1));
                }                                 
                memcpy2((char *)ptrDest, (char *)ptrSend, bytes);
                delete [] ptrSend;
              }
            }
            else
            {
              for(int iy=s[1]; iy<e[1]; iy+=4*yStep)
              {
                char * ptrDest0 = (char*)&m_cacheBlock->LinAccess(my_izx + ( abs(code[1])*(iy+0*yStep-m_stencilStart[1]) + (1-abs(code[1]) )*((iy+0*yStep)/2-m_stencilStart[1] + aux*(e[1]-s[1])/2)  )*m_vSize0);
                char * ptrDest1 = (char*)&m_cacheBlock->LinAccess(my_izx + ( abs(code[1])*(iy+1*yStep-m_stencilStart[1]) + (1-abs(code[1]) )*((iy+1*yStep)/2-m_stencilStart[1] + aux*(e[1]-s[1])/2)  )*m_vSize0);
                char * ptrDest2 = (char*)&m_cacheBlock->LinAccess(my_izx + ( abs(code[1])*(iy+2*yStep-m_stencilStart[1]) + (1-abs(code[1]) )*((iy+2*yStep)/2-m_stencilStart[1] + aux*(e[1]-s[1])/2)  )*m_vSize0);
                char * ptrDest3 = (char*)&m_cacheBlock->LinAccess(my_izx + ( abs(code[1])*(iy+3*yStep-m_stencilStart[1]) + (1-abs(code[1]) )*((iy+3*yStep)/2-m_stencilStart[1] + aux*(e[1]-s[1])/2)  )*m_vSize0);
     
                const int XX  =  s[0] - code[0]*nX  + min(0,code[0])*  (e[0]-s[0]);
                const int YY0 = (abs(code[1]) == 1) ? 2*(iy+0*yStep- code[1]*nY) + min(0,code[1])*nY : iy+0*yStep ;
                const int YY1 = (abs(code[1]) == 1) ? 2*(iy+1*yStep- code[1]*nY) + min(0,code[1])*nY : iy+1*yStep ;
                const int YY2 = (abs(code[1]) == 1) ? 2*(iy+2*yStep- code[1]*nY) + min(0,code[1])*nY : iy+2*yStep ;
                const int YY3 = (abs(code[1]) == 1) ? 2*(iy+3*yStep- code[1]*nY) + min(0,code[1])*nY : iy+3*yStep ;
                const int ZZ  = (abs(code[2]) == 1) ? 2*(iz  - code[2]*nZ) + min(0,code[2])*nZ : iz   ;                                      
                          
                const ElementType * ptrSrc_00 = (const ElementType *)&b( XX, YY0  , ZZ   );
                const ElementType * ptrSrc_10 = (const ElementType *)&b( XX, YY0  , ZZ +1);
                const ElementType * ptrSrc_20 = (const ElementType *)&b( XX, YY0+1, ZZ   );
                const ElementType * ptrSrc_30 = (const ElementType *)&b( XX, YY0+1, ZZ +1);

                const ElementType * ptrSrc_01 = (const ElementType *)&b( XX, YY1  , ZZ   );
                const ElementType * ptrSrc_11 = (const ElementType *)&b( XX, YY1  , ZZ +1);
                const ElementType * ptrSrc_21 = (const ElementType *)&b( XX, YY1+1, ZZ   );
                const ElementType * ptrSrc_31 = (const ElementType *)&b( XX, YY1+1, ZZ +1);

                const ElementType * ptrSrc_02 = (const ElementType *)&b( XX, YY2  , ZZ   );
                const ElementType * ptrSrc_12 = (const ElementType *)&b( XX, YY2  , ZZ +1);
                const ElementType * ptrSrc_22 = (const ElementType *)&b( XX, YY2+1, ZZ   );
                const ElementType * ptrSrc_32 = (const ElementType *)&b( XX, YY2+1, ZZ +1);

                const ElementType * ptrSrc_03 = (const ElementType *)&b( XX, YY3  , ZZ   );
                const ElementType * ptrSrc_13 = (const ElementType *)&b( XX, YY3  , ZZ +1);
                const ElementType * ptrSrc_23 = (const ElementType *)&b( XX, YY3+1, ZZ   );
                const ElementType * ptrSrc_33 = (const ElementType *)&b( XX, YY3+1, ZZ +1);

   
                //average down elements of block b to send to coarser neighbor
                ElementType * ptrSend0 = new ElementType[bytes / sizeof (ElementType)]; 
                ElementType * ptrSend1 = new ElementType[bytes / sizeof (ElementType)]; 
                ElementType * ptrSend2 = new ElementType[bytes / sizeof (ElementType)];   
                ElementType * ptrSend3 = new ElementType[bytes / sizeof (ElementType)];                                                                                             
                for (int ee=0; ee< ( abs(code[0])*(e[0]-s[0]) + (1-abs(code[0]))*((e[0]-s[0])/2) ); ee++)
                {
                  ptrSend0[ee] = AverageDown(* (ptrSrc_00 + 2*ee   ),* (ptrSrc_10 + 2*ee   ),* (ptrSrc_20 + 2*ee   ),* (ptrSrc_30 + 2*ee   ),* (ptrSrc_00 + 2*ee +1),* (ptrSrc_10 + 2*ee +1),* (ptrSrc_20 + 2*ee +1),* (ptrSrc_30 + 2*ee +1));
                  ptrSend1[ee] = AverageDown(* (ptrSrc_01 + 2*ee   ),* (ptrSrc_11 + 2*ee   ),* (ptrSrc_21 + 2*ee   ),* (ptrSrc_31 + 2*ee   ),* (ptrSrc_01 + 2*ee +1),* (ptrSrc_11 + 2*ee +1),* (ptrSrc_21 + 2*ee +1),* (ptrSrc_31 + 2*ee +1));
                  ptrSend2[ee] = AverageDown(* (ptrSrc_02 + 2*ee   ),* (ptrSrc_12 + 2*ee   ),* (ptrSrc_22 + 2*ee   ),* (ptrSrc_32 + 2*ee   ),* (ptrSrc_02 + 2*ee +1),* (ptrSrc_12 + 2*ee +1),* (ptrSrc_22 + 2*ee +1),* (ptrSrc_32 + 2*ee +1));
                  ptrSend3[ee] = AverageDown(* (ptrSrc_03 + 2*ee   ),* (ptrSrc_13 + 2*ee   ),* (ptrSrc_23 + 2*ee   ),* (ptrSrc_33 + 2*ee   ),* (ptrSrc_03 + 2*ee +1),* (ptrSrc_13 + 2*ee +1),* (ptrSrc_23 + 2*ee +1),* (ptrSrc_33 + 2*ee +1));
  
                             //     ptrSend0[ee] = 0.125* ( * (ptrSrc_00 + 2*ee ) +   
                             //                             * (ptrSrc_10 + 2*ee ) +   
                             //                             * (ptrSrc_20 + 2*ee ) +   
                             //                             * (ptrSrc_30 + 2*ee ) +   
                             //                             * (ptrSrc_00 + 2*ee +1) + 
                             //                             * (ptrSrc_10 + 2*ee +1) + 
                             //                             * (ptrSrc_20 + 2*ee +1) + 
                             //                             * (ptrSrc_30 + 2*ee +1) ); 
                             //     ptrSend1[ee] = 0.125* ( * (ptrSrc_01 + 2*ee ) +   
                             //                             * (ptrSrc_11 + 2*ee ) +   
                             //                             * (ptrSrc_21 + 2*ee ) +   
                             //                             * (ptrSrc_31 + 2*ee ) +   
                             //                             * (ptrSrc_01 + 2*ee +1) + 
                             //                             * (ptrSrc_11 + 2*ee +1) + 
                             //                             * (ptrSrc_21 + 2*ee +1) + 
                             //                             * (ptrSrc_31 + 2*ee +1) );  
                             //     ptrSend2[ee] = 0.125* ( * (ptrSrc_02 + 2*ee ) +   
                             //                             * (ptrSrc_12 + 2*ee ) +   
                             //                             * (ptrSrc_22 + 2*ee ) +   
                             //                             * (ptrSrc_32 + 2*ee ) +   
                             //                             * (ptrSrc_02 + 2*ee +1) + 
                             //                             * (ptrSrc_12 + 2*ee +1) + 
                             //                             * (ptrSrc_22 + 2*ee +1) + 
                             //                             * (ptrSrc_32 + 2*ee +1) ); 
                             //     ptrSend3[ee] = 0.125* ( * (ptrSrc_03 + 2*ee ) +   
                             //                             * (ptrSrc_13 + 2*ee ) +   
                             //                             * (ptrSrc_23 + 2*ee ) +   
                             //                             * (ptrSrc_33 + 2*ee ) +   
                             //                             * (ptrSrc_03 + 2*ee +1) + 
                             //                             * (ptrSrc_13 + 2*ee +1) + 
                             //                             * (ptrSrc_23 + 2*ee +1) + 
                             //                             * (ptrSrc_33 + 2*ee +1) );                                            
                }                                 
                memcpy2((char *)ptrDest0, (char *)ptrSend0, bytes);
                memcpy2((char *)ptrDest1, (char *)ptrSend1, bytes);
                memcpy2((char *)ptrDest2, (char *)ptrSend2, bytes);
                memcpy2((char *)ptrDest3, (char *)ptrSend3, bytes);
                delete [] ptrSend0;
                delete [] ptrSend1;
                delete [] ptrSend2;
                delete [] ptrSend3;
              }
            }
          #endif
        }
      }//B
    }












    //Improve the following 4 functions (1/4)
    void CoarseFineExchange(BlockInfo  info, const int * const code)
    {
      //Coarse neighbors send their cells. Those are stored in m_CoarsenedBlock and are later used
      //in function CoarseFineInterpolation to interpolate fine values.
      const Grid<BlockType,allocator>& grid = *m_refGrid;
      const int nX = BlockType::sizeX;
      const int nY = BlockType::sizeY;
      const int nZ = BlockType::sizeZ;
      BlockInfo infoNei = grid.getBlockInfoAll(info.level,info.Znei_(code[0],code[1],code[2]));

                    
      const int s[3] = {code[0]<1? (code[0]<0 ? ((m_stencilStart[0]-1)/2+ m_InterpStencilStart[0]) :0 ) : nX/2,
                        code[1]<1? (code[1]<0 ? ((m_stencilStart[1]-1)/2+ m_InterpStencilStart[1]) :0 ) : nY/2,
                        code[2]<1? (code[2]<0 ? ((m_stencilStart[2]-1)/2+ m_InterpStencilStart[2]) :0 ) : nZ/2 };

      const int e[3] = {code[0]<1? (code[0]<0 ? 0:nX/2 ) : nX/2+(m_stencilEnd[0])/2+ m_InterpStencilEnd[0] -1,
                        code[1]<1? (code[1]<0 ? 0:nY/2 ) : nY/2+(m_stencilEnd[1])/2+ m_InterpStencilEnd[1] -1,
                        code[2]<1? (code[2]<0 ? 0:nZ/2 ) : nZ/2+(m_stencilEnd[2])/2+ m_InterpStencilEnd[2] -1};

      const int offset[3] = {(m_stencilStart[0]-1)/2+ m_InterpStencilStart[0],
                             (m_stencilStart[1]-1)/2+ m_InterpStencilStart[1],
                             (m_stencilStart[2]-1)/2+ m_InterpStencilStart[2]};

      const int base[3] = { (info.index[0]+ code[0])%2,
                            (info.index[1]+ code[1])%2,
                            (info.index[2]+ code[2])%2};
   

      if (!grid.avail((infoNei.index[0])/2, 
                      (infoNei.index[1])/2, 
                      (infoNei.index[2])/2,info.level-1)) return;
                        

      BlockType& b = grid((infoNei.index[0])/2, 
                          (infoNei.index[1])/2, 
                          (infoNei.index[2])/2,info.level-1);
                   
      #if 1
        const int m_vSize0 = m_CoarsenedBlock->getSize(0); 
        const int m_nElemsPerSlice = m_CoarsenedBlock->getNumberOfElementsPerSlice();   
        const int my_ix = s[0]-offset[0];
        const int bytes = (e[0]-s[0])*sizeof(ElementType);
        if (!bytes) return;

        for(int iz=s[2]; iz<e[2]; iz++)
        {
          const int my_izx = (iz-offset[2])*m_nElemsPerSlice + my_ix;
          #if 1
            for(int iy=s[1]; iy<e[1]; iy++)
            {
              #if 1
                char * ptrDest = (char*)&m_CoarsenedBlock->LinAccess(my_izx + (iy-offset[1])*m_vSize0);
                int CoarseEdge[3];

                CoarseEdge[0] = (code[0] == 0) ? 0 :   (   ( (info.index_(0)%2 ==0)&&(infoNei.index_(0)>info.index_(0)) ) || ( (info.index_(0)%2 ==1)&&(infoNei.index_(0)<info.index_(0)) )  )? 1:0  ;
                CoarseEdge[1] = (code[1] == 0) ? 0 :   (   ( (info.index_(1)%2 ==0)&&(infoNei.index_(1)>info.index_(1)) ) || ( (info.index_(1)%2 ==1)&&(infoNei.index_(1)<info.index_(1)) )  )? 1:0  ;
                CoarseEdge[2] = (code[2] == 0) ? 0 :   (   ( (info.index_(2)%2 ==0)&&(infoNei.index_(2)>info.index_(2)) ) || ( (info.index_(2)%2 ==1)&&(infoNei.index_(2)<info.index_(2)) )  )? 1:0  ;
                               
                const char * ptrSrc = (const char*)&b(s[0] + max(code[0],0)*nX/2 + (1-abs(code[0]))*base[0]*nX/2 - code[0]*nX  + CoarseEdge[0] *code[0]*nX/2    , 
                                                        iy + max(code[1],0)*nY/2 + (1-abs(code[1]))*base[1]*nY/2 - code[1]*nY  + CoarseEdge[1] *code[1]*nY/2    , 
                                                        iz + max(code[2],0)*nZ/2 + (1-abs(code[2]))*base[2]*nZ/2 - code[2]*nZ  + CoarseEdge[2] *code[2]*nZ/2    );
                memcpy2((char *)ptrDest, (char *)ptrSrc, bytes);
              #else
                for(int ix=s[0]; ix<e[0]; ix++)
                  m_cacheBlock->Access(ix-m_stencilStart[0], iy-m_stencilStart[1], iz-m_stencilStart[2]) =
                 (ElementType)b(ix - code[0]*BlockType::sizeX, iy - code[1]*BlockType::sizeY, iz - code[2]*BlockType::sizeZ);
              #endif
            }
          #else
                                  if ((e[1]-s[1]) % 4 != 0)
                                  {
                                      for(int iy=s[1]; iy<e[1]; iy++)
                                      {
                                          char * ptrDest = (char*)&m_cacheBlock->
                                                              LinAccess(my_izx + (iy-m_stencilStart[1])*m_vSize0);
              
                                          const char * ptrSrc = (const char*)&b(s[0] - code[0]*BlockType::sizeX, 
                                                                                  iy - code[1]*BlockType::sizeY, 
                                                                                  iz - code[2]*BlockType::sizeZ);
              
                                          const int cpybytes = (e[0]-s[0])*sizeof(ElementType);
              
                                          memcpy2((char *)ptrDest, (char *)ptrSrc, cpybytes);
                                      }
                                  }
                                  else
                                  {
                                    for(int iy=s[1]; iy<e[1]; iy+=4)
                                    {
                                        char * ptrDest0 = (char*)&m_cacheBlock->LinAccess(my_izx + 
                                                                             (iy+0-m_stencilStart[1])*m_vSize0);
                                        char * ptrDest1 = (char*)&m_cacheBlock->LinAccess(my_izx + 
                                                                             (iy+1-m_stencilStart[1])*m_vSize0);
                                        char * ptrDest2 = (char*)&m_cacheBlock->LinAccess(my_izx + 
                                                                             (iy+2-m_stencilStart[1])*m_vSize0);
                                        char * ptrDest3 = (char*)&m_cacheBlock->LinAccess(my_izx + 
                                                                             (iy+3-m_stencilStart[1])*m_vSize0);
            
            
                                        const char * ptrSrc0 = (const char*)&b(s[0] - code[0]*nX,iy + 0 - code[1]*nY,iz - code[2]*nZ);
                                        const char * ptrSrc1 = (const char*)&b(s[0] - code[0]*nX,iy + 1 - code[1]*nY,iz - code[2]*nZ);
                                        const char * ptrSrc2 = (const char*)&b(s[0] - code[0]*nX,iy + 2 - code[1]*nY,iz - code[2]*nZ);
                                        const char * ptrSrc3 = (const char*)&b(s[0] - code[0]*nX,iy + 3 - code[1]*nY,iz - code[2]*nZ);
            
                                        memcpy2((char *)ptrDest0, (char *)ptrSrc0, bytes);
                                        memcpy2((char *)ptrDest1, (char *)ptrSrc1, bytes);
                                        memcpy2((char *)ptrDest2, (char *)ptrSrc2, bytes);
                                        memcpy2((char *)ptrDest3, (char *)ptrSrc3, bytes);
                                    }
                                }
            #endif
                        }
      #else
                          const int off_x = - code[0]*nX + m_stencilStart[0];
                          const int off_y = - code[1]*nY + m_stencilStart[1];
                          const int off_z = - code[2]*nZ + m_stencilStart[2];
                          const int nbytes = (e[0]-s[0])*sizeof(ElementType);
                          #if 1
                              const int _iz0 = s[2] -m_stencilStart[2];
                              const int _iz1 = e[2] -m_stencilStart[2];
                              const int _iy0 = s[1] -m_stencilStart[1];
                              const int _iy1 = e[1] -m_stencilStart[1];
  
                              for(int iz=_iz0; iz<_iz1; iz++)
                                  for(int iy=_iy0; iy<_iy1; iy++)
                      
                          #else
                              for(int iz=s[2]-m_stencilStart[2]; iz<e[2]-m_stencilStart[2]; iz++)
                                  for(int iy=s[1]-m_stencilStart[1]; iy<e[1]-m_stencilStart[1]; iy++)
                          #endif
                                  {
                                      #if 0
                                          char * ptrDest = (char*)&m_cacheBlock->Access(s[0]-m_stencilStart[0], iy, iz);
                                          const char * ptrSrc = (const char*)&b(0 + off_x, iy + off_y, iz + off_z);
                                          memcpy2(ptrDest, ptrSrc, nbytes);
                                      #else
                                           for(int ix=s[0]-m_stencilStart[0]; ix<e[0]-m_stencilStart[0]; ix++)
                                              m_cacheBlock->Access(ix, iy, iz) = (ElementType)b(ix + off_x , iy + off_y, iz + off_z);
                                      #endif
                                  }
      #endif

    }

    //Improve the following 4 functions (2/4)
    void FillCoarseVersion(BlockInfo  info, const int * const code)
    {
      //If a neighboring block is on the same level it might need to average down some cells and 
      //use them to fill the coarsened version of this block. Those cells are needed to refine the
      //coarsened version and obtain ghosts from coarser neighbors (those cells are inside the
      //interpolation stencil for refinement).

      const Grid<BlockType,allocator>& grid = *m_refGrid;
       
      const int nX = BlockType::sizeX;
      const int nY = BlockType::sizeY;
      const int nZ = BlockType::sizeZ;


      const int eC[3] = { (m_stencilEnd[0])/2+ m_InterpStencilEnd[0] +1-1,
                          (m_stencilEnd[1])/2+ m_InterpStencilEnd[1] +1-1,
                          (m_stencilEnd[2])/2+ m_InterpStencilEnd[2] +1-1};

      const int sC[3] = {(m_stencilStart[0]-1)/2+ m_InterpStencilStart[0],
                         (m_stencilStart[1]-1)/2+ m_InterpStencilStart[1],
                         (m_stencilStart[2]-1)/2+ m_InterpStencilStart[2]};


      const int s[3] = { code[0]<1? (code[0]<0 ? sC[0]:0  ) : nX/2,
                         code[1]<1? (code[1]<0 ? sC[1]:0  ) : nY/2,
                         code[2]<1? (code[2]<0 ? sC[2]:0  ) : nZ/2};

      const int e[3] = { code[0]<1? (code[0]<0 ? 0    :nX/2 ) : nX/2+eC[0]-1,
                         code[1]<1? (code[1]<0 ? 0    :nY/2 ) : nY/2+eC[1]-1,
                         code[2]<1? (code[2]<0 ? 0    :nZ/2 ) : nZ/2+eC[2]-1};
                               
    

     if (!grid.avail(info.index_(0) + code[0], info.index_(1) + code[1], info.index_(2) + code[2],info.level_())) return;
     
      BlockType& b = grid(info.index_(0) + code[0], info.index_(1) + code[1], info.index_(2) + code[2],info.level_());

      const int bytes = (e[0]-s[0])*sizeof(ElementType);
      if (!bytes) return;

      const int m_vSize0         = m_CoarsenedBlock->getSize(0); 
      const int m_nElemsPerSlice = m_CoarsenedBlock->getNumberOfElementsPerSlice(); 
      const int my_ix = s[0]-sC[0];

      for(int iz=s[2]; iz<e[2]; iz++)
      {
        const int my_izx = (iz-sC[2])*m_nElemsPerSlice + my_ix;
        #if 1
          for(int iy=s[1]; iy<e[1]; iy++)
          {  
            if (code[1]==0 && code[2]==0 && iy >-m_InterpStencilStart[1] && iy <nY/2 -  m_InterpStencilEnd[1] && iz >-m_InterpStencilStart[2] && iz <nZ/2 -  m_InterpStencilEnd[2] ) continue;
 
            char * ptrDest = (char*)&m_CoarsenedBlock->LinAccess(my_izx + (iy-sC[1])*m_vSize0);

            const int XX =                s[0]+ max(code[0],0)*nX/2 - code[0]*nX + min(0,code[0])*(e[0]-s[0]);                
            const int YY =  2*(iy -s[1]) +s[1]+ max(code[1],0)*nY/2 - code[1]*nY + min(0,code[1])*(e[1]-s[1]);
            const int ZZ =  2*(iz -s[2]) +s[2]+ max(code[2],0)*nZ/2 - code[2]*nZ + min(0,code[2])*(e[2]-s[2]);                                      
                    
            const ElementType * ptrSrc_0 = (const ElementType *)&b( XX, YY  , ZZ   );
            const ElementType * ptrSrc_1 = (const ElementType *)&b( XX, YY  , ZZ +1);
            const ElementType * ptrSrc_2 = (const ElementType *)&b( XX, YY+1, ZZ   );
            const ElementType * ptrSrc_3 = (const ElementType *)&b( XX, YY+1, ZZ +1);

            //average down elements of block b to send to coarser neighbor
            ElementType * ptrSend = new ElementType[bytes / sizeof (ElementType)];                                   
            for (int ee=0; ee<  e[0]-s[0]; ee++)
            {
              //ptrSend[ee] = 0.125* ( * (ptrSrc_0 + 2*ee ) +   
              //                       * (ptrSrc_1 + 2*ee ) +   
              //                       * (ptrSrc_2 + 2*ee ) +   
              //                       * (ptrSrc_3 + 2*ee ) +   
              //                       * (ptrSrc_0 + 2*ee +1) + 
              //                       * (ptrSrc_1 + 2*ee +1) + 
              //                       * (ptrSrc_2 + 2*ee +1) + 
              //                       * (ptrSrc_3 + 2*ee +1) );   
              ptrSend[ee] = AverageDown(* (ptrSrc_0 + 2*ee   ),* (ptrSrc_1 + 2*ee   ),* (ptrSrc_2 + 2*ee   ),* (ptrSrc_3 + 2*ee   ),* (ptrSrc_0 + 2*ee +1),* (ptrSrc_1 + 2*ee +1),* (ptrSrc_2 + 2*ee +1),* (ptrSrc_3 + 2*ee +1));
            }                                 
                  
            memcpy2((char *)ptrDest, (char *)ptrSend, bytes );
            delete [] ptrSend;              
          }
          #else //'Vectorized' version 
                if ((e[1]-s[1]) % 4 != 0)
                {
                    for(int iy=s[1]; iy<e[1]; iy++)
                    {
                      char * ptrDest = (char*)&m_CoarsenedBlock->LinAccess(my_izx + (iy-sC[1])*m_vSize0);
  
                      const int XX =  (s[0] + max(code[0],0)*nX/2 - code[0]*nX + min(0,code[0])*(e[0]-s[0]));                
                      const int YY =  2*(iy -s[1]) +s[1]+ max(code[1],0)*nY/2 - code[1]*nY + min(0,code[1])*(e[1]-s[1]);
                      const int ZZ =  2*(iz -s[2]) +s[2]+ max(code[2],0)*nZ/2 - code[2]*nZ + min(0,code[2])*(e[2]-s[2]);                                      
                          
                      const ElementType * ptrSrc_0 = (const ElementType *)&b( XX, YY  , ZZ   );
                      const ElementType * ptrSrc_1 = (const ElementType *)&b( XX, YY  , ZZ +1);
                      const ElementType * ptrSrc_2 = (const ElementType *)&b( XX, YY+1, ZZ   );
                      const ElementType * ptrSrc_3 = (const ElementType *)&b( XX, YY+1, ZZ +1);
  
  
                      //average down elements of block b to send to coarser neighbor
                      ElementType * ptrSend = new ElementType[bytes / sizeof (ElementType)];                                   
                      for (int ee=0; ee<  e[0]-s[0]; ee++)
                      {
                          ptrSend[ee] = 0.125* ( * (ptrSrc_0 + 2*ee ) +   
                                                 * (ptrSrc_1 + 2*ee ) +   
                                                 * (ptrSrc_2 + 2*ee ) +   
                                                 * (ptrSrc_3 + 2*ee ) +   
                                                 * (ptrSrc_0 + 2*ee +1) + 
                                                 * (ptrSrc_1 + 2*ee +1) + 
                                                 * (ptrSrc_2 + 2*ee +1) + 
                                                 * (ptrSrc_3 + 2*ee +1) );                         
                      }                                 
  
                    memcpy2((char *)ptrDest, (char *)ptrSend, bytes);
                      delete [] ptrSend;
                    }
                }
                else
                {
                    for(int iy=s[1]; iy<e[1]; iy+=4)
                    {
                        char * ptrDest0 = (char*)&m_CoarsenedBlock->LinAccess(my_izx + (iy+0-sC[1])*m_vSize0);
                        char * ptrDest1 = (char*)&m_CoarsenedBlock->LinAccess(my_izx + (iy+1-sC[1])*m_vSize0);
                        char * ptrDest2 = (char*)&m_CoarsenedBlock->LinAccess(my_izx + (iy+2-sC[1])*m_vSize0);
                        char * ptrDest3 = (char*)&m_CoarsenedBlock->LinAccess(my_izx + (iy+3-sC[1])*m_vSize0);
  
                      const int XX =  (s[0] + max(code[0],0)*nX/2 - code[0]*nX + min(0,code[0])*(e[0]-s[0]));                
                          
                      const int YY0 =  2*(iy+0 -s[1]) +s[1]+ max(code[1],0)*nY/2 - code[1]*nY + min(0,code[1])*(e[1]-s[1]);
                      const int YY1 =  2*(iy+1 -s[1]) +s[1]+ max(code[1],0)*nY/2 - code[1]*nY + min(0,code[1])*(e[1]-s[1]);
                      const int YY2 =  2*(iy+2 -s[1]) +s[1]+ max(code[1],0)*nY/2 - code[1]*nY + min(0,code[1])*(e[1]-s[1]);
                      const int YY3 =  2*(iy+3 -s[1]) +s[1]+ max(code[1],0)*nY/2 - code[1]*nY + min(0,code[1])*(e[1]-s[1]);


                        const int ZZ =  2*(iz -s[2]) +s[2]+ max(code[2],0)*nZ/2 - code[2]*nZ + min(0,code[2])*(e[2]-s[2]);                                      
                          

                      const ElementType * ptrSrc_00 = (const ElementType *)&b( XX, YY0 , ZZ   );
                      const ElementType * ptrSrc_01 = (const ElementType *)&b( XX, YY1 , ZZ   );
                      const ElementType * ptrSrc_02 = (const ElementType *)&b( XX, YY2 , ZZ   );
                      const ElementType * ptrSrc_03 = (const ElementType *)&b( XX, YY3 , ZZ   );   

                        const ElementType * ptrSrc_10 = (const ElementType *)&b( XX, YY0 , ZZ +1);
                        const ElementType * ptrSrc_11 = (const ElementType *)&b( XX, YY1 , ZZ +1);
                        const ElementType * ptrSrc_12 = (const ElementType *)&b( XX, YY2 , ZZ +1);
                        const ElementType * ptrSrc_13 = (const ElementType *)&b( XX, YY3 , ZZ +1);
                          
                        const ElementType * ptrSrc_20 = (const ElementType *)&b( XX, YY0+1, ZZ   );
                        const ElementType * ptrSrc_21 = (const ElementType *)&b( XX, YY1+1, ZZ   );
                        const ElementType * ptrSrc_22 = (const ElementType *)&b( XX, YY2+1, ZZ   );
                        const ElementType * ptrSrc_23 = (const ElementType *)&b( XX, YY3+1, ZZ   );

                        const ElementType * ptrSrc_30 = (const ElementType *)&b( XX, YY0+1, ZZ +1);
                        const ElementType * ptrSrc_31 = (const ElementType *)&b( XX, YY1+1, ZZ +1);
                        const ElementType * ptrSrc_32 = (const ElementType *)&b( XX, YY2+1, ZZ +1);
                        const ElementType * ptrSrc_33 = (const ElementType *)&b( XX, YY3+1, ZZ +1);
  
    
                      //average down elements of block b to send to coarser neighbor
                        ElementType * ptrSend0 = new ElementType[bytes / sizeof (ElementType)];
                        ElementType * ptrSend1 = new ElementType[bytes / sizeof (ElementType)];
                        ElementType * ptrSend2 = new ElementType[bytes / sizeof (ElementType)];
                        ElementType * ptrSend3 = new ElementType[bytes / sizeof (ElementType)];                                   
                      for (int ee=0; ee<  e[0]-s[0]; ee++)
                      {
                            ptrSend0[ee] = 0.125* ( * (ptrSrc_00 + 2*ee ) +   
                                                    * (ptrSrc_10 + 2*ee ) +   
                                                    * (ptrSrc_20 + 2*ee ) +   
                                                    * (ptrSrc_30 + 2*ee ) +   
                                                    * (ptrSrc_00 + 2*ee +1) + 
                                                    * (ptrSrc_10 + 2*ee +1) + 
                                                    * (ptrSrc_20 + 2*ee +1) + 
                                                    * (ptrSrc_30 + 2*ee +1) ); 
                            ptrSend1[ee] = 0.125* ( * (ptrSrc_01 + 2*ee ) +   
                                                    * (ptrSrc_11 + 2*ee ) +   
                                                    * (ptrSrc_21 + 2*ee ) +   
                                                    * (ptrSrc_31 + 2*ee ) +   
                                                    * (ptrSrc_01 + 2*ee +1) + 
                                                    * (ptrSrc_11 + 2*ee +1) + 
                                                    * (ptrSrc_21 + 2*ee +1) + 
                                                    * (ptrSrc_31 + 2*ee +1) ); 
                            ptrSend2[ee] = 0.125* ( * (ptrSrc_02 + 2*ee ) +   
                                                    * (ptrSrc_12 + 2*ee ) +   
                                                    * (ptrSrc_22 + 2*ee ) +   
                                                    * (ptrSrc_32 + 2*ee ) +   
                                                    * (ptrSrc_02 + 2*ee +1) + 
                                                    * (ptrSrc_12 + 2*ee +1) + 
                                                    * (ptrSrc_22 + 2*ee +1) + 
                                                    * (ptrSrc_32 + 2*ee +1) ); 
                            ptrSend3[ee] = 0.125* ( * (ptrSrc_03 + 2*ee ) +   
                                                    * (ptrSrc_13 + 2*ee ) +   
                                                    * (ptrSrc_23 + 2*ee ) +   
                                                    * (ptrSrc_33 + 2*ee ) +   
                                                    * (ptrSrc_03 + 2*ee +1) + 
                                                    * (ptrSrc_13 + 2*ee +1) + 
                                                    * (ptrSrc_23 + 2*ee +1) + 
                                                    * (ptrSrc_33 + 2*ee +1) );                                                 
                      }                                 

                      memcpy2((char *)ptrDest0, (char *)ptrSend0, bytes);
            memcpy2((char *)ptrDest1, (char *)ptrSend1, bytes);
                        memcpy2((char *)ptrDest2, (char *)ptrSend2, bytes);
                        memcpy2((char *)ptrDest3, (char *)ptrSend3, bytes);
                         
                        delete [] ptrSend0;
                        delete [] ptrSend1;
                        delete [] ptrSend2;
                        delete [] ptrSend3;
                    }
                }
            #endif
        }
    }

    //Improve the following 4 functions (3/4) 
    void CoarseFineInterpolation(BlockInfo info)
    {
      const Grid<BlockType,allocator>& grid = *m_refGrid;

      const int nX = BlockType::sizeX;
      const int nY = BlockType::sizeY;
      const int nZ = BlockType::sizeZ;
    
      const bool xperiodic = is_xperiodic();
      const bool yperiodic = is_yperiodic();
      const bool zperiodic = is_zperiodic();
      std::array <int,3> blocksPerDim = grid.getMaxBlocks();
      int aux = pow(2,info.level_());
      const bool xskin = info.index_(0)==0 || info.index_(0)==blocksPerDim[0]*aux-1;
      const bool yskin = info.index_(1)==0 || info.index_(1)==blocksPerDim[1]*aux-1;
      const bool zskin = info.index_(2)==0 || info.index_(2)==blocksPerDim[2]*aux-1;
      const int xskip  = info.index_(0)==0 ? -1 : 1;
      const int yskip  = info.index_(1)==0 ? -1 : 1;
      const int zskip  = info.index_(2)==0 ? -1 : 1;

      for(int icode=0; icode<27; icode++)
      {
        if (icode == 1*1 + 3*1 + 9*1) continue;
        const int code[3] = { icode%3-1, (icode/3)%3-1, (icode/9)%3-1};
            
        if (!xperiodic && code[0] == xskip && xskin) continue;
        if (!yperiodic && code[1] == yskip && yskin) continue;
        if (!zperiodic && code[2] == zskip && zskin) continue;
        if (!istensorial && abs(code[0])+abs(code[1])+abs(code[2])>1) continue;
            
        //mike : s and e correspond to start and end of this lab's cells that are filled by neighbors
        const int s[3] = { code[0]<1? (code[0]<0 ? m_stencilStart[0]:0 ) : nX,
                           code[1]<1? (code[1]<0 ? m_stencilStart[1]:0 ) : nY,
                           code[2]<1? (code[2]<0 ? m_stencilStart[2]:0 ) : nZ};
        const int e[3] = { code[0]<1? (code[0]<0 ? 0                :nX ): nX+m_stencilEnd[0]-1,
                           code[1]<1? (code[1]<0 ? 0                :nY ): nY+m_stencilEnd[1]-1,
                           code[2]<1? (code[2]<0 ? 0                :nZ ): nZ+m_stencilEnd[2]-1};

        const int offset[3] =   {(m_stencilStart[0]-1)/2+ m_InterpStencilStart[0],
                                 (m_stencilStart[1]-1)/2+ m_InterpStencilStart[1],
                                 (m_stencilStart[2]-1)/2+ m_InterpStencilStart[2]};

        const int sC[3] = {
        code[0]<1? (code[0]<0 ? ((m_stencilStart[0]-1)/2+ 0*m_InterpStencilStart[0]) :0 ) : nX/2,
        code[1]<1? (code[1]<0 ? ((m_stencilStart[1]-1)/2+ 0*m_InterpStencilStart[1]) :0 ) : nY/2,
        code[2]<1? (code[2]<0 ? ((m_stencilStart[2]-1)/2+ 0*m_InterpStencilStart[2]) :0 ) : nZ/2 };

        // /*comment to silence warnings*/const int eC[3] = {
        // /*comment to silence warnings*/code[0]<1? (code[0]<0 ? 0:nX/2 ) : nX/2+(m_stencilEnd[0])/2 + 1 + 0*(m_InterpStencilEnd[0] -1),
        // /*comment to silence warnings*/code[1]<1? (code[1]<0 ? 0:nY/2 ) : nY/2+(m_stencilEnd[1])/2 + 1 + 0*(m_InterpStencilEnd[1] -1),
        // /*comment to silence warnings*/code[2]<1? (code[2]<0 ? 0:nZ/2 ) : nZ/2+(m_stencilEnd[2])/2 + 1 + 0*(m_InterpStencilEnd[2] -1)};
              
            
        //mike: get neighbor on same level of resolution
        BlockInfo infoNei = grid.getBlockInfoAll(info.level,info.Znei_(code[0],code[1],code[2]));
            
            {//mike: this check should be done by grid.avail
              //  int rank;
              //  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
              //  if (infoNei.myRank != rank) continue;
            }
            
        if (infoNei.TreePos == CheckCoarser)
        {
          #if 1
            const int bytes = (e[0]-s[0])*sizeof(ElementType);
            if (!bytes) return;
        
            // /*comment to silence warnings*/const int m_vSize0 = m_cacheBlock->getSize(0); 
            // /*comment to silence warnings*/const int m_nElemsPerSlice = m_cacheBlock->getNumberOfElementsPerSlice(); 
            // /*comment to silence warnings*/const int my_ix = s[0]-m_stencilStart[0];
        
            for(int iz=s[2]; iz<e[2]; iz++)
            {
              // /*comment to silence warnings*/const int my_izx = (iz-m_stencilStart[2])*m_nElemsPerSlice + my_ix;
              #if 1
                for(int iy=s[1]; iy<e[1]; iy++)
                {
                  for(int ix=s[0]; ix<e[0]; ix++)
                  {
                    int XX = (ix - s[0] -min(0,code[0])*( (e[0]-s[0])%2 ))/2+ sC[0]         ;
                    int YY = (iy - s[1] -min(0,code[1])*( (e[1]-s[1])%2 ))/2+ sC[1]         ;
                    int ZZ = (iz - s[2] -min(0,code[2])*( (e[2]-s[2])%2 ))/2+ sC[2]         ;

                    ElementType  RefinedValue;
                    
                    ElementType Test [3][3][3];
                    for (int i=0;i<3;i++)
                    for (int j=0;j<3;j++)
                    for (int k=0;k<3;k++) 
                      Test[i][j][k] = m_CoarsenedBlock->Access(XX-1+i-offset[0],YY-1+j-offset[1],ZZ-1+k-offset[2]);
                    

                    TestInterp(Test,RefinedValue,abs(ix-s[0]-min(0,code[0])*( (e[0]-s[0])%2 ))%2 ,
                                                 abs(iy-s[1]-min(0,code[1])*( (e[1]-s[1])%2 ))%2 ,
                                                 abs(iz-s[2]-min(0,code[2])*( (e[2]-s[2])%2 ))%2 );
         
                    //   ElementType Test [5][5][5];
                    //   for (int i=0;i<5;i++)
                    //   for (int j=0;j<5;j++)
                    //   for (int k=0;k<5;k++) 
                    //       Test[i][j][k] = m_CoarsenedBlock->Access(XX-2+i-offset[0],YY-2+j-offset[1],ZZ-2+k-offset[2]);
                    //   TestInterp(Test,RefinedValue,abs(ix-s[0]-min(0,code[0])*( (e[0]-s[0])%2 ))%2 ,
                    //                                abs(iy-s[1]-min(0,code[1])*( (e[1]-s[1])%2 ))%2 ,
                    //                                abs(iz-s[2]-min(0,code[2])*( (e[2]-s[2])%2 ))%2 );

                    m_cacheBlock->Access(ix-m_stencilStart[0]    , iy-m_stencilStart[1]  , iz-m_stencilStart[2]  ) = RefinedValue;
                  }       
                }
              #else
                    if ((e[1]-s[1]) % 4 != 0)
                    {
                        for(int iy=s[1]; iy<e[1]; iy++)
                        {
                            char * ptrDest = (char*)&m_cacheBlock->LinAccess(my_izx + (iy-m_stencilStart[1])*m_vSize0);
                            const char * ptrSrc = (const char*)&b(s[0] - code[0]*nX, iy - code[1]*nY, iz - code[2]*nZ);
                            const int cpybytes = (e[0]-s[0])*sizeof(ElementType);
                            memcpy2((char *)ptrDest, (char *)ptrSrc, cpybytes);
                        }
                    }
                    else
                    {
                        for(int iy=s[1]; iy<e[1]; iy+=4)
                        {
                            char * ptrDest0 = (char*)&m_cacheBlock->LinAccess(my_izx + 
                                                                      (iy+0-m_stencilStart[1])*m_vSize0);
                                 char * ptrDest1 = (char*)&m_cacheBlock->LinAccess(my_izx + 
                                                                      (iy+1-m_stencilStart[1])*m_vSize0);
                                 char * ptrDest2 = (char*)&m_cacheBlock->LinAccess(my_izx + 
                                                                      (iy+2-m_stencilStart[1])*m_vSize0);
                                 char * ptrDest3 = (char*)&m_cacheBlock->LinAccess(my_izx + 
                                                                      (iy+3-m_stencilStart[1])*m_vSize0);
     
     
                                 const char * ptrSrc0 = (const char*)&b(s[0] - code[0]*nX,iy + 0 - code[1]*nY,iz - code[2]*nZ);
                                 const char * ptrSrc1 = (const char*)&b(s[0] - code[0]*nX,iy + 1 - code[1]*nY,iz - code[2]*nZ);
                                 const char * ptrSrc2 = (const char*)&b(s[0] - code[0]*nX,iy + 2 - code[1]*nY,iz - code[2]*nZ);
                                 const char * ptrSrc3 = (const char*)&b(s[0] - code[0]*nX,iy + 3 - code[1]*nY,iz - code[2]*nZ);
     
                                 memcpy2((char *)ptrDest0, (char *)ptrSrc0, bytes);
                                 memcpy2((char *)ptrDest1, (char *)ptrSrc1, bytes);
                                 memcpy2((char *)ptrDest2, (char *)ptrSrc2, bytes);
                                 memcpy2((char *)ptrDest3, (char *)ptrSrc3, bytes);
                             }
                    }
              #endif
                    }
          #else
                           const int off_x = - code[0]*nX + m_stencilStart[0];
                           const int off_y = - code[1]*nY + m_stencilStart[1];
                           const int off_z = - code[2]*nZ + m_stencilStart[2];
                           const int nbytes = (e[0]-s[0])*sizeof(ElementType);
                           #if 1
                               const int _iz0 = s[2] -m_stencilStart[2];
                               const int _iz1 = e[2] -m_stencilStart[2];
                               const int _iy0 = s[1] -m_stencilStart[1];
                               const int _iy1 = e[1] -m_stencilStart[1];
                               for(int iz=_iz0; iz<_iz1; iz++)
                                   for(int iy=_iy0; iy<_iy1; iy++)    
                           #else
                               for(int iz=s[2]-m_stencilStart[2]; iz<e[2]-m_stencilStart[2]; iz++)
                                   for(int iy=s[1]-m_stencilStart[1]; iy<e[1]-m_stencilStart[1]; iy++)
                           #endif
                           {
                               #if 0
                                   char * ptrDest = (char*)&m_cacheBlock->Access(s[0]-m_stencilStart[0], iy, iz);
                                   const char * ptrSrc = (const char*)&b(0 + off_x, iy + off_y, iz + off_z);
                                   memcpy2(ptrDest, ptrSrc, nbytes);
                               #else
                                    for(int ix=s[0]-m_stencilStart[0]; ix<e[0]-m_stencilStart[0]; ix++)
                                       m_cacheBlock->Access(ix, iy, iz) = (ElementType)b(ix + off_x , iy + off_y, iz + off_z);
                               #endif
                           }
          #endif
        }
      }
    }


    //Improve the following 4 functions (4/4)
    void TestInterp(ElementType C[3][3][3], ElementType & R, int x, int y, int z)
   // void TestInterp(ElementType C[5][5][5], ElementType & R, int x, int y, int z)  
    {   
        //simple linear for now
        ElementType dudx = 0.5*(C[2][1][1] - C[0][1][1]);
        ElementType dudy = 0.5*(C[1][2][1] - C[1][0][1]);
        ElementType dudz = 0.5*(C[1][1][2] - C[1][1][0]);
   

       #if 0
        // ElementType dudx = (2./3)*(C[3][2][2] - C[1][2][2]) - (1./12)*(C[4][2][2] - C[0][2][2]);
        // ElementType dudy = (2./3)*(C[2][3][2] - C[2][1][2]) - (1./12)*(C[2][4][2] - C[2][0][2]);
        // ElementType dudz = (2./3)*(C[2][2][3] - C[2][2][1]) - (1./12)*(C[2][2][4] - C[2][2][0]);
        // R = C[2][2][2] + (2*x-1)*0.25*dudx + (2*y-1)*0.25*dudy + (2*z-1)*0.25*dudz;
       #endif
         R = C[1][1][1] + (2*x-1)*0.25*dudx + (2*y-1)*0.25*dudy + (2*z-1)*0.25*dudz;
         

         //R.dummy = 1;
         //assert (!isnan(R.alpha1rho1));
         //assert (!isnan(R.alpha2rho2));
         //assert (!isnan(R.ru));
         //assert (!isnan(R.rv));
         //assert (!isnan(R.rw));
         //assert (!isnan(R.energy));
         //assert (!isnan(R.alpha2));
         //assert (!isnan(R.dummy));
         //assert (abs   (R.alpha1rho1) < 1e40);
         //assert (abs   (R.alpha2rho2) < 1e40);
         //assert (abs   (R.ru)         < 1e40);
         //assert (abs   (R.rv)         < 1e40);
         //assert (abs   (R.rw)         < 1e40);
         //assert (abs   (R.energy)     < 1e40);
         //assert (abs   (R.alpha2)     < 1e40);
         //assert (abs   (R.dummy)      < 1e40);

        
        #if 0
         	const int Nweno = 3;
          	ElementType Lines [Nweno][Nweno][2];
   			ElementType Planes[Nweno][4];
   			ElementType Ref          [8]; 

			for (int i2= -Nweno/2 ; i2<= Nweno/2; i2++)
			{
				for (int i1= -Nweno/2 ; i1<= Nweno/2; i1++)
        		{
        			Kernel_1D(C[0][i1+Nweno/2][i2+Nweno/2],
        				      C[1][i1+Nweno/2][i2+Nweno/2],
        				      C[2][i1+Nweno/2][i2+Nweno/2],
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
		#endif

    }








    /**
     * Get a single element from the block.
     * stencil_start and stencil_end refer to the values passed in BlockLab::prepare().
     *
     * @param ix    Index in x-direction (stencil_start[0] <= ix < BlockType::sizeX + stencil_end[0] - 1).
     * @param iy    Index in y-direction (stencil_start[1] <= iy < BlockType::sizeY + stencil_end[1] - 1).
     * @param iz    Index in z-direction (stencil_start[2] <= iz < BlockType::sizeZ + stencil_end[2] - 1).
     */
    ElementType& operator()(int ix, int iy=0, int iz=0)
    {
    #ifndef NDEBUG
          assert(m_state == eMRAGBlockLab_Loaded);
  
          const int nX = m_cacheBlock->getSize()[0];
          const int nY = m_cacheBlock->getSize()[1];
          const int nZ = m_cacheBlock->getSize()[2];
  
          assert(ix-m_stencilStart[0]>=0 && ix-m_stencilStart[0]<nX);
          assert(iy-m_stencilStart[1]>=0 && iy-m_stencilStart[1]<nY);
          assert(iz-m_stencilStart[2]>=0 && iz-m_stencilStart[2]<nZ);
    #endif
        return m_cacheBlock->Access(ix-m_stencilStart[0], iy-m_stencilStart[1], iz-m_stencilStart[2]);
    }

    /** Just as BlockLab::operator() but returning a const. */
    const ElementType& read(int ix, int iy=0, int iz=0) const
    {
    #ifndef NDEBUG
          assert(m_state == eMRAGBlockLab_Loaded);
  
          const int nX = m_cacheBlock->getSize()[0];
          const int nY = m_cacheBlock->getSize()[1];
          const int nZ = m_cacheBlock->getSize()[2];
  
          assert(ix-m_stencilStart[0]>=0 && ix-m_stencilStart[0]<nX);
          assert(iy-m_stencilStart[1]>=0 && iy-m_stencilStart[1]<nY);
          assert(iz-m_stencilStart[2]>=0 && iz-m_stencilStart[2]<nZ);
    #endif

        return m_cacheBlock->Access(ix-m_stencilStart[0], iy-m_stencilStart[1], iz-m_stencilStart[2]);
    }

    void release()
    {
        _release(m_cacheBlock);
        _release(m_CoarsenedBlock);
    }

private:
    BlockLab(const BlockLab&) = delete;
    BlockLab& operator=(const BlockLab&) = delete;
};



}//namespace AMR_CUBISM
