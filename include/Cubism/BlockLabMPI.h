#pragma once


#include "GridMPI.h"
#include "AMR_SynchronizerMPI.h"

CUBISM_NAMESPACE_BEGIN

template<typename MyBlockLab>
class BlockLabMPI : public MyBlockLab
{
public:
    typedef typename MyBlockLab::Real Real;

private:
    typedef typename MyBlockLab::BlockType BlockType; 
    typedef SynchronizerMPI_AMR<Real> SynchronizerMPIType;   
    const SynchronizerMPIType * refSynchronizerMPI;


public:
    template< typename TGrid >
    void prepare(GridMPI<TGrid>& grid, const SynchronizerMPIType& synchronizer)
    {
        refSynchronizerMPI = &synchronizer;
        StencilInfo stencil = refSynchronizerMPI->getstencil();
        assert(stencil.isvalid());
        MyBlockLab::prepare(grid, stencil.sx, stencil.ex, stencil.sy, stencil.ey, stencil.sz, stencil.ez, stencil.tensorial);
    }




    bool UseCoarseStencil (Interface & f)
    {
        //return false;
        BlockInfo & a = *f.infos[0];
        BlockInfo & b = *f.infos[1];


        if(a.level != b.level) return false;

        int imin [3];
        int imax [3];
     
        for (int d=0; d<3; d++)
            if (a.index[d] == b.index[d])
            {
                imin[d] = a.index[d] - 1;
                imax[d] = a.index[d] + 1;
            }
            else
            {
                imin[d] = min(a.index[d],b.index[d]);
                imax[d] = max(a.index[d],b.index[d]);
            }
    
        bool retval = false;

        for (int i2 = imin[2]; i2 <= imax[2]; i2++)
        for (int i1 = imin[1]; i1 <= imax[1]; i1++)
        for (int i0 = imin[0]; i0 <= imax[0]; i0++)
        {
            int n = MyBlockLab::m_refGrid->getZforward(a.level,i0,i1,i2);
            if ( (MyBlockLab::m_refGrid->getBlockInfoAll(a.level,n)).TreePos == CheckCoarser )
            {
                retval = true;
                break;
            }
        }

        return retval;
    }










    void load(const BlockInfo& info, const Real t=0, const bool applybc=true)
    {
        MyBlockLab::load(info, t, applybc);     
        assert(refSynchronizerMPI != NULL);

   
    	int rank;
    	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    	typedef typename MyBlockLab::ElementType ET;

    	const int nX = BlockType::sizeX;
        const int nY = BlockType::sizeY;
        const int nZ = BlockType::sizeZ;
        
        std::array <int,3> blocksPerDim = MyBlockLab::m_refGrid->getMaxBlocks();
        int aux = pow(2,info.level);
        const bool xperiodic = MyBlockLab::is_xperiodic();
        const bool yperiodic = MyBlockLab::is_yperiodic();
        const bool zperiodic = MyBlockLab::is_zperiodic();  
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
            //if (!istensorial && abs(code[0])+abs(code[1])+abs(code[2])>1) continue;
            BlockInfo  infoNei = MyBlockLab::m_refGrid->getBlockInfoAll(info.level,info.Znei_(code[0],code[1],code[2]));   



            const int s[3] = { code[0]<1? (code[0]<0 ? MyBlockLab::m_stencilStart[0]:0 ):nX,
                               code[1]<1? (code[1]<0 ? MyBlockLab::m_stencilStart[1]:0 ):nY,
                               code[2]<1? (code[2]<0 ? MyBlockLab::m_stencilStart[2]:0 ):nZ};
            const int e[3] = { code[0]<1? (code[0]<0 ? 0                :nX):nX+ MyBlockLab::m_stencilEnd[0]-1,
                               code[1]<1? (code[1]<0 ? 0                :nY):nY+ MyBlockLab::m_stencilEnd[1]-1,
                               code[2]<1? (code[2]<0 ? 0                :nZ):nZ+ MyBlockLab::m_stencilEnd[2]-1};

            if (infoNei.TreePos == Exists && infoNei.myrank != rank)
            {
            	int icode2 = (-code[0]+1) + (-code[1]+1)*3 + (-code[2]+1)*9;
                Real * dst = &MyBlockLab::m_cacheBlock->Access(s[0]-MyBlockLab::m_stencilStart[0], 
                	                                           s[1]-MyBlockLab::m_stencilStart[1], 
                	                                           s[2]-MyBlockLab::m_stencilStart[2]).alpha1rho1;
               
                refSynchronizerMPI->fetch(infoNei,info,icode2,icode,&dst[0],
                                          MyBlockLab::m_cacheBlock->getSize(0),
                                          MyBlockLab::m_cacheBlock->getSize(1),
                                          MyBlockLab::m_cacheBlock->getSize(2),sizeof(ET)/sizeof(Real));
                           
                BlockInfo i0 = infoNei;
                BlockInfo i1 = info;

                Interface f(i0,i1,icode2,icode);
                //if (refSynchronizerMPI->UseCoarseStencil(f))
                if (UseCoarseStencil(f))
                {
                    const int sC[3] = {(MyBlockLab::m_stencilStart[0]-1)/2+ MyBlockLab::m_InterpStencilStart[0],
                                       (MyBlockLab::m_stencilStart[1]-1)/2+ MyBlockLab::m_InterpStencilStart[1],
                                       (MyBlockLab::m_stencilStart[2]-1)/2+ MyBlockLab::m_InterpStencilStart[2]};
                   

                   const int s1[3] = { code[0]<1? (code[0]<0 ? sC[0]:0  ) : nX/2,
                                      code[1]<1? (code[1]<0 ? sC[1]:0  ) : nY/2,
                                      code[2]<1? (code[2]<0 ? sC[2]:0  ) : nZ/2};


                    const int m_vSize0         = MyBlockLab::m_CoarsenedBlock->getSize(0); 
                    const int m_nElemsPerSlice = MyBlockLab::m_CoarsenedBlock->getNumberOfElementsPerSlice(); 
                    const int my_ix = s1[0]-sC[0];
                    const int my_izx = (s1[2]-sC[2])*m_nElemsPerSlice + my_ix;

              	    Real * dst1 = &MyBlockLab::m_CoarsenedBlock->LinAccess(my_izx + (s1[1]-sC[1])*m_vSize0).alpha1rho1;

              	    refSynchronizerMPI->fetch(infoNei,info,icode2,icode,&dst1[0],
                                              MyBlockLab::m_CoarsenedBlock->getSize(0),
                                              MyBlockLab::m_CoarsenedBlock->getSize(1),
                                              MyBlockLab::m_CoarsenedBlock->getSize(2),sizeof(ET)/sizeof(Real),true);
                }

            }
              else if (infoNei.TreePos == CheckCoarser)
              {
                int nCoarse = MyBlockLab::m_refGrid->getZforward(infoNei.level-1,infoNei.index[0]/2,infoNei.index[1]/2,infoNei.index[2]/2);
                BlockInfo  infoNeiCoarser = MyBlockLab::m_refGrid->getBlockInfoAll(infoNei.level-1,nCoarse);

                if (infoNeiCoarser.myrank != rank)
                {
                    const int sC[3] = {code[0]<1? (code[0]<0 ? ((MyBlockLab::m_stencilStart[0]-1)/2+ MyBlockLab::m_InterpStencilStart[0]) :0 ) : nX/2,
                                       code[1]<1? (code[1]<0 ? ((MyBlockLab::m_stencilStart[1]-1)/2+ MyBlockLab::m_InterpStencilStart[1]) :0 ) : nY/2,
                                       code[2]<1? (code[2]<0 ? ((MyBlockLab::m_stencilStart[2]-1)/2+ MyBlockLab::m_InterpStencilStart[2]) :0 ) : nZ/2 };
              
                    //const int eC[3] = {code[0]<1? (code[0]<0 ? 0:nX/2 ) : nX/2+(MyBlockLab::m_stencilEnd[0])/2+ MyBlockLab::m_InterpStencilEnd[0] -1,
                    //                   code[1]<1? (code[1]<0 ? 0:nY/2 ) : nY/2+(MyBlockLab::m_stencilEnd[1])/2+ MyBlockLab::m_InterpStencilEnd[1] -1,
                    //                   code[2]<1? (code[2]<0 ? 0:nZ/2 ) : nZ/2+(MyBlockLab::m_stencilEnd[2])/2+ MyBlockLab::m_InterpStencilEnd[2] -1};
              
                    const int offset[3] = {(MyBlockLab::m_stencilStart[0]-1)/2+ MyBlockLab::m_InterpStencilStart[0],
                                           (MyBlockLab::m_stencilStart[1]-1)/2+ MyBlockLab::m_InterpStencilStart[1],
                                           (MyBlockLab::m_stencilStart[2]-1)/2+ MyBlockLab::m_InterpStencilStart[2]};

                   
                    const int m_vSize0 = MyBlockLab::m_CoarsenedBlock->getSize(0); 
                    const int m_nElemsPerSlice = MyBlockLab::m_CoarsenedBlock->getNumberOfElementsPerSlice();   
                    const int my_ix = sC[0]-offset[0];
                    const int my_izx = (sC[2]-offset[2])*m_nElemsPerSlice + my_ix;

                    Real * dst = & MyBlockLab::m_CoarsenedBlock->LinAccess(my_izx + (sC[1]-offset[1])*m_vSize0).alpha1rho1;

                    int icode2 = (-code[0]+1) + (-code[1]+1)*3 + (-code[2]+1)*9;
                    
                    refSynchronizerMPI->fetch(infoNeiCoarser,info,icode2,icode,&dst[0],  
                                              MyBlockLab::m_CoarsenedBlock->getSize(0),
                                              MyBlockLab::m_CoarsenedBlock->getSize(1),
                                              MyBlockLab::m_CoarsenedBlock->getSize(2),sizeof(ET)/sizeof(Real));
                }
              }
              else if (infoNei.TreePos == CheckFiner)
              {
                int Bstep = 1; //face
                if      ((abs(code[0])+abs(code[1])+abs(code[2])==2 )) Bstep = 3; //edge
                else if ((abs(code[0])+abs(code[1])+abs(code[2])==3 )) Bstep = 4; //corner
               
                for (int B = 0 ; B <= 3 ; B += Bstep) //loop over blocks that make up face/edge/corner (respectively 4,2 or 1 blocks)
                {
                    const int aux1 = (abs(code[0])==1) ? (B%2) : (B/2) ;
                    int nFine = MyBlockLab::m_refGrid->getZforward(infoNei.level+1,2*info.index[0] + max(code[0],0) +code[0]  + (B%2)*max(0, 1 - abs(code[0])),
                                                                  2*info.index[1] + max(code[1],0) +code[1]  +  aux1 *max(0, 1 - abs(code[1])),
                                                                  2*info.index[2] + max(code[2],0) +code[2]  + (B/2)*max(0, 1 - abs(code[2])));
                    BlockInfo infoNeiFiner = MyBlockLab::m_refGrid->getBlockInfoAll(infoNei.level+1,nFine);
             
                    if (infoNeiFiner.myrank != rank)
                    {

                    	int iz = s[2];
                    	int iy = s[1];

                    	const int my_ix =  abs(code[0])*(s[0]-MyBlockLab::m_stencilStart[0]) + (1-abs(code[0]) )*(  s[0]  -MyBlockLab::m_stencilStart[0] + (B%2)*(e[0]-s[0])/2);                                                
     
                    	const int m_vSize0         = MyBlockLab::m_cacheBlock->getSize(0);
                 
                        const int m_nElemsPerSlice = MyBlockLab::m_cacheBlock->getNumberOfElementsPerSlice();  
                     
                        const int my_izx = ( abs(code[2])*(iz-MyBlockLab::m_stencilStart[2]) + (1-abs(code[2]) )*(iz/2-MyBlockLab::m_stencilStart[2] + 
                                             	(B/2)*(e[2]-s[2])/2)  )*m_nElemsPerSlice + my_ix;
       
                        Real * dst = & MyBlockLab::m_cacheBlock->LinAccess(
                        	my_izx + ( abs(code[1])*(iy-MyBlockLab::m_stencilStart[1]) + (1-abs(code[1]) )*(iy/2-MyBlockLab::m_stencilStart[1] + aux1*(e[1]-s[1])/2)  )*m_vSize0).alpha1rho1;



                        int icode2 = (-code[0]+1) + (-code[1]+1)*3 + (-code[2]+1)*9;
                        refSynchronizerMPI->fetch(infoNeiFiner,info,icode2,icode,&dst[0],
                                                  MyBlockLab::m_cacheBlock->getSize(0),
                                                  MyBlockLab::m_cacheBlock->getSize(1),
                                                  MyBlockLab::m_cacheBlock->getSize(2),sizeof(ET)/sizeof(Real));
                    }
                }
              } 
            
            }//icode = 0,...,26    


        MyBlockLab::post_load(info, t, applybc);


	

    }

    void release()
    {
        MyBlockLab::release();
    }

























};

CUBISM_NAMESPACE_END
