#pragma once

#include <map>
#include <vector>
#include <cmath>
#include <algorithm>
#include <mpi.h>
#include <omp.h>

#include "BlockInfo.h"
#include "StencilInfo.h"
#include "PUPkernelsMPI.h"

CUBISM_NAMESPACE_BEGIN



struct Interface
{
    BlockInfo * infos [2];
    int icode [2];
    bool CoarseStencil;
    Interface (BlockInfo & i0,BlockInfo & i1, int a_icode0, int a_icode1)
    {
        infos[0] = &i0;
        infos[1] = &i1;
        icode[0] = a_icode0;
        icode[1] = a_icode1;
    }
    bool operator<(const Interface & other) const 
    {
        if (infos[0]->level == other.infos[0]->level && infos[0]->Z == other.infos[0]->Z) //same sender
        {
     
            if (infos[1]->level == other.infos[1]->level && infos[1]->Z == other.infos[1]->Z) //same receiver
            {
                //then the sender will be coarser
                //assert(infos[0]->level != infos[1]->level);
                return (icode[1] < other.icode[1]);
            } 
     
            if (infos[1]->level == other.infos[1]->level) return (infos[1]->Z     < other.infos[1]->Z    );
            else                                          return (infos[1]->level < other.infos[1]->level);
        }  
        if (infos[0]->level == other.infos[0]->level) return (infos[0]->Z     < other.infos[0]->Z    );
        else                                          return (infos[0]->level < other.infos[0]->level);       
    }   
};




template <typename Real>
class SynchronizerMPI_AMR
{
    StencilInfo  stencil; // stencil associated with kernel (advection,diffusion etc.)
    StencilInfo Cstencil; // stencil required to do coarse->fine interpolation
    
    MPI_Comm comm;
    int rank;
    int size;
    bool isroot;
    const bool xperiodic,yperiodic,zperiodic;

    std::vector< std::vector<Real> > send_buffer;
    std::vector< std::vector<Real> > recv_buffer;

    struct PackInfo { Real * block, * pack; int sx, sy, sz, ex, ey, ez; };
    std::vector<std::vector<PackInfo>> send_packinfos;










    //communication & computation overlap
    std::vector<BlockInfo> inner_blocks;
    std::vector<BlockInfo>  halo_blocks;
    std::vector <MPI_Request> send_requests;
    std::vector <MPI_Request> recv_requests;












    //grid parameters
    const int levelMax;
    int blocksPerDim [3];
    int blocksize[3];
    SpaceFillingCurve Zcurve;
    std::vector <BlockInfo> myInfos;
    std::vector <std::vector<BlockInfo>> BlockInfoAll;
    int getZforward(const int level,const int i, const int j, const int k) const 
    {
        int TwoPower = pow(2,level);
        int ix = (i+TwoPower*blocksPerDim[0]) % (blocksPerDim[0]*TwoPower);
        int iy = (j+TwoPower*blocksPerDim[1]) % (blocksPerDim[1]*TwoPower);
        int iz = (k+TwoPower*blocksPerDim[2]) % (blocksPerDim[2]*TwoPower);
        return Zcurve.forward(level,ix,iy,iz);
    }
    inline BlockInfo & getBlockInfoAll(int m, int n) 
    {
        return BlockInfoAll[m][n];
    }



   

    struct MyRange
    {
      int index;
      int sx, sy, sz, ex, ey, ez;                     
      bool needed;
      int removedBecause;
      bool outside(MyRange range) const
      {
        const int x0 = std::max(sx, range.sx);
        const int y0 = std::max(sy, range.sy);
        const int z0 = std::max(sz, range.sz);
        const int x1 = std::min(ex, range.ex);
        const int y1 = std::min(ey, range.ey);
        const int z1 = std::min(ez, range.ez);
    
        return (x0 >= x1) || (y0 >= y1) || (z0 >= z1);
      }


      bool contains(MyRange r) const
      {
        return ( sx <= r.sx && r.ex <= ex ) && ( sy <= r.sy && r.ey <= ey ) && ( sz <= r.sz && r.ez <= ez );
      }

      bool isEqual(MyRange r) const
      {
        return ( sx == r.sx && r.ex == ex ) && ( sy == r.sy && r.ey == ey ) && ( sz == r.sz && r.ez == ez );
      }

      bool operator<(const MyRange & other) const 
      {
          return index < other.index;
      }   


    };


    struct cube //could be more efficient, fix later
    {
        std::vector< std::vector<MyRange> > compass;
      
        cube()
        {
            compass.resize(27);
        }
    
        void remEl(std::vector<int> & v)
        {   
            for (int i=0; i<27; i++)
            {
                std::vector<MyRange> & ranges = compass[i];
                for (auto & r:ranges) if (!r.needed)
                    v.push_back(r.index);
            }    
        }

        void remEl_index(std::vector<int> & v)
        {   
            for (int i=0; i<27; i++)
            {
                std::vector<MyRange> & ranges = compass[i];
                for (auto & r:ranges) if (!r.needed)
                    v.push_back(r.removedBecause);
            }    
        }



        std::vector<MyRange> keepEl()
        {
            std::vector<MyRange> retval;
        
            for (int i=0; i<27; i++)
            {
                std::vector<MyRange> & ranges = compass[i];
                for (auto & r:ranges) if (r.needed)
                    retval.push_back(r);
            }    

            std::sort(retval.begin(),retval.end());
            
            return retval;
        }

    
        void __needed()
        {
            std::vector< std::array<int,3> > faces_and_edges(6 + 12);

            faces_and_edges[0 ] = {0,1,1};
            faces_and_edges[1 ] = {2,1,1};
            faces_and_edges[2 ] = {1,0,1};
            faces_and_edges[3 ] = {1,2,1};
            faces_and_edges[4 ] = {1,1,0};
            faces_and_edges[5 ] = {1,1,2};
           
            faces_and_edges[6 ] = {0,0,1};
            faces_and_edges[7 ] = {0,2,1};
            faces_and_edges[8 ] = {2,0,1};
            faces_and_edges[9 ] = {2,2,1};           
            faces_and_edges[10] = {1,0,0};
            faces_and_edges[11] = {1,0,2};
            faces_and_edges[12] = {1,2,0};
            faces_and_edges[13] = {1,2,2};
            faces_and_edges[14] = {0,1,0};
            faces_and_edges[15] = {0,1,2};
            faces_and_edges[16] = {2,1,0};
            faces_and_edges[17] = {2,1,2};

            for (auto & f:faces_and_edges)
            if ( compass[f[0] + f[1]*3 + f[2]*9].size() != 0 )
            {
                std::vector<MyRange> & me = compass[f[0] + f[1]*3 + f[2]*9];
             
                int imax = (f[0] == 1) ? 2:f[0];
                int imin = (f[0] == 1) ? 0:f[0]; 
                
                int jmax = (f[1] == 1) ? 2:f[1];
                int jmin = (f[1] == 1) ? 0:f[1]; 
                
                int kmax = (f[2] == 1) ? 2:f[2];
                int kmin = (f[2] == 1) ? 0:f[2]; 
#if 1
                for (int k=kmin;k<=kmax;k++)
                for (int j=jmin;j<=jmax;j++)
                for (int i=imin;i<=imax;i++)
                {
                    //if ( (i==1 && j==1) || (i==1 && k==1) || (j==1 && k==1)) continue;

                    std::vector<MyRange> & other = compass[i + j*3 + k*9];
             
                    for (auto & o:other) if (o.needed)
                    {
                       for (auto & m:me   )
                       {  
                        if (m.needed && m.contains(o) && (!m.isEqual(o)))
                        {
                           o.needed = false;
                           o.removedBecause = m.index;
                        }
                      }
                    }
                }
#endif
            }       
        }
    };




    int CoarseStencilVolume(const int * code)
    {
        int eC[3] = { stencil.ex/2+ Cstencil.ex,
                      stencil.ey/2+ Cstencil.ey,
                      stencil.ez/2+ Cstencil.ez};
        int sC[3] = {(stencil.sx-1)/2+ Cstencil.sx,
                     (stencil.sy-1)/2+ Cstencil.sy,
                     (stencil.sz-1)/2+ Cstencil.sz};
        sC[0] = code[0]<1? (code[0]<0 ? sC[0]:0  ) : blocksize[0]/2;
        sC[1] = code[1]<1? (code[1]<0 ? sC[1]:0  ) : blocksize[1]/2;
        sC[2] = code[2]<1? (code[2]<0 ? sC[2]:0  ) : blocksize[2]/2;
        eC[0] = code[0]<1? (code[0]<0 ? 0    :blocksize[0]/2 ) : blocksize[0]/2+eC[0]-1 ; 
        eC[1] = code[1]<1? (code[1]<0 ? 0    :blocksize[1]/2 ) : blocksize[1]/2+eC[1]-1 ; 
        eC[2] = code[2]<1? (code[2]<0 ? 0    :blocksize[2]/2 ) : blocksize[2]/2+eC[2]-1 ;   

   
        int V = (eC[0]-sC[0])*(eC[1]-sC[1])*(eC[2]-sC[2]);
        return V;
    }                                  
        

    void CoarseStencilLength(const int * code, int * L) const 
    {
        int eC[3] = { stencil.ex/2+ Cstencil.ex,
                      stencil.ey/2+ Cstencil.ey,
                      stencil.ez/2+ Cstencil.ez};
        int sC[3] = {(stencil.sx-1)/2+ Cstencil.sx,
                     (stencil.sy-1)/2+ Cstencil.sy,
                     (stencil.sz-1)/2+ Cstencil.sz};
        sC[0] = code[0]<1? (code[0]<0 ? sC[0]:0  ) : blocksize[0]/2;
        sC[1] = code[1]<1? (code[1]<0 ? sC[1]:0  ) : blocksize[1]/2;
        sC[2] = code[2]<1? (code[2]<0 ? sC[2]:0  ) : blocksize[2]/2;
        eC[0] = code[0]<1? (code[0]<0 ? 0    :blocksize[0]/2 ) : blocksize[0]/2+eC[0]-1 ; 
        eC[1] = code[1]<1? (code[1]<0 ? 0    :blocksize[1]/2 ) : blocksize[1]/2+eC[1]-1 ; 
        eC[2] = code[2]<1? (code[2]<0 ? 0    :blocksize[2]/2 ) : blocksize[2]/2+eC[2]-1 ;   

   
        L[0] = eC[0]-sC[0];
        L[1] = eC[1]-sC[1];
        L[2] = eC[2]-sC[2];
        
    }      



    struct UnPackInfo
    {
        int offset;
        int lx,ly,lz;
        int srcxstart, srcystart, srczstart;
        int LX,LY;
        int CoarseVersionOffset;
       

        int CoarseVersionLX,CoarseVersionLY;
        int CoarseVersionsrcxstart,CoarseVersionsrcystart,CoarseVersionsrczstart;
    };




    std::map < Interface, UnPackInfo > MapOfPacks;




    void __FixDuplicates( Interface & f, Interface & f_dup, int lx, int ly, int lz, int lx_dup, int ly_dup, int lz_dup, int & sx,int & sy, int & sz)
    {
        BlockInfo & receiver     = * f.    infos[1];
        BlockInfo & receiver_dup = * f_dup.infos[1];
 
     
        //BlockInfo & sender   = * f.infos[0];
        //assert (sender.level == f_dup.infos[0]->level && sender.Z == f_dup.infos[0]->Z);

        if (receiver.level >= receiver_dup.level )
        {
            int icode_dup = f_dup.icode[1];
            const int code_dup[3] = { icode_dup%3-1, (icode_dup/3)%3-1, (icode_dup/9)%3-1};
            //sx = (lx == lx_dup || code_dup[0] != -1) ? 0 :lx - lx_dup;
            //sy = (ly == ly_dup || code_dup[1] != -1) ? 0 :ly - ly_dup;
            //sz = (lz == lz_dup || code_dup[2] != -1) ? 0 :lz - lz_dup; 
            


            if (lx == lx_dup)
                sx = 0;
            else 
            {
                if (code_dup[0] != -1)
                    sx = 0;
                else
                    sx = lx - lx_dup;
            }


            if (ly == ly_dup)
                sy = 0;
            else 
            {
                if (code_dup[1] != -1)
                    sy = 0;
                else
                    sy = ly - ly_dup;
            }   

            if (lz == lz_dup)
                sz = 0;
            else 
            {
                if (code_dup[2] != -1)
                    sz = 0;
                else
                    sz = lz - lz_dup;
            }



        }
        else
        {
            MyRange range,range_dup;
            DetermineStencil(f,range);
            DetermineStencil(f_dup,range_dup);

            sx =  range_dup.sx-range.sx;
            sy =  range_dup.sy-range.sy;
            sz =  range_dup.sz-range.sz;
        }
    }



    void __FixDuplicates2( Interface & f, Interface & f_dup, int & sx,int & sy, int & sz)
    {
        if (f.infos[0]->level != f.infos[1]->level || f_dup.infos[0]->level != f_dup.infos[1]->level) return;
        
        MyRange range,range_dup;
        DetermineStencil(f,range,true);
        DetermineStencil(f_dup,range_dup,true);

        sx =  range_dup.sx-range.sx;
        sy =  range_dup.sy-range.sy;
        sz =  range_dup.sz-range.sz;
    }




public:
    SynchronizerMPI_AMR(StencilInfo a_stencil, 
                        StencilInfo a_Cstencil, 
                        MPI_Comm a_comm, 
                        const bool a_periodic[3],
                        const int a_levelMax,
                        const int a_nx,
                        const int a_ny,
                        const int a_nz,
                        const int a_bx,
                        const int a_by,
                        const int a_bz,
                        std::vector<BlockInfo> & a_myInfos,
                        std::vector<std::vector<BlockInfo>> & a_BlockInfoAll):
    
    stencil(a_stencil), 
    Cstencil(a_Cstencil), 
    comm(a_comm), 
    xperiodic(a_periodic[0]),
    yperiodic(a_periodic[1]),
    zperiodic(a_periodic[2]),
    levelMax(a_levelMax),Zcurve(a_bx,a_by,a_bz),
    myInfos(a_myInfos),
    BlockInfoAll(a_BlockInfoAll)

    {
        MPI_Comm_rank(comm,&rank);
        MPI_Comm_size(comm,&size);
        isroot = (rank == 0);
                   
        blocksize[0] = a_nx;
        blocksize[1] = a_ny;
        blocksize[2] = a_nz;
        blocksPerDim[0] = a_bx;
        blocksPerDim[1] = a_by;
        blocksPerDim[2] = a_bz;
    }

    bool UseCoarseStencil (Interface & f)
    {
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
            int n = getZforward(a.level,i0,i1,i2);
            if ( (getBlockInfoAll(a.level,n)).TreePos == CheckCoarser )
            {
                retval = true;
                break;
            }
        }

        return retval;
    }




    void DefineInterfaces(std::vector<std::vector<Interface>> &send_interfaces, std::vector<std::vector<Interface>> &recv_interfaces)
    {        
        inner_blocks.clear();
        halo_blocks.clear();        
        send_interfaces.resize(size);
        recv_interfaces.resize(size);   

        for (int i=0; i<(int)myInfos.size(); i++)
        {
            BlockInfo & info = myInfos[i];

            int aux = pow(2,info.level);
            const bool xskin = info.index[0]==0 || info.index[0]==blocksPerDim[0]*aux-1;
            const bool yskin = info.index[1]==0 || info.index[1]==blocksPerDim[1]*aux-1;
            const bool zskin = info.index[2]==0 || info.index[2]==blocksPerDim[2]*aux-1;
            const int xskip  = info.index[0]==0 ? -1 : 1;
            const int yskip  = info.index[1]==0 ? -1 : 1;
            const int zskip  = info.index[2]==0 ? -1 : 1;


            bool isInner = true;

            for(int icode=0; icode<27; icode++)
            {
                if (icode == 1*1 + 3*1 + 9*1) continue;
                const int code[3] = { icode%3-1, (icode/3)%3-1, (icode/9)%3-1};
                if (!xperiodic && code[0] == xskip && xskin) continue;
                if (!yperiodic && code[1] == yskip && yskin) continue;
                if (!zperiodic && code[2] == zskip && zskin) continue; 
                
                //if (!stencil.tensorial && !Cstencil.tensorial && abs(code[0])+abs(code[1])+abs(code[2])>1) continue;
         
                BlockInfo & infoNei = getBlockInfoAll(info.level,info.Znei_(code[0],code[1],code[2]));   
      


                if (infoNei.TreePos == Exists && infoNei.myrank != rank)
                {
                  isInner = false;
                  int icode2 = (-code[0]+1) + (-code[1]+1)*3 + (-code[2]+1)*9;
                  send_interfaces[infoNei.myrank].push_back( Interface(info,infoNei,icode,icode2) );
                  recv_interfaces[infoNei.myrank].push_back( Interface(infoNei,info,icode2,icode) );
                }

                else if (infoNei.TreePos == CheckCoarser)
                {
                    int nCoarse = getZforward(infoNei.level-1,infoNei.index[0]/2,infoNei.index[1]/2,infoNei.index[2]/2);
                    BlockInfo & infoNeiCoarser = getBlockInfoAll(infoNei.level-1,nCoarse);
                    if (infoNeiCoarser.myrank != rank)
                    {
                        isInner = false;
                  
                        int code2[3] = {-code[0],-code[1],-code[2]};
                        int icode2 = (code2[0]+1) + (code2[1]+1)*3 + (code2[2]+1)*9;
                        recv_interfaces[infoNeiCoarser.myrank].push_back( Interface(infoNeiCoarser,info,icode2,icode) );   
                 
                        BlockInfo & test = getBlockInfoAll(infoNeiCoarser.level,infoNeiCoarser.Znei_(code2[0],code2[1],code2[2]));

                        if (info.index[0]/2 == test.index[0] && info.index[1]/2 == test.index[1] && info.index[2]/2 == test.index[2])
                            send_interfaces[infoNeiCoarser.myrank].push_back( Interface(info,infoNeiCoarser,icode,icode2) );
                    }
                }
                else if (infoNei.TreePos == CheckFiner)
                {
                    int Bstep = 1; //face
                    if      ((abs(code[0])+abs(code[1])+abs(code[2])==2 )) Bstep = 3; //edge
                    else if ((abs(code[0])+abs(code[1])+abs(code[2])==3 )) Bstep = 4; //corner
                   
                    for (int B = 0 ; B <= 3 ; B += Bstep) //loop over blocks that make up face/edge/corner (respectively 4,2 or 1 blocks)
                    {
                        const int temp = (abs(code[0])==1) ? (B%2) : (B/2) ;
                        int nFine = getZforward(infoNei.level+1,2*info.index[0] + max(code[0],0) +code[0]  + (B%2)*max(0, 1 - abs(code[0])),
                                                                2*info.index[1] + max(code[1],0) +code[1]  + temp *max(0, 1 - abs(code[1])),
                                                                2*info.index[2] + max(code[2],0) +code[2]  + (B/2)*max(0, 1 - abs(code[2])));
                        BlockInfo & infoNeiFiner = getBlockInfoAll(infoNei.level+1,nFine);
                        if (infoNeiFiner.myrank != rank)
                        {
                            isInner = false;
                  
                            int icode2 = (-code[0]+1) + (-code[1]+1)*3 + (-code[2]+1)*9;
                            send_interfaces[infoNeiFiner.myrank].push_back( Interface(info,infoNeiFiner,icode,icode2) );
                            recv_interfaces[infoNeiFiner.myrank].push_back( Interface(infoNeiFiner,info,icode2,icode) );

                           
                            if (Bstep == 3) //if I'm filling an edge then I'm also filling a corner
                            {
                                int code3[3];

                                code3[0] = (code[0]==0) ? ( B==0 ? 1 : -1): -code[0];
                                code3[1] = (code[1]==0) ? ( B==0 ? 1 : -1): -code[1];
                                code3[2] = (code[2]==0) ? ( B==0 ? 1 : -1): -code[2];

                                int icode3 = (code3[0]+1) + (code3[1]+1)*3 + (code3[2]+1)*9;
                                send_interfaces[infoNeiFiner.myrank].push_back( Interface(info,infoNeiFiner,icode,icode3) );
                            }
                            else if (Bstep == 1) //if I'm filling a face then I'm also filling two edges and a corner
                            {
                                int code3[3];
                                int code4[3];
                                int code5[3];

                                int d0,d1,d2;

                                assert (code[0] !=0 || code[1] !=0 || code[2] !=0);
                                assert (abs(code[0])+abs(code[1])+abs(code[2]) == 1);


                                if      (code[0]!=0) {d0 = 0; d1=1; d2=2;}
                                else if (code[1]!=0) {d0 = 1; d1=0; d2=2;}
                                else /*if (code[2]!=0)*/ {d0 = 2; d1=0; d2=1;}

                                code3[d0] = -code[d0];
                                code4[d0] = -code[d0];
                                code5[d0] = -code[d0];
                             
                                if (B==0)
                                {
                                    code3[d1] =  1;
                                    code3[d2] =  1;

                                    code4[d1] =  1;
                                    code4[d2] =  0;

                                    code5[d1] =  0;
                                    code5[d2] =  1;
                                }
                                else if (B==1)
                                {
                                    code3[d1] = -1;
                                    code3[d2] =  1;

                                    code4[d1] = -1;
                                    code4[d2] =  0;

                                    code5[d1] =  0;
                                    code5[d2] =  1;
                                }
                                else if (B==2)
                                {
                                    code3[d1] =  1;
                                    code3[d2] = -1;

                                    code4[d1] =  1;
                                    code4[d2] =  0;

                                    code5[d1] =  0;
                                    code5[d2] = -1;
                                }
                                else if (B==3)
                                {
                                    code3[d1] = -1;
                                    code3[d2] = -1;

                                    code4[d1] = -1;
                                    code4[d2] =  0;

                                    code5[d1] =  0;
                                    code5[d2] = -1;
                                }
                                int icode3 = (code3[0]+1) + (code3[1]+1)*3 + (code3[2]+1)*9;
                                int icode4 = (code4[0]+1) + (code4[1]+1)*3 + (code4[2]+1)*9;
                                int icode5 = (code5[0]+1) + (code5[1]+1)*3 + (code5[2]+1)*9;
                         
                                send_interfaces[infoNeiFiner.myrank].push_back( Interface(info,infoNeiFiner,icode,icode3) );
                                send_interfaces[infoNeiFiner.myrank].push_back( Interface(info,infoNeiFiner,icode,icode4) );
                                send_interfaces[infoNeiFiner.myrank].push_back( Interface(info,infoNeiFiner,icode,icode5) );

                            }
         
                        }
                    }
                } 
            }//icode = 0,...,26  


            if (isInner)
                inner_blocks.push_back(info);
            else
                halo_blocks.push_back(info);



        }//i-loop
    }





    std::vector<BlockInfo> avail_inner()
    {
          //MPI_Waitall(size, &recv_requests[0], MPI_STATUSES_IGNORE);
          //MPI_Waitall(size, &send_requests[0], MPI_STATUSES_IGNORE);    
        return inner_blocks;
    }

    std::vector<BlockInfo> avail_halo()
    {
        MPI_Waitall(size, &recv_requests[0], MPI_STATUSES_IGNORE);
        MPI_Waitall(size, &send_requests[0], MPI_STATUSES_IGNORE);
        return halo_blocks;
    }









 
    void DetermineStencil(Interface f, MyRange & range, bool CoarseVersion = false)
    {
        const int nX = blocksize[0];
        const int nY = blocksize[1];
        const int nZ = blocksize[2];
        range.needed = true;
       
        const int code[3] = { f.icode[1]%3-1, (f.icode[1]/3)%3-1, (f.icode[1]/9)%3-1};
 

        if (CoarseVersion)
        {
            assert((f.infos[0]->level == f.infos[1]->level));

            const int eC[3] = { stencil.ex/2+ Cstencil.ex +1-1,
                                stencil.ey/2+ Cstencil.ey +1-1,
                                stencil.ez/2+ Cstencil.ez +1-1};

            const int sC[3] = {(stencil.sx-1)/2+ Cstencil.sx,
                               (stencil.sy-1)/2+ Cstencil.sy,
                               (stencil.sz-1)/2+ Cstencil.sz};

            range.sx = code[0]<1? (code[0]<0 ? nX/2 + sC[0]:0    ) : 0;
            range.sy = code[1]<1? (code[1]<0 ? nY/2 + sC[1]:0    ) : 0;
            range.sz = code[2]<1? (code[2]<0 ? nZ/2 + sC[2]:0    ) : 0;
      
            range.ex = code[0]<1? (code[0]<0 ? nX/2        :nX/2 ) : eC[0];
            range.ey = code[1]<1? (code[1]<0 ? nY/2        :nY/2 ) : eC[1];
            range.ez = code[2]<1? (code[2]<0 ? nZ/2        :nZ/2 ) : eC[2];                              
        }
        else
        {
            if (f.infos[0]->level == f.infos[1]->level)
            {
    
                range.sx = code[0]<1? (code[0]<0 ? nX + stencil.sx:0 ):0;
                range.sy = code[1]<1? (code[1]<0 ? nY + stencil.sy:0 ):0;
                range.sz = code[2]<1? (code[2]<0 ? nZ + stencil.sz:0 ):0;
                range.ex = code[0]<1? (code[0]<0 ? nX             :nX):stencil.ex-1; 
                range.ey = code[1]<1? (code[1]<0 ? nY             :nY):stencil.ey-1; 
                range.ez = code[2]<1? (code[2]<0 ? nZ             :nZ):stencil.ez-1;           
            }
            else if (f.infos[0]->level > f.infos[1]->level)
            {
    
                range.sx =  code[0]<1? (code[0]<0 ? nX + 2*stencil.sx:0 ): 0;
                range.sy =  code[1]<1? (code[1]<0 ? nY + 2*stencil.sy:0 ): 0;
                range.sz =  code[2]<1? (code[2]<0 ? nZ + 2*stencil.sz:0 ): 0;          
                range.ex =  code[0]<1? (code[0]<0 ? nX               :nX): 2*stencil.ex-1;
                range.ey =  code[1]<1? (code[1]<0 ? nY               :nY): 2*stencil.ey-1;
                range.ez =  code[2]<1? (code[2]<0 ? nZ               :nZ): 2*stencil.ez-1;
            }
            else
            {
              
                const int s[3] = {code[0]<1? (code[0]<0 ? ((stencil.sx-1)/2+ Cstencil.sx) :0 ) : nX/2,
                                  code[1]<1? (code[1]<0 ? ((stencil.sy-1)/2+ Cstencil.sy) :0 ) : nY/2,
                                  code[2]<1? (code[2]<0 ? ((stencil.sz-1)/2+ Cstencil.sz) :0 ) : nZ/2 };
    
                const int e[3] = {code[0]<1? (code[0]<0 ? 0:nX/2 ) : nX/2+(stencil.ex)/2+ Cstencil.ex -1,
                                  code[1]<1? (code[1]<0 ? 0:nY/2 ) : nY/2+(stencil.ey)/2+ Cstencil.ey -1,
                                  code[2]<1? (code[2]<0 ? 0:nZ/2 ) : nZ/2+(stencil.ez)/2+ Cstencil.ez -1};
           
                const int base[3] = { (f.infos[1]->index[0]+ code[0])%2,
                                      (f.infos[1]->index[1]+ code[1])%2,
                                      (f.infos[1]->index[2]+ code[2])%2};       
    
                BlockInfo  CoarseSender =  getBlockInfoAll(f.infos[1]->level,f.infos[1]->Znei_(code[0],code[1],code[2]));
    
                int CoarseEdge[3];
              
                CoarseEdge[0] = (code[0] == 0) ? 0 :   (   ( (f.infos[1]->index[0]%2 ==0)&&(CoarseSender.index_(0)>f.infos[1]->index_(0)) ) || ( (f.infos[1]->index_(0)%2 ==1)&&(CoarseSender.index_(0)<f.infos[1]->index_(0)) )  )? 1:0  ;
                CoarseEdge[1] = (code[1] == 0) ? 0 :   (   ( (f.infos[1]->index[1]%2 ==0)&&(CoarseSender.index_(1)>f.infos[1]->index_(1)) ) || ( (f.infos[1]->index_(1)%2 ==1)&&(CoarseSender.index_(1)<f.infos[1]->index_(1)) )  )? 1:0  ;
                CoarseEdge[2] = (code[2] == 0) ? 0 :   (   ( (f.infos[1]->index[2]%2 ==0)&&(CoarseSender.index_(2)>f.infos[1]->index_(2)) ) || ( (f.infos[1]->index_(2)%2 ==1)&&(CoarseSender.index_(2)<f.infos[1]->index_(2)) )  )? 1:0  ;
                                   
                range.sx = s[0] + max(code[0],0)*nX/2 + (1-abs(code[0]))*base[0]*nX/2 - code[0]*nX  + CoarseEdge[0] *code[0]*nX/2;     
                range.sy = s[1] + max(code[1],0)*nY/2 + (1-abs(code[1]))*base[1]*nY/2 - code[1]*nY  + CoarseEdge[1] *code[1]*nY/2;     
                range.sz = s[2] + max(code[2],0)*nZ/2 + (1-abs(code[2]))*base[2]*nZ/2 - code[2]*nZ  + CoarseEdge[2] *code[2]*nZ/2;    
               
                range.ex = e[0] + max(code[0],0)*nX/2 + (1-abs(code[0]))*base[0]*nX/2 - code[0]*nX  + CoarseEdge[0] *code[0]*nX/2;
                range.ey = e[1] + max(code[1],0)*nY/2 + (1-abs(code[1]))*base[1]*nY/2 - code[1]*nY  + CoarseEdge[1] *code[1]*nY/2;
                range.ez = e[2] + max(code[2],0)*nZ/2 + (1-abs(code[2]))*base[2]*nZ/2 - code[2]*nZ  + CoarseEdge[2] *code[2]*nZ/2;      

            }
        }
    }




    void DetermineStencilLength(const int level_sender, const int level_receiver, const int * code, int * L)
    {
        if (level_sender == level_receiver)
        {
            L[0] =  code[0]<1? (code[0]<0 ? -stencil.sx:blocksize[0]):stencil.ex-1;
            L[1] =  code[1]<1? (code[1]<0 ? -stencil.sy:blocksize[1]):stencil.ey-1;
            L[2] =  code[2]<1? (code[2]<0 ? -stencil.sz:blocksize[2]):stencil.ez-1;
        }
        else if (level_sender > level_receiver)
        {
            L[0] = code[0]<1? (code[0]<0 ? -stencil.sx:blocksize[0]/2):stencil.ex-1;
            L[1] = code[1]<1? (code[1]<0 ? -stencil.sy:blocksize[1]/2):stencil.ey-1;
            L[2] = code[2]<1? (code[2]<0 ? -stencil.sz:blocksize[2]/2):stencil.ez-1;
        }
        else
        {
            L[0] = code[0]<1? (code[0]<0 ? -((stencil.sx-1)/2+ Cstencil.sx):blocksize[0]/2):(stencil.ex)/2+ Cstencil.ex -1;
            L[1] = code[1]<1? (code[1]<0 ? -((stencil.sy-1)/2+ Cstencil.sy):blocksize[1]/2):(stencil.ey)/2+ Cstencil.ey -1;
            L[2] = code[2]<1? (code[2]<0 ? -((stencil.sz-1)/2+ Cstencil.sz):blocksize[2]/2):(stencil.ez)/2+ Cstencil.ez -1;
        }
    }





    void DiscardDuplicates(std::vector<Interface> & f, int & total_size, bool updateMap, int NC, int r, int  & count )
    {    
        total_size = 0;
        int index = 0;

        std::vector<int> remEl;

        std::vector<int> remEl_index;


        int offset = 0;


        if (f.size()>0) do
        {
            int m = f[index].infos[0]->level;
            int n = f[index].infos[0]->Z;
               
            int index_end=f.size();
                
            for (int i=index; i<(int)f.size();i++)
            {
                int m_ = f[i].infos[0]->level;
                int n_ = f[i].infos[0]->Z;
                if (m_ != m || n_ != n)
                {
                    index_end = i;
                    break;
                }
            }
    
            cube C;
            bool skip_needed = false;
            for (int i=index; i<index_end; i++)
            {
                MyRange range;
                range.index = i;
                DetermineStencil(f[i],range);
                C.compass[f[i].icode[0]].push_back(range);


                skip_needed =  UseCoarseStencil(f[i]);
            }
            if (!skip_needed)
            C.__needed();





            C.remEl(remEl); 
            C.remEl_index(remEl_index);   

           


            for (auto & i:C.keepEl())
            {
                int L[3];
                int k = i.index;
               

                int code[3] = { f[k].icode[1]%3-1, (f[k].icode[1]/3)%3-1, (f[k].icode[1]/9)%3-1};

                DetermineStencilLength(f[k].infos[0]->level,f[k].infos[1]->level,&code[0],&L[0]);
                int V = L[0]*L[1]*L[2];
                total_size+= V;


                if (UseCoarseStencil(f[k]))
                {
                    total_size += CoarseStencilVolume(&code[0]);
                }







                
                if (updateMap)
                {
                    if (rank==0)
                    {
                        //std::cout << "rank 0 receives from rank " << r << "  k=" << k << " offset=" <<offset << "\n"; 
                    }

     
                    UnPackInfo info = {offset,L[0],L[1],L[2],0,0,0,L[0],L[1],0,  0,0,0,0,0};
                    offset += V*NC;
                
                    if (UseCoarseStencil(f[k]))
                    {
                        offset += CoarseStencilVolume(&code[0])*NC;
                        info.CoarseVersionOffset = V*NC;                       
                
                        CoarseStencilLength(&code[0],&L[0]);
                
                        info.CoarseVersionLX = L[0];
                        info.CoarseVersionLY = L[1];
                    }

                    MapOfPacks.insert(std::pair<Interface,UnPackInfo>(f[k],info));
                    count ++;
                }
       
            }
  
            index = index_end;
        }
        while (index < (int)f.size());
 
       

        
        if (updateMap)
        for (int k=0; k<(int)remEl.size();k++)
        {  


            int L[3];
            int code[3] = { f[remEl[k]].icode[1]%3-1, (f[remEl[k]].icode[1]/3)%3-1, (f[remEl[k]].icode[1]/9)%3-1};
         
            DetermineStencilLength(f[remEl[k]].infos[0]->level,f[remEl[k]].infos[1]->level,&code[0],&L[0]);
          
            int srcxstart, srcystart, srczstart;



            auto search = MapOfPacks.find( f[remEl_index[k]] );
            __FixDuplicates(f[remEl_index[k]],f[remEl[k]], 
                            search->second.lx,search->second.ly,search->second.lz,
                            L[0],L[1],L[2],
                            srcxstart, srcystart, srczstart);

            int CoarseVersionsrcxstart=0;
            int CoarseVersionsrcystart=0;
            int CoarseVersionsrczstart=0;


            __FixDuplicates2(f[remEl_index[k]],f[remEl[k]],CoarseVersionsrcxstart,CoarseVersionsrcystart,CoarseVersionsrczstart);
            UnPackInfo info = {search->second.offset,L[0],L[1],L[2],srcxstart, srcystart, srczstart,search->second.LX,search->second.LY,search->second.CoarseVersionOffset,search->second.CoarseVersionLX,search->second.CoarseVersionLY,CoarseVersionsrcxstart, CoarseVersionsrcystart, CoarseVersionsrczstart};
            MapOfPacks.insert(std::pair<Interface,UnPackInfo>(f[remEl[k]],info));
            count ++; 
            if (rank==0)
            {
               // std::cout << "rank 0 receives from rank " << r << "  k=" << remEl[k] << " offset=" <<search->second.offset << "\n"; 
            }

        }



        std::sort(remEl.begin(), remEl.end());
        for (int k=0; k<(int)remEl.size();k++)
        {  
            f.erase(f.begin()+remEl[k]-k);
        }


    }







    void sync(unsigned int gptfloats, MPI_Datatype MPIREAL, const int timestamp)
    {     
        const int nX = blocksize[0];
        const int nY = blocksize[1];
        const int nZ = blocksize[2];
        

        std::vector<int> selcomponents = stencil.selcomponents;
        std::sort(selcomponents.begin(), selcomponents.end());
        

        const int NC = selcomponents.size();
        





        MapOfPacks.clear();

        //1.Find all interfaces with neighboring ranks
        std::vector<std::vector<Interface>> send_interfaces;
        std::vector<std::vector<Interface>> recv_interfaces;
        DefineInterfaces(send_interfaces,recv_interfaces); 

        //2.Sort interfaces 
        for (int r=0; r<size; r++)
        {
            std::sort (send_interfaces[r].begin(), send_interfaces[r].end());
            std::sort (recv_interfaces[r].begin(), recv_interfaces[r].end());
        }
    
        //3.Determine buffer sizes
        int count = 0;
        std::vector<int> send_buffer_size(size,0);
        std::vector<int> recv_buffer_size(size,0);
        for (int r=0; r<size; r++)
        {
            DiscardDuplicates(send_interfaces[r],send_buffer_size[r],false,NC,r,count);
            DiscardDuplicates(recv_interfaces[r],recv_buffer_size[r],true ,NC,r,count);
        }
        assert (MapOfPacks.size() == count );

        //4.Allocate buffer memory 
        send_buffer.resize(size);
        recv_buffer.resize(size);
        for (int r=0; r<size; r++)
        {
            send_buffer[r].resize(send_buffer_size[r]*NC, 666.0);
            recv_buffer[r].resize(recv_buffer_size[r]*NC, 777.0);
        

            
        }
        
        #if 1
            for (int r=0;r<size;r++)
            {
                assert( r!=rank || send_interfaces[r].size() == 0);
                assert( r!=rank || recv_interfaces[r].size() == 0);
                


                //if (r==rank) continue; 
                //std::cout << " Rank " << rank << " will send/receive " << send_interfaces[r].size() << "/" << recv_interfaces[r].size() << " from/to rank " << r << "\n";
                //std::cout << " Rank " << rank << " will send/receive " << send_buffer_size[r] << "/" << recv_buffer_size[r] << " from/to rank " << r << "\n";
            }
        #endif
  

        //5.Pack data        
        send_packinfos.resize(size);
        std::vector<int> displacement(size,0);
        for (int r=0; r<size; r++)
        {
            for (int i=0; i<(int) send_interfaces[r].size(); i++)
            {
                Interface & f = send_interfaces[r][i];

                const int code[3] = { f.icode[0]%3-1, (f.icode[0]/3)%3-1, (f.icode[0]/9)%3-1};              
                const int code0[3] = { -code[0], -code[1] , -code[2]}; 

                
                //if (rank !=0 && r==0)
                //    std::cout << "rank=" <<rank << " k=" << i << " displacement="<<displacement[r]<<"\n";
                

                if (f.infos[0]->level <= f.infos[1]->level)
                {
                    MyRange range;
                    DetermineStencil(f,range);
                    int V = (range.ex-range.sx)* (range.ey-range.sy)* (range.ez-range.sz);     
                    PackInfo info_tmp = {(Real *)f.infos[0]->ptrBlock, &send_buffer[r][ displacement[r] ], range.sx,range.sy,range.sz,range.ex,range.ey,range.ez};      
                    send_packinfos[r].push_back(info_tmp);              
                    displacement[r]+= V*NC;
                    
                    if (UseCoarseStencil(f))
                    {
                        std::vector< Real > avgDown = AverageDownAndFill2(f.infos[0],code0,&selcomponents[0],NC,gptfloats);          
                        
                        for (int k = 0; k <CoarseStencilVolume(&code0[0])*NC; k++)
                        {
                            send_buffer[r][ displacement[r] + k] = avgDown[k];
                        }
                        displacement[r] += CoarseStencilVolume(&code0[0])*NC;
                    }
                }
                else //receiver is coarser, so sender averages down data first 
                {
                    const int s [3] = { code0[0]<1? (code0[0]<0 ? stencil.sx:0 ):nX,
                                        code0[1]<1? (code0[1]<0 ? stencil.sy:0 ):nY,
                                        code0[2]<1? (code0[2]<0 ? stencil.sz:0 ):nZ};
                    const int e [3] = { code0[0]<1? (code0[0]<0 ? 0         :nX):nX+stencil.ex-1,
                                        code0[1]<1? (code0[1]<0 ? 0         :nY):nY+stencil.ey-1,
                                        code0[2]<1? (code0[2]<0 ? 0         :nZ):nZ+stencil.ez-1};
                    int V  = ( abs(code0[0])*(e[0]-s[0]) + (1-abs(code0[0]))*((e[0]-s[0])/2) ) * 
                             ( abs(code0[1])*(e[1]-s[1]) + (1-abs(code0[1]))*((e[1]-s[1])/2) ) * 
                             ( abs(code0[2])*(e[2]-s[2]) + (1-abs(code0[2]))*((e[2]-s[2])/2) ) ;
                       

                    std::vector< Real > avgDown = AverageDownAndFill(f.infos[0],s,e,code0,&selcomponents[0],NC, gptfloats);          
                    for (int k = 0; k <V*NC; k++)
                    {
                        send_buffer[r][ displacement[r] + k] = avgDown[k];
                    }
         
                    displacement[r]+= V*NC;             
                }
            }


            if (send_buffer_size[r] == 0) continue;
            const int N = send_packinfos[r].size();
            
            for (int i = 0; i < N ; i++)
            {
                PackInfo info = send_packinfos[r][i];               
                pack(info.block, info.pack, gptfloats, &selcomponents.front(),NC,info.sx,info.sy,info.sz,info.ex,info.ey,info.ez,blocksize[0],blocksize[1]);
                assert(info.block != nullptr && info.pack  != nullptr);
            }
        }
       
        
        MPI_Barrier(MPI_COMM_WORLD);
        //std::vector <MPI_Request> send_requests(size);
        //std::vector <MPI_Request> recv_requests(size);
        
        send_requests.resize(size);
        recv_requests.resize(size);

        for (int r = 0 ; r < size; r ++ )
        {
            MPI_Irecv(&recv_buffer[r][0], recv_buffer_size[r]*NC, MPIREAL, r, timestamp , comm, &recv_requests[r]);
            MPI_Isend(&send_buffer[r][0], send_buffer_size[r]*NC, MPIREAL, r, timestamp , comm, &send_requests[r]);
        }
        //MPI_Waitall(size, &recv_requests[0], MPI_STATUSES_IGNORE);
        //MPI_Waitall(size, &send_requests[0], MPI_STATUSES_IGNORE);
    }









    StencilInfo getstencil() const
    {
        return stencil;
    }


    StencilInfo getCstencil() const
    {
        return Cstencil;
    }


//    std::vector<BlockInfo> avail_halo()
//    {
//        std::vector<BlockInfo> retval;
//        return retval;
//    }
//
//    std::vector<BlockInfo> avail_inner()
//    {
//        std::vector<BlockInfo> retval;
//        return retval;
//    }


 

    void fetch(BlockInfo sender, BlockInfo receiver, int icode_sender, int icode_receiver, Real * dst, int dst_sizeX,int dst_sizeY,int dst_sizeZ, int gptfloats, bool fillCoarse=false) const
    {
        Interface f(sender,receiver,icode_sender,icode_receiver);
        auto search = MapOfPacks.find(f);

        
        if (search == MapOfPacks.end())
        {
            std::cout << " Searching for interface with:\n";
            std::cout << " Sender  : " << sender.level << " " << sender.Z << " " << sender.index[0] << " " << sender.index[1] << " " << sender.index[2]  <<   "\n";
            std::cout << " Receiver: " << receiver.level << " " << receiver.Z << " " << receiver.index[0] << " " << receiver.index[1] << " " << receiver.index[2] <<"\n";     

            assert (search != MapOfPacks.end());    
        }


        UnPackInfo temp = search->second;        

        if (fillCoarse)
        {       
            int L[3];
            const int code[3] = { f.icode[0]%3-1, (f.icode[0]/3)%3-1, (f.icode[0]/9)%3-1};
 
            CoarseStencilLength(&code[0],&L[0]);

            unpack_subregion<Real>(&recv_buffer[sender.myrank][temp.offset+temp.CoarseVersionOffset],
                          &dst[0],
                          gptfloats, 
                          &stencil.selcomponents[0],
                          stencil.selcomponents.size(),                    
                          
                          
                          temp.CoarseVersionsrcxstart,temp.CoarseVersionsrcystart,temp.CoarseVersionsrczstart,


                          temp.CoarseVersionLX,temp.CoarseVersionLY,
                          0,0,0,L[0],L[1],L[2],dst_sizeX,dst_sizeY,dst_sizeZ);
   
        }
        else
        {
            unpack_subregion<Real>(&recv_buffer[sender.myrank][temp.offset],
                          &dst[0],
                          gptfloats, 
                          &stencil.selcomponents[0],
                          stencil.selcomponents.size(),                    
                          temp.srcxstart,temp.srcystart,temp.srczstart,
                          temp.LX,temp.LY,
                          0,0,0,temp.lx,temp.ly,temp.lz,dst_sizeX,dst_sizeY,dst_sizeZ);
    
        }

    }







    std::vector< Real > AverageDownAndFill(BlockInfo * info, const int s[3], const int e[3], 
        const int code[3], int * selcomponents, int NC, int gptfloats)
    {
        const int nX = blocksize[0];
        const int nY = blocksize[1];
        const int nZ = blocksize[2];

        Real * src = (Real *)(*info).ptrBlock;


        int V  = ( abs(code[0])*(e[0]-s[0]) + (1-abs(code[0]))*((e[0]-s[0])/2) ) * 
                 ( abs(code[1])*(e[1]-s[1]) + (1-abs(code[1]))*((e[1]-s[1])/2) ) * 
                 ( abs(code[2])*(e[2]-s[2]) + (1-abs(code[2]))*((e[2]-s[2])/2) ) ;


        std::vector<Real> retval (V*NC,0);

        const int xStep = (code[0] == 0) ? 2:1;
        const int yStep = (code[1] == 0) ? 2:1;
        const int zStep = (code[2] == 0) ? 2:1;

        int pos =0 ;

        for(int iz=s[2]; iz<e[2]; iz+= zStep)
        for(int iy=s[1]; iy<e[1]; iy+= yStep)
        for(int ix=s[0]; ix<e[0]; ix+= xStep)
        {
            const int XX = (abs(code[0]) == 1) ? 2*(ix- code[0]*nX) + min(0,code[0])*nX : ix ;
            const int YY = (abs(code[1]) == 1) ? 2*(iy- code[1]*nY) + min(0,code[1])*nY : iy ;
            const int ZZ = (abs(code[2]) == 1) ? 2*(iz- code[2]*nZ) + min(0,code[2])*nZ : iz ;     
        

            for (int c=0; c<NC; c++)
            {
                int comp = selcomponents[c];

                Real avg = 0;
                for (int k=0; k<2; k++)
                for (int j=0; j<2; j++)
                for (int i=0; i<2; i++)
                {
                    int index = (XX+i) + (YY+j)*nX + (ZZ+k)*nX*nY;
                    avg += *(src + gptfloats*index+comp); 
                }
             
                avg *= 0.125;

  
                retval[pos] = avg;
                pos ++ ;
            }


        }
   
        return retval;


    }






    std::vector< Real > AverageDownAndFill2(BlockInfo * info,  const int code[3], int * selcomponents, int NC, int gptfloats)
    {
        const int nX = blocksize[0];
        const int nY = blocksize[1];
        const int nZ = blocksize[2];

      const int eC[3] = { (stencil.ex)/2+ Cstencil.ex +1-1,
                          (stencil.ey)/2+ Cstencil.ey +1-1,
                          (stencil.ez)/2+ Cstencil.ez +1-1};

      const int sC[3] = {(stencil.sx-1)/2+ Cstencil.sx,
                         (stencil.sy-1)/2+ Cstencil.sy,
                         (stencil.sz-1)/2+ Cstencil.sz};


      const int s[3] = { code[0]<1? (code[0]<0 ? sC[0]:0  ) : nX/2,
                         code[1]<1? (code[1]<0 ? sC[1]:0  ) : nY/2,
                         code[2]<1? (code[2]<0 ? sC[2]:0  ) : nZ/2};

      const int e[3] = { code[0]<1? (code[0]<0 ? 0    :nX/2 ) : nX/2+eC[0]-1,
                         code[1]<1? (code[1]<0 ? 0    :nY/2 ) : nY/2+eC[1]-1,
                         code[2]<1? (code[2]<0 ? 0    :nZ/2 ) : nZ/2+eC[2]-1};



        Real * src = (Real *)(*info).ptrBlock;


        int V  =(e[0]-s[0])*(e[1]-s[1])*(e[2]-s[2]) ;
        std::vector<Real> retval (V*NC,0);

        int pos =0 ;

        for(int iz=s[2]; iz<e[2]; iz++)
        for(int iy=s[1]; iy<e[1]; iy++)
        for(int ix=s[0]; ix<e[0]; ix++)
        {
            const int XX =  2*(ix -s[0]) +s[0]+ max(code[0],0)*nX/2 - code[0]*nX + min(0,code[0])*(e[0]-s[0]);       
            const int YY =  2*(iy -s[1]) +s[1]+ max(code[1],0)*nY/2 - code[1]*nY + min(0,code[1])*(e[1]-s[1]);
            const int ZZ =  2*(iz -s[2]) +s[2]+ max(code[2],0)*nZ/2 - code[2]*nZ + min(0,code[2])*(e[2]-s[2]);

            for (int c=0; c<NC; c++)
            {
                int comp = selcomponents[c];

                Real avg = 0;
                for (int k=0; k<2; k++)
                for (int j=0; j<2; j++)
                for (int i=0; i<2; i++)
                {
                    int index = (XX+i) + (YY+j)*nX + (ZZ+k)*nX*nY;
                    avg += *(src + gptfloats*index+comp); 
                }
             
                avg *= 0.125;

  
                retval[pos] = avg;
                pos ++ ;
            }


        }
   
        return retval;


    }



};


CUBISM_NAMESPACE_END
