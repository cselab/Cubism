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
        CoarseStencil = false;
    }
    bool operator<(const Interface & other) const 
    {
      	#if 0
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
		#else

			#if 1
        	if (infos[0]->blockID == other.infos[0]->blockID) //same sender
            {
                if (infos[1]->blockID == other.infos[1]->blockID) //same receiver
                {
                    return (icode[1] < other.icode[1]);
                } 
                return (infos[1]->blockID < other.infos[1]->blockID);  
            }  
            return (infos[0]->blockID < other.infos[0]->blockID);    
			#else
        	if (infos[1]->blockID == other.infos[1]->blockID) //same receiver
            {
                if (infos[0]->blockID == other.infos[0]->blockID) //same sender
                {
                    return (icode[1] < other.icode[1]);
                } 
                return (infos[0]->blockID < other.infos[0]->blockID);  
            }  
            return (infos[1]->blockID < other.infos[1]->blockID);    
            #endif

		#endif
    }
};


struct MyRange
{
    int index;
    int sx, sy, sz, ex, ey, ez;                     
    bool needed;
    std::vector<int> removedIndices;
    bool contains(MyRange r) const
    {
        int V  = (  ez-  sz)*(  ey-  sy)*(  ex-  sx);
        int Vr = (r.ez-r.sz)*(r.ey-r.sy)*(r.ex-r.sx);
        return ( sx <= r.sx && r.ex <= ex ) && ( sy <= r.sy && r.ey <= ey ) && ( sz <= r.sz && r.ez <= ez ) && (Vr < V);
    }
    bool operator<(const MyRange & other) const 
    {
      return index < other.index;
    }   
    void copy(const MyRange & other) 
    {
      sx = other.sx;
      sy = other.sy;
      sz = other.sz;
      ex = other.ex;
      ey = other.ey;
      ez = other.ez;
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


    struct cube //could be more efficient, fix later
    {
        std::vector< std::vector<MyRange> > compass;

        cube()
        {
            compass.resize(27);
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
    
        void __needed(std::vector<int> & v)
        {
        	static constexpr std::array <int,3> faces_and_edges [18] = {
	            {0,1,1},{2,1,1},{1,0,1},{1,2,1},{1,1,0},{1,1,2},
	            
	            {0,0,1},{0,2,1},{2,0,1},{2,2,1},{1,0,0},{1,0,2},
	            {1,2,0},{1,2,2},{0,1,0},{0,1,2},{2,1,0},{2,1,2}};

            for (auto & f:faces_and_edges)
            if ( compass[f[0] + f[1]*3 + f[2]*9].size() != 0 )
            {
                std::vector<MyRange> & me = compass[f[0] + f[1]*3 + f[2]*9];
             
            
                bool needme = false;
                std::vector<MyRange> & other = compass[f[0] + f[1]*3 + f[2]*9];       
                for (auto & o:other)
                { 
                    if (o.needed)
                    {
                        needme = true;

                        for (auto & m:me   )
                         if (m.needed && m.contains(o) )
                         {
                            o.needed = false;
                            m.removedIndices.push_back(o.index);
                            v.push_back(o.index);
                            break;
                         }
                    }
                }             
                if (!needme) continue;

                int imax = (f[0] == 1) ? 2:f[0];
                int imin = (f[0] == 1) ? 0:f[0]; 
                
                int jmax = (f[1] == 1) ? 2:f[1];
                int jmin = (f[1] == 1) ? 0:f[1]; 
                
                int kmax = (f[2] == 1) ? 2:f[2];
                int kmin = (f[2] == 1) ? 0:f[2];  

                for (int k=kmin;k<=kmax;k++)
                for (int j=jmin;j<=jmax;j++)
                for (int i=imin;i<=imax;i++)
                {
                    if (i==f[0] && j==f[1] && k==f[2]) continue;
                    std::vector<MyRange> & other = compass[i + j*3 + k*9];       

                    for (auto & o:other) if (o.needed)
                    {
                       for (auto & m:me   )
                       {  
                        if (m.needed && m.contains(o) )
                        {
                           o.needed = false;
                           m.removedIndices.push_back(o.index);
                           v.push_back(o.index);
                           break;
                        }
                      }
                    }
                }








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


        //      int icode = code[0] + 1 + 3*(code[1]+1) + 9*(code[2]+1);
        //    	int V = CoarseStencil_L[3*icode  ]*CoarseStencil_L[3*icode+1]*CoarseStencil_L[3*icode+2];


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

    void __FixDuplicates( const Interface & f, const Interface & f_dup, int lx, int ly, int lz, int lx_dup, int ly_dup, int lz_dup, int & sx,int & sy, int & sz)
    {
        BlockInfo & receiver     = * f.    infos[1];
        BlockInfo & receiver_dup = * f_dup.infos[1];
 
        if (receiver.level >= receiver_dup.level )
        {
            int icode_dup = f_dup.icode[1];
            const int code_dup[3] = { icode_dup%3-1, (icode_dup/3)%3-1, (icode_dup/9)%3-1};
            sx = (lx == lx_dup || code_dup[0] != -1) ? 0 :lx - lx_dup;
            sy = (ly == ly_dup || code_dup[1] != -1) ? 0 :ly - ly_dup;
            sz = (lz == lz_dup || code_dup[2] != -1) ? 0 :lz - lz_dup;            
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

    void __FixDuplicates2( const Interface & f, const Interface & f_dup, int & sx,int & sy, int & sz)
    {
        if (f.infos[0]->level != f.infos[1]->level || f_dup.infos[0]->level != f_dup.infos[1]->level) return;
        
        MyRange range,range_dup;
        DetermineStencil(f,range,true);
        DetermineStencil(f_dup,range_dup,true);

        sx =  range_dup.sx-range.sx;
        sy =  range_dup.sy-range.sy;
        sz =  range_dup.sz-range.sz;
    }


    void DefineInterfaces(std::vector<std::vector<Interface>> &send_interfaces, std::vector<std::vector<Interface>> &recv_interfaces)
    {        
        inner_blocks.clear();
        halo_blocks.clear();        
        send_interfaces.resize(size);
        recv_interfaces.resize(size);  


        //        std::vector <int> maxZ (levelMax,-1  );
        //        std::vector <int> minZ (levelMax,100000);
        //
        //        for (int i=0; i<(int)myInfos.size(); i++)
        //        {
        //            BlockInfo & info = myInfos[i];
        //            maxZ[info.level] = std::max(maxZ[info.level],info.Z);
        //            minZ[info.level] = std::min(minZ[info.level],info.Z);
        //        }
  
        for (int i=0; i<(int)myInfos.size(); i++)
        {
            BlockInfo & info = myInfos[i];

            getBlockInfoAll(info.level,info.Z).unpacks.clear();
            info.unpacks.clear();

            int aux = pow(2,info.level);
            const bool xskin = info.index[0]==0 || info.index[0]==blocksPerDim[0]*aux-1;
            const bool yskin = info.index[1]==0 || info.index[1]==blocksPerDim[1]*aux-1;
            const bool zskin = info.index[2]==0 || info.index[2]==blocksPerDim[2]*aux-1;
            const int xskip  = info.index[0]==0 ? -1 : 1;
            const int yskip  = info.index[1]==0 ? -1 : 1;
            const int zskip  = info.index[2]==0 ? -1 : 1;


            bool isInner = true;

            
            std::vector < int > ToBeChecked;
            bool Coarsened = false;


            for(int icode=0; icode<27; icode++)
            {
                if (icode == 1*1 + 3*1 + 9*1) continue;
                const int code[3] = { icode%3-1, (icode/3)%3-1, (icode/9)%3-1};
                if (!xperiodic && code[0] == xskip && xskin) continue;
                if (!yperiodic && code[1] == yskip && yskin) continue;
                if (!zperiodic && code[2] == zskip && zskin) continue; 
                
                //if (!stencil.tensorial && !Cstencil.tensorial && abs(code[0])+abs(code[1])+abs(code[2])>1) continue;
         
                BlockInfo & infoNei = getBlockInfoAll(info.level,info.Znei_(code[0],code[1],code[2]));   
               
             
                //if (infoNei.Z <= maxZ[info.level] && infoNei.Z >= minZ[info.level]) continue; 
    


                if (infoNei.TreePos == Exists && infoNei.myrank != rank)
                {

                    //if (infoNei.Z <= maxZ[info.level] && infoNei.Z >= minZ[info.level])
                    //{
                    //    std::cout << " Rank = " << rank << " infoNei.rank = " << infoNei.myrank << "\n";
                    //    std::cout << " level = " << info.level << "\n";
                    //    std::cout << "max/min Z = " << maxZ[info.level] << " " << minZ[info.level] << "\n";
                    //    std::cout << "info.Z = " << info.Z << "\n";
                    //    std::cout << "infoNei.Z = " << infoNei.Z << "\n";
                    //    abort(); 
                    //}  
    
                  isInner = false;
                  int icode2 = (-code[0]+1) + (-code[1]+1)*3 + (-code[2]+1)*9;

                  Interface FS (info,infoNei,icode,icode2);
                  Interface FR (infoNei,info,icode2,icode);
         
                  send_interfaces[infoNei.myrank].push_back( FS );
                  recv_interfaces[infoNei.myrank].push_back( FR );              

                  ToBeChecked.push_back(infoNei.myrank);
                  ToBeChecked.push_back(send_interfaces[infoNei.myrank].size()-1);
                  ToBeChecked.push_back(recv_interfaces[infoNei.myrank].size()-1);             
                }

                else if (infoNei.TreePos == CheckCoarser)
                {
                	Coarsened = true;

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
            {
                halo_blocks.push_back(info);
            }

            if (Coarsened)
            { 
            	
            	for (int j = 0 ; j <(int) ToBeChecked.size() ; j +=3 )
            	{
            		bool temp  = UseCoarseStencil(send_interfaces[ToBeChecked[j]][ToBeChecked[j+1]]);
          	 		send_interfaces[ToBeChecked[j]][ToBeChecked[j+1]].CoarseStencil = temp;
          	 		recv_interfaces[ToBeChecked[j]][ToBeChecked[j+2]].CoarseStencil = temp;
            	}

            }

        }//i-loop
    }




    std::vector < MyRange > AllStencils;

 
    void DetermineStencil(const Interface & f, MyRange & retval /*range*/, bool CoarseVersion = false)
    {
		auto started =MPI_Wtime();
	     
      	retval.needed = true;
 
        if (CoarseVersion)
        {
        	retval.copy(AllStencils[f.icode[1] + 2*27]);
        }
        else
        {
            if (f.infos[0]->level == f.infos[1]->level)
            {
                retval.copy(AllStencils[f.icode[1]]);
            }
            else if (f.infos[0]->level > f.infos[1]->level)
            {
        	    retval.copy(AllStencils[f.icode[1] + 27]);
            }
            else
            {
		        const int code[3] = { f.icode[1]%3-1, (f.icode[1]/3)%3-1, (f.icode[1]/9)%3-1};
                
            	const int nX = blocksize[0];
                const int nY = blocksize[1];
                const int nZ = blocksize[2];
              
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
                                   
                retval.sx = s[0] + max(code[0],0)*nX/2 + (1-abs(code[0]))*base[0]*nX/2 - code[0]*nX  + CoarseEdge[0] *code[0]*nX/2;     
                retval.sy = s[1] + max(code[1],0)*nY/2 + (1-abs(code[1]))*base[1]*nY/2 - code[1]*nY  + CoarseEdge[1] *code[1]*nY/2;     
                retval.sz = s[2] + max(code[2],0)*nZ/2 + (1-abs(code[2]))*base[2]*nZ/2 - code[2]*nZ  + CoarseEdge[2] *code[2]*nZ/2;    
               
                retval.ex = e[0] + max(code[0],0)*nX/2 + (1-abs(code[0]))*base[0]*nX/2 - code[0]*nX  + CoarseEdge[0] *code[0]*nX/2;
                retval.ey = e[1] + max(code[1],0)*nY/2 + (1-abs(code[1]))*base[1]*nY/2 - code[1]*nY  + CoarseEdge[1] *code[1]*nY/2;
                retval.ez = e[2] + max(code[2],0)*nZ/2 + (1-abs(code[2]))*base[2]*nZ/2 - code[2]*nZ  + CoarseEdge[2] *code[2]*nZ/2;      

            }
        }
		




		#if 0
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
        #endif
   		auto done = MPI_Wtime(); //std::chrono::high_resolution_clock::now();
		TIMINGS [8] += done-started;


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



    void DiscardDuplicates(std::vector<Interface> & f, int & total_size, bool updateMap, int NC, int r)
    {    
        total_size = 0;
        int index = 0;

        std::vector<int> remEl;


        int offset = 0;


        if (f.size()>0) do
        {
        	cube C;
           	int id = f[index].infos[0]->blockID;
            int index_end=f.size();   
            bool skip_needed = false;
            for (int i=index; i<(int)f.size();i++)
                if (f[i].infos[0]->blockID != id)
                {
                    index_end = i;
                    break;
                }
                else
                {
                	MyRange range;
                    range.index = i;
                    DetermineStencil(f[i],range);

                    C.compass[f[i].icode[0]].push_back(range);
                	if (!skip_needed) skip_needed =  f[i].CoarseStencil;
                }
            if (!skip_needed) C.__needed(remEl);


            for (auto & i:C.keepEl())
            {
                int L [3];
                int Lc[3];
                int k = i.index;              
                int code[3] = { f[k].icode[1]%3-1, (f[k].icode[1]/3)%3-1, (f[k].icode[1]/9)%3-1};

                DetermineStencilLength(f[k].infos[0]->level,f[k].infos[1]->level,&code[0],&L[0]);

                int V = L[0]*L[1]*L[2];
                int Vc = 0;

                total_size+= V;
                if (f[k].CoarseStencil)
                {
                	CoarseStencilLength(&code[0],&Lc[0]);
                    Vc = Lc[0]*Lc[1]*Lc[2];
                    total_size += Vc;
                }                    

                if (updateMap)
                {   
                    UnPackInfo info = {offset,L[0],L[1],L[2],0,0,0,L[0],L[1],-1, 0,0,0,0,0,f[k].infos[0]->level,f[k].infos[0]->Z,f[k].icode[1]};
                    offset += V*NC;

                    if (f[k].CoarseStencil)
                    {
                        offset += Vc*NC; 
                        info.CoarseVersionOffset = V*NC;                                       
                        info.CoarseVersionLX = Lc[0];
                        info.CoarseVersionLY = Lc[1];
                    }
                    
					getBlockInfoAll(f[k].infos[1]->level,f[k].infos[1]->Z).unpacks.push_back(info);

                    for (int kk=0; kk< (int)i.removedIndices.size();kk++)
                    {
                        int remEl1 = i.removedIndices[kk];

                        //crap.push_back(i.removedIndices[kk]); //WTF
 
                        code[0] =  f[remEl1].icode[1]   %3-1;
                        code[1] = (f[remEl1].icode[1]/3)%3-1; 
                        code[2] = (f[remEl1].icode[1]/9)%3-1;

                        DetermineStencilLength(f[remEl1].infos[0]->level,f[remEl1].infos[1]->level,&code[0],&L[0]);
                        
                        int srcx, srcy, srcz;

                        __FixDuplicates(f[k],f[remEl1], info.lx,info.ly,info.lz,L[0],L[1],L[2], srcx,srcy,srcz);

                        int Csrcx=0;
                        int Csrcy=0;
                        int Csrcz=0;

                        __FixDuplicates2(f[k],f[remEl1],Csrcx,Csrcy,Csrcz);

             			getBlockInfoAll(f[remEl1].infos[1]->level,f[remEl1].infos[1]->Z).unpacks.push_back({info.offset,L[0],L[1],L[2],srcx, srcy, srcz,info.LX,info.LY,info.CoarseVersionOffset,info.CoarseVersionLX,info.CoarseVersionLY,Csrcx, Csrcy, Csrcz,f[remEl1].infos[0]->level,f[remEl1].infos[0]->Z,f[remEl1].icode[1]});
			        }    
                }
            }

            index = index_end;
        }
        while (index < (int)f.size());


        std::sort(remEl.begin(), remEl.end());
        for (int k=0; k<(int)remEl.size();k++)
        {  
            f.erase(f.begin()+remEl[k]-k);
        }

    }


    bool UseCoarseStencil (Interface & f)
    {
    	auto started1 = MPI_Wtime();
		
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
 
    	auto done1 = MPI_Wtime();
		TIMINGS [5] += done1-started1;

        return retval;
    }

 

    void AverageDownAndFill(Real *  dst, BlockInfo * info, const int s[3], const int e[3], const int code[3], int * selcomponents, int NC, int gptfloats)
    {
        static const int nX = blocksize[0];
        static const int nY = blocksize[1];
        static const int nZ = blocksize[2];

        Real * src = (Real *)(*info).ptrBlock;
      
        const int xStep = (code[0] == 0) ? 2:1;
        const int yStep = (code[1] == 0) ? 2:1;
        const int zStep = (code[2] == 0) ? 2:1;

        int pos =0 ;
  
        for(int iz=s[2]; iz<e[2]; iz+= zStep)
        {
            const int ZZ = (abs(code[2]) == 1) ? 2*(iz- code[2]*nZ) + min(0,code[2])*nZ : iz ;     
            for(int iy=s[1]; iy<e[1]; iy+= yStep)
            {
                const int YY = (abs(code[1]) == 1) ? 2*(iy- code[1]*nY) + min(0,code[1])*nY : iy ;
                for(int ix=s[0]; ix<e[0]; ix+= xStep)
                {
                    const int XX = (abs(code[0]) == 1) ? 2*(ix- code[0]*nX) + min(0,code[0])*nX : ix ;
                       
                    for (int c=0; c<NC; c++)
                    {
                        int comp = selcomponents[c];
        
                        dst[pos]    =0.125*( ( *(src + gptfloats* ((XX  ) + ( (YY  ) + (ZZ  )*nY ) *nX  ) +comp) ) +
                                             ( *(src + gptfloats* ((XX  ) + ( (YY  ) + (ZZ+1)*nY ) *nX  ) +comp) ) +
                                             ( *(src + gptfloats* ((XX  ) + ( (YY+1) + (ZZ  )*nY ) *nX  ) +comp) ) +
                                             ( *(src + gptfloats* ((XX  ) + ( (YY+1) + (ZZ+1)*nY ) *nX  ) +comp) ) +
                                             ( *(src + gptfloats* ((XX+1) + ( (YY  ) + (ZZ  )*nY ) *nX  ) +comp) ) +
                                             ( *(src + gptfloats* ((XX+1) + ( (YY  ) + (ZZ+1)*nY ) *nX  ) +comp) ) +
                                             ( *(src + gptfloats* ((XX+1) + ( (YY+1) + (ZZ  )*nY ) *nX  ) +comp) ) +
                                             ( *(src + gptfloats* ((XX+1) + ( (YY+1) + (ZZ+1)*nY ) *nX  ) +comp) ) );   

                        pos ++ ;
                    }
                }
            }
    	}  
    }






    void AverageDownAndFill2(Real *  dst, BlockInfo * info,  const int code[3], int * selcomponents, int NC, int gptfloats)
    {
        static const int nX = blocksize[0];
        static const int nY = blocksize[1];
        static const int nZ = blocksize[2];

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
         
        int pos =0 ;

        for(int iz=s[2]; iz<e[2]; iz++)
        {
        	const int ZZ =  2*(iz -s[2]) +s[2]+ max(code[2],0)*nZ/2 - code[2]*nZ + min(0,code[2])*(e[2]-s[2]);
            for(int iy=s[1]; iy<e[1]; iy++)
            {
                const int YY =  2*(iy -s[1]) +s[1]+ max(code[1],0)*nY/2 - code[1]*nY + min(0,code[1])*(e[1]-s[1]);
                for(int ix=s[0]; ix<e[0]; ix++)
                {
                    const int XX =  2*(ix -s[0]) +s[0]+ max(code[0],0)*nX/2 - code[0]*nX + min(0,code[0])*(e[0]-s[0]);       
      
                    for (int c=0; c<NC; c++)
                    {
                        int comp = selcomponents[c];
        
                        dst[pos] =0.125*( ( *(src + gptfloats* ((XX  ) + ( (YY  ) + (ZZ  )*nY ) *nX  ) +comp) ) +
                                          ( *(src + gptfloats* ((XX  ) + ( (YY  ) + (ZZ+1)*nY ) *nX  ) +comp) ) +
                                          ( *(src + gptfloats* ((XX  ) + ( (YY+1) + (ZZ  )*nY ) *nX  ) +comp) ) +
                                          ( *(src + gptfloats* ((XX  ) + ( (YY+1) + (ZZ+1)*nY ) *nX  ) +comp) ) +
                                          ( *(src + gptfloats* ((XX+1) + ( (YY  ) + (ZZ  )*nY ) *nX  ) +comp) ) +
                                          ( *(src + gptfloats* ((XX+1) + ( (YY  ) + (ZZ+1)*nY ) *nX  ) +comp) ) +
                                          ( *(src + gptfloats* ((XX+1) + ( (YY+1) + (ZZ  )*nY ) *nX  ) +comp) ) +
                                          ( *(src + gptfloats* ((XX+1) + ( (YY+1) + (ZZ+1)*nY ) *nX  ) +comp) ) );  
          
                        pos ++ ;
                    }
                }
        	}	
   		}


   		//return dst;

    }





public:
    SynchronizerMPI_AMR(StencilInfo a_stencil,StencilInfo a_Cstencil,MPI_Comm a_comm, 
                        const bool a_periodic[3],const int a_levelMax,
                        const int a_nx,const int a_ny,const int a_nz,
                        const int a_bx,const int a_by,const int a_bz,
                        std::vector<BlockInfo> & a_myInfos,
                        std::vector<std::vector<BlockInfo>> & a_BlockInfoAll):
    
    stencil(a_stencil),Cstencil(a_Cstencil), 
    comm(a_comm),xperiodic(a_periodic[0]),yperiodic(a_periodic[1]),zperiodic(a_periodic[2]),
    levelMax(a_levelMax),Zcurve(a_bx,a_by,a_bz),myInfos(a_myInfos),BlockInfoAll(a_BlockInfoAll)
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
    

		if (AllStencils.size() == 0)
		{
			AllStencils.resize(3*27);

		    const int nX = blocksize[0];
	        const int nY = blocksize[1];
	        const int nZ = blocksize[2];


			for (int icode = 0; icode < 27; icode ++)
			{
				const int code[3] = { icode%3-1, (icode/3)%3-1, (icode/9)%3-1};
				MyRange & range = AllStencils[icode];

				range.sx = code[0]<1? (code[0]<0 ? nX + stencil.sx:0 ):0;
                range.sy = code[1]<1? (code[1]<0 ? nY + stencil.sy:0 ):0;
                range.sz = code[2]<1? (code[2]<0 ? nZ + stencil.sz:0 ):0;
                range.ex = code[0]<1? (code[0]<0 ? nX             :nX):stencil.ex-1; 
                range.ey = code[1]<1? (code[1]<0 ? nY             :nY):stencil.ey-1; 
                range.ez = code[2]<1? (code[2]<0 ? nZ             :nZ):stencil.ez-1; 
			}


			for (int icode = 0; icode < 27; icode ++)
			{
				const int code[3] = { icode%3-1, (icode/3)%3-1, (icode/9)%3-1};
				MyRange & range = AllStencils[icode + 27];

				range.sx =  code[0]<1? (code[0]<0 ? nX + 2*stencil.sx:0 ): 0;
                range.sy =  code[1]<1? (code[1]<0 ? nY + 2*stencil.sy:0 ): 0;
                range.sz =  code[2]<1? (code[2]<0 ? nZ + 2*stencil.sz:0 ): 0;          
                range.ex =  code[0]<1? (code[0]<0 ? nX               :nX): 2*stencil.ex-1;
                range.ey =  code[1]<1? (code[1]<0 ? nY               :nY): 2*stencil.ey-1;
                range.ez =  code[2]<1? (code[2]<0 ? nZ               :nZ): 2*stencil.ez-1;
			}

			for (int icode = 0; icode < 27; icode ++)
			{
				const int code[3] = { icode%3-1, (icode/3)%3-1, (icode/9)%3-1};
				MyRange & range = AllStencils[icode + 2*27];

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
		}
    }

    double TIMINGS[20];

    std::vector<BlockInfo> avail_inner()
    {
        return inner_blocks;
    }

    std::vector<BlockInfo> avail_halo()
    {
        MPI_Waitall(size, &recv_requests[0], MPI_STATUSES_IGNORE);
        MPI_Waitall(size, &send_requests[0], MPI_STATUSES_IGNORE);
        return halo_blocks;
    }


    void sync(unsigned int gptfloats, MPI_Datatype MPIREAL, const int timestamp)
    { 
        for (int i=0;i<20;i++) TIMINGS[i] = 0;

        const int nX = blocksize[0];
        const int nY = blocksize[1];
        const int nZ = blocksize[2];
        
        std::vector<int> selcomponents = stencil.selcomponents;
        std::sort(selcomponents.begin(), selcomponents.end());
        
        const int NC = selcomponents.size();

        //1.Find all interfaces with neighboring ranks

        auto started = MPI_Wtime();
  
        std::vector<std::vector<Interface>> send_interfaces;
        std::vector<std::vector<Interface>> recv_interfaces;
        DefineInterfaces(send_interfaces,recv_interfaces); 

        auto done = MPI_Wtime();
  		TIMINGS [0] += done-started;
           
  
  		started = MPI_Wtime();    
        //2.Sort interfaces 
        for (int r=0; r<size; r++)
        {
            std::sort (send_interfaces[r].begin(), send_interfaces[r].end());
            std::sort (recv_interfaces[r].begin(), recv_interfaces[r].end());
        }   

        done = MPI_Wtime();
  		TIMINGS [1] += done-started;

      
		started = MPI_Wtime();
        //3.Determine buffer sizes
        send_buffer.resize(size);
        recv_buffer.resize(size);
        std::vector<int> send_buffer_size(size,0);
        std::vector<int> recv_buffer_size(size,0);
        for (int r=0; r<size; r++)
        {
            DiscardDuplicates(send_interfaces[r],send_buffer_size[r],false,NC,r);
            DiscardDuplicates(recv_interfaces[r],recv_buffer_size[r],true ,NC,r);   
            send_buffer[r].resize(send_buffer_size[r]*NC, 666.0);
            recv_buffer[r].resize(recv_buffer_size[r]*NC, 777.0);          
        }    
		done = MPI_Wtime();
  		TIMINGS [2] += done-started;
        



  		started = MPI_Wtime();
        //4.Pack data         
        send_packinfos.resize(size);
        std::vector<int> displacement(size,0);
        for (int r=0; r<size; r++)
        {
            for (int i=0; i<(int) send_interfaces[r].size(); i++)
            {
                Interface & f = send_interfaces[r][i];

                const int code[3] = { f.icode[0]%3-1, (f.icode[0]/3)%3-1, (f.icode[0]/9)%3-1};              
                const int code0[3] = { -code[0], -code[1] , -code[2]}; 

                if (f.infos[0]->level <= f.infos[1]->level)
                {
                    MyRange range;
                    DetermineStencil(f,range);
                    int V = (range.ex-range.sx)* (range.ey-range.sy)* (range.ez-range.sz);     
                    PackInfo info_tmp = {(Real *)f.infos[0]->ptrBlock, &send_buffer[r][ displacement[r] ], range.sx,range.sy,range.sz,range.ex,range.ey,range.ez};      
                    send_packinfos[r].push_back(info_tmp);              
                    displacement[r]+= V*NC;
                    
                    if (f.CoarseStencil) 
                    {              
                        AverageDownAndFill2(send_buffer[r].data() + displacement[r],f.infos[0],code0,&selcomponents[0],NC,gptfloats);
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

                    AverageDownAndFill(send_buffer[r].data() + displacement[r],f.infos[0],s,e,code0,&selcomponents[0],NC, gptfloats); 
                                      
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
      	done = MPI_Wtime();
  		TIMINGS [3] += done-started;
    
            
        send_requests.resize(size);
        recv_requests.resize(size);

        for (int r = 0 ; r < size; r ++ )
        {
            MPI_Irecv(&recv_buffer[r][0], recv_buffer_size[r]*NC, MPIREAL, r, timestamp , comm, &recv_requests[r]);
            MPI_Isend(&send_buffer[r][0], send_buffer_size[r]*NC, MPIREAL, r, timestamp , comm, &send_requests[r]);
        }     
    }


    StencilInfo getstencil() const
    {
        return stencil;
    }



    void fetch (const BlockInfo& info, const int gptfloats, const size_t Length[3], const size_t CLength[3], const size_t ElemsPerSlice [2], Real * cacheBlock, Real *coarseBlock)
    {
        static const int nX = blocksize[0];
        static const int nY = blocksize[1];
        static const int nZ = blocksize[2];

		std::vector <UnPackInfo > & unpacks = getBlockInfoAll(info.level,info.Z).unpacks;

  
        for (auto & unpack : unpacks)
        {       	
        	const int code[3] = { unpack.icode%3-1, (unpack.icode/3)%3-1, (unpack.icode/9)%3-1};

        	BlockInfo & other = getBlockInfoAll(unpack.level,unpack.Z);

            const int s[3] = { code[0]<1? (code[0]<0 ? stencil.sx:0 ):nX,
                               code[1]<1? (code[1]<0 ? stencil.sy:0 ):nY,
                               code[2]<1? (code[2]<0 ? stencil.sz:0 ):nZ};
            const int e[3] = { code[0]<1? (code[0]<0 ? 0         :nX):nX+ stencil.ex-1,
                               code[1]<1? (code[1]<0 ? 0         :nY):nY+ stencil.ey-1,
                               code[2]<1? (code[2]<0 ? 0         :nZ):nZ+ stencil.ez-1};

        	if (other.level == info.level)
        	{             
                Real * dst =  cacheBlock + ( (s[2]-stencil.sz)*ElemsPerSlice[0] + (s[1]-stencil.sy)*Length[0] + s[0]-stencil.sx  )*gptfloats;

                unpack_subregion<Real>(&recv_buffer[other.myrank][unpack.offset],
                &dst[0],gptfloats,&stencil.selcomponents[0],stencil.selcomponents.size(),                    
                unpack.srcxstart,unpack.srcystart,unpack.srczstart,unpack.LX,unpack.LY,       
                0,0,0,unpack.lx,unpack.ly,unpack.lz,Length[0],Length[1],Length[2]);
                      
                
                if (unpack.CoarseVersionOffset != 0)
                {
                    const int sC[3] = {(stencil.sx-1)/2+ Cstencil.sx,
                                       (stencil.sy-1)/2+ Cstencil.sy,
                                       (stencil.sz-1)/2+ Cstencil.sz};
                   
                   const int s1[3] = { code[0]<1? (code[0]<0 ? sC[0]:0  ) : nX/2,
                                       code[1]<1? (code[1]<0 ? sC[1]:0  ) : nY/2,
                                       code[2]<1? (code[2]<0 ? sC[2]:0  ) : nZ/2};

                    Real * dst1 = coarseBlock + ((s1[2]-sC[2])*ElemsPerSlice[1] + (s1[1]-sC[1])*CLength[0] + s1[0]-sC[0])*gptfloats;

              	    int L[3];
              	    int code__[3] = {-code[0],-code[1],-code[2]};
          			CoarseStencilLength(&code__[0],&L[0]);

                    unpack_subregion<Real>(&recv_buffer[other.myrank][unpack.offset+unpack.CoarseVersionOffset],
                    &dst1[0],gptfloats,&stencil.selcomponents[0],stencil.selcomponents.size(),                               
                    unpack.CoarseVersionsrcxstart,unpack.CoarseVersionsrcystart,unpack.CoarseVersionsrczstart,
                    unpack.CoarseVersionLX,unpack.CoarseVersionLY,
                    0,0,0,L[0],L[1],L[2],CLength[0],CLength[1],CLength[2]);
                }
        	}
        	else if (other.level < info.level)
        	{
                const int sC[3] = {code[0]<1? (code[0]<0 ? ((stencil.sx-1)/2+ Cstencil.sx) :0 ) : nX/2,
                                   code[1]<1? (code[1]<0 ? ((stencil.sy-1)/2+ Cstencil.sy) :0 ) : nY/2,
                                   code[2]<1? (code[2]<0 ? ((stencil.sz-1)/2+ Cstencil.sz) :0 ) : nZ/2 };              

                const int offset[3] = {(stencil.sx-1)/2+ Cstencil.sx, (stencil.sy-1)/2+ Cstencil.sy, (stencil.sz-1)/2+ Cstencil.sz};

                Real * dst = coarseBlock + ((sC[2]-offset[2])*ElemsPerSlice[1] + sC[0]-offset[0] + (sC[1]-offset[1])* CLength[0])*gptfloats;
                   
            	unpack_subregion<Real>(&recv_buffer[other.myrank][unpack.offset],
                &dst[0],gptfloats,&stencil.selcomponents[0],stencil.selcomponents.size(),                    
                unpack.srcxstart,unpack.srcystart,unpack.srczstart,unpack.LX,unpack.LY,
                0,0,0,unpack.lx,unpack.ly,unpack.lz,CLength[0],CLength[1],CLength[2]);
        	}
        	else 
        	{
        		int B;
                if ((abs(code[0])+abs(code[1])+abs(code[2])==3 )) B = 0;//corner
                else if ((abs(code[0])+abs(code[1])+abs(code[2])==2 )) //edge
                {
                	int t;
                	if ( code[0] == 0 )
                	{
                		t = other.index[0] - (2*info.index[0] + max(code[0],0) +code[0]);
                	}
                	else if (code[1] == 0)
                	{
	               		t = other.index[1] - (2*info.index[1] + max(code[1],0) +code[1]);
                	}
                	else // if (code[2] ==0)
                	{
                		t = other.index[2] - (2*info.index[2] + max(code[2],0) +code[2]);
                	}

                	assert (t == 0 || t == 1);
                	if (t == 1) B = 3;
                	else        B = 0;
                }
                else
                {
                	int Bmod,Bdiv;
                	if ( abs(code[0]) == 1 )
                	{
                		Bmod = other.index[1] - (2*info.index[1] + max(code[1],0) +code[1]);
                		Bdiv = other.index[2] - (2*info.index[2] + max(code[2],0) +code[2]);
                	}
                	else if (abs(code[1]) == 1)
       	         	{
                		Bmod = other.index[0] - (2*info.index[0] + max(code[0],0) +code[0]);
                		Bdiv = other.index[2] - (2*info.index[2] + max(code[2],0) +code[2]);
                	}
                	else
       	         	{
                		Bmod = other.index[0] - (2*info.index[0] + max(code[0],0) +code[0]);
                		Bdiv = other.index[1] - (2*info.index[1] + max(code[1],0) +code[1]);
                	}
       
                	B = 2*Bdiv + Bmod;
                }              
                const int aux1 = (abs(code[0])==1) ? (B%2) : (B/2) ;
                
              	int iz = s[2];
               	int iy = s[1];

               	const int my_ix =  abs(code[0])*(s[0]-stencil.sx) + (1-abs(code[0]) )*(  s[0]  -stencil.sx + (B%2)*(e[0]-s[0])/2);   
                const int m_vSize0         = Length[0];
                const int m_nElemsPerSlice = ElemsPerSlice[0];
                const int my_izx = ( abs(code[2])*(iz-stencil.sz) + (1-abs(code[2]) )*(iz/2-stencil.sz + (B/2)*(e[2]-s[2])/2)  )*m_nElemsPerSlice + my_ix;
       
              	Real * dst = cacheBlock +  (my_izx + ( abs(code[1])*(iy-stencil.sy) + (1-abs(code[1]) )*(iy/2-stencil.sy + aux1*(e[1]-s[1])/2)  )*m_vSize0) * gptfloats;
              
                unpack_subregion<Real>(&recv_buffer[other.myrank][unpack.offset],
                &dst[0],gptfloats,
                &stencil.selcomponents[0],
                stencil.selcomponents.size(),                    
                unpack.srcxstart,unpack.srcystart,unpack.srczstart,
                unpack.LX,unpack.LY,       
                0,0,0,unpack.lx,unpack.ly,unpack.lz,
                Length[0],
                Length[1],
                Length[2]);
        	}  
        }   
    }

};


CUBISM_NAMESPACE_END
