/*
 *  GridMPI.h
 *  FacesMPI
 *
 *  Created by Diego Rossinelli on 10/21/11.
 *  Copyright 2011 ETH Zurich. All rights reserved.
 *
 */
#pragma once

#include <vector>
#include <map>
#include <mpi.h>
#include <set>

#include "BlockInfo.h"
#include "StencilInfo.h"
#include "AMR_SynchronizerMPI.h"



#include <chrono>

CUBISM_NAMESPACE_BEGIN

template < typename TGrid >
class GridMPI : public TGrid
{
public:
    typedef typename TGrid::Real Real;

    double TIMINGS [100];


private:
    size_t timestamp;

protected:
    typedef SynchronizerMPI_AMR<Real> SynchronizerMPIType;
    //friend class SynchronizerMPI_AMR<Real>;


    std::vector <BlockInfo *> boundary_blocks; 


    int myrank, mypeindex[3], pesize[3];
    int periodic[3];
    int mybpd[3], myblockstotalsize, blocksize[3];

    std::vector<BlockInfo> cached_blockinfo;


    MPI_Comm worldcomm;
    MPI_Comm cartcomm;

    // Subdomain handled by this node.
    double subdomain_low[3];
    double subdomain_high[3];
public:

    typedef typename TGrid::BlockType Block;
    typedef typename TGrid::BlockType BlockType;


    std::map<StencilInfo, SynchronizerMPIType *> SynchronizerMPIs;



    GridMPI(const int npeX, const int npeY, const int npeZ,
            const int nX, const int nY=1, const int nZ=1,
            const double _maxextent = 1, const int a_levelStart = 0, const int a_levelMax =1 ,const MPI_Comm comm = MPI_COMM_WORLD, const bool a_xperiodic = true,const bool a_yperiodic = true,const bool a_zperiodic = true):
   

      TGrid(nX, nY, nZ, _maxextent, a_levelStart,a_levelMax,false, a_xperiodic,a_yperiodic,a_zperiodic), timestamp(0), worldcomm(comm)
    {
        assert(npeX > 0 && "Number of processes per X must be greater than 0.");
        assert(npeY > 0 && "Number of processes per Y must be greater than 0.");
        assert(npeZ > 0 && "Number of processes per Z must be greater than 0.");
        blocksize[0] = Block::sizeX;
        blocksize[1] = Block::sizeY;
        blocksize[2] = Block::sizeZ;

        
        mybpd[0] = nX;
        mybpd[1] = nY;
        mybpd[2] = nZ;
        myblockstotalsize = nX*nY*nZ;

        periodic[0] = true;
        periodic[1] = true;
        periodic[2] = true;

        pesize[0] = npeX;
        pesize[1] = npeY;
        pesize[2] = npeZ;

        int world_size;
        MPI_Comm_size(worldcomm, &world_size);
        
        cartcomm = worldcomm;
        mypeindex[0] = 0;
        mypeindex[1] = 0;
        mypeindex[2] = 0;

        
        for (int k=0;k<100;k++)
            TIMINGS[k]=0;

        int total_blocks = nX*nY*nZ*pow(pow(2,a_levelStart),3);
        MPI_Comm_rank(worldcomm, &myrank);   

        if (myrank==0)
        	std::cout << "Total blocks = " << total_blocks <<"\n";

        int my_blocks = total_blocks / world_size;         
        if (myrank < total_blocks % world_size) my_blocks ++;    
        int n_start = myrank* (total_blocks / world_size);

        if (total_blocks % world_size > 0)
        {
            if (myrank < total_blocks % world_size) n_start+= myrank;
            else                                    n_start+= total_blocks % world_size;
        }

        std::cout << "rank " << myrank << " gets " << my_blocks << " \n"; 

        for (int n=n_start; n<n_start+my_blocks; n++)
            TGrid::_alloc(a_levelStart,n);       

        for (int m = 0 ; m < a_levelMax ; m++)
        for (int n=0; n<nX*nY*nZ*pow(pow(2,m),3); n++)
        {
            if (m==a_levelStart)
            {
                int r;
                if (total_blocks % world_size > 0)
                {
                    if (n + 1 > (total_blocks / world_size + 1) * (total_blocks % world_size) )
                    {
                        int aux = (total_blocks / world_size + 1) * (total_blocks % world_size);
                        
                        r = (n - aux) / (total_blocks / world_size) + total_blocks % world_size;
                    }
                    else
                    {
                        r = n /(total_blocks / world_size + 1);
                    }
             
                }
                else
                {
                    r = n / my_blocks;
                }
              
                TGrid::BlockInfoAll[m][n].myrank = r; 
            }
            else
            {
                TGrid::BlockInfoAll[m][n].myrank = -1;   
            }          
        }

        FillPos(); 

        #if 0

                const std::vector<BlockInfo> vInfo = TGrid::getBlocksInfo();
                // Doesn't make sense to export `h_gridpoint` and `h_block` as a member
                // variable + getter, as they are not fixed values in case of
                // non-uniform grids.
                const double h_gridpoint = _maxextent / (double)std::max(
                        getBlocksPerDimension(0) * blocksize[0],
                        std::max(getBlocksPerDimension(1) * blocksize[1],
                                 getBlocksPerDimension(2) * blocksize[2]));
                const double h_block[3] = {
                    blocksize[0] * h_gridpoint,
                    blocksize[1] * h_gridpoint,
                    blocksize[2] * h_gridpoint,
                };
                // setup uniform (global) meshmaps
                // discard single process mappings
                for (int i = 0; i < 3; ++i)
                    delete this->m_mesh_maps[i];
                std::vector<MeshMap<Block>*> clearme;
                this->m_mesh_maps.swap(clearme);
                // global number of blocks and extents
                const int nBlocks[3] = {
                    mybpd[0]*pesize[0],
                    mybpd[1]*pesize[1],
                    mybpd[2]*pesize[2]
                };
                const double extents[3] = {
                    h_block[0]*nBlocks[0],
                    h_block[1]*nBlocks[1],
                    h_block[2]*nBlocks[2]
                };
                for (int i = 0; i < 3; ++i)
                {
                    MeshMap<Block>* m = new MeshMap<Block>(0.0, extents[i], nBlocks[i]);
                    UniformDensity uniform;
                    m->init(&uniform); // uniform only for this constructor
                    this->m_mesh_maps.push_back(m);
                }
		#endif
        std::cout << "mike: GridMPI::skipping cached_blockinfo initialization...\n";
        #if 0
                for(size_t i=0; i<vInfo.size(); ++i)
                {
                    BlockInfo info = vInfo[i];
                    info.h_gridpoint = h_gridpoint;
                    info.h = h_block[0];// only for blocksize[0]=blocksize[1]=blocksize[2]
        
                    for(int j=0; j<3; ++j)
                    {
                        info.index[j] += mypeindex[j]*mybpd[j];
        
                        info.origin[j] = this->m_mesh_maps[j]->block_origin(info.index[j]);
        
                        info.uniform_grid_spacing[j] = h_gridpoint;
        
                        info.block_extent[j] = this->m_mesh_maps[j]->block_width(info.index[j]);
        
                        info.ptr_grid_spacing[j] = this->m_mesh_maps[j]->get_grid_spacing(info.index[j]);
                    }
                    cached_blockinfo.push_back(info);
                }
        #endif
        MPI_Barrier(MPI_COMM_WORLD);
        std::cout<<"GridMPI constructor called (ok)\n";
    }


    GridMPI(const MeshMap<Block>* const mapX,
            const MeshMap<Block>* const mapY,
            const MeshMap<Block>* const mapZ,
            const int npeX, const int npeY, const int npeZ,
            const int nX=0, const int nY=0, const int nZ=0,
            const MPI_Comm comm = MPI_COMM_WORLD): TGrid(mapX,mapY,mapZ,nX,nY,nZ), timestamp(0), worldcomm(comm)
    {
        assert(false && "GridMPI MeshMap constructor");
        assert(this->m_mesh_maps[0]->nblocks() == static_cast<size_t>(npeX*nX));
        assert(this->m_mesh_maps[1]->nblocks() == static_cast<size_t>(npeY*nY));
        assert(this->m_mesh_maps[2]->nblocks() == static_cast<size_t>(npeZ*nZ));

        blocksize[0] = Block::sizeX;
        blocksize[1] = Block::sizeY;
        blocksize[2] = Block::sizeZ;

        mybpd[0] = nX;
        mybpd[1] = nY;
        mybpd[2] = nZ;
        myblockstotalsize = nX*nY*nZ;

        periodic[0] = true;
        periodic[1] = true;
        periodic[2] = true;

        pesize[0] = npeX;
        pesize[1] = npeY;
        pesize[2] = npeZ;

        int world_size;
        MPI_Comm_size(worldcomm, &world_size);
        assert(npeX*npeY*npeZ == world_size);

        MPI_Cart_create(worldcomm, 3, pesize, periodic, true, &cartcomm);
        MPI_Comm_rank(cartcomm, &myrank);
        MPI_Cart_coords(cartcomm, myrank, 3, mypeindex);

        const std::vector<BlockInfo> vInfo = TGrid::getBlocksInfo();

        for(size_t i=0; i<vInfo.size(); ++i)
        {
            BlockInfo info = vInfo[i];

            for(int j=0; j<3; ++j)
            {
                info.index[j] += mypeindex[j]*mybpd[j];

                info.origin[j] = this->m_mesh_maps[j]->block_origin(info.index[j]);

                info.block_extent[j] = this->m_mesh_maps[j]->block_width(info.index[j]);

                info.ptr_grid_spacing[j] = this->m_mesh_maps[j]->get_grid_spacing(info.index[j]);
            }

            cached_blockinfo.push_back(info);
        }
    }


    virtual ~GridMPI() override
    { 
        for (auto it = SynchronizerMPIs.begin(); it != SynchronizerMPIs.end(); ++it)
            delete it->second;

        SynchronizerMPIs.clear();
        //MPI_Comm_free(&cartcomm);

       	MPI_Barrier(MPI_COMM_WORLD);



        double res [100];
        MPI_Reduce(&TIMINGS[0], &res[0] , 100, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (myrank == 0)
        {
        	std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
            std::cout << "TIMINGS:\n";

            printf( "UpdateBlockInfoAll_States :  %6.2f \n",res[0]);
            printf( "SynchronizerMPI sync      :  %6.2f \n",res[1]);
            printf( "FillPos                   :  %6.2f \n",res[2]);
            printf( "FillBlockCases            :  %6.2f \n",res[3]);
            printf( "Kernels (inner)           :  %6.2f \n",res[4]);
            printf( "Kernels (outer)           :  %6.2f \n",res[5]);
            printf( "PrepareCompression        :  %6.2f \n",res[6]);
            printf( "BalanceDiffusion          :  %6.2f \n",res[7]);


            std::cout <<  "\n";
            printf( "---> FluxCorrection::prepare     : %6.3f \n" , res[9 ]);
            printf( "---> FluxCorrection::pack data   : %6.3f \n" , res[10]);
            printf( "---> FluxCorrection::inner faces : %6.3f \n" , res[11]);
            printf( "---> FluxCorrection::waiting     : %6.3f \n" , res[12]);
            printf( "---> FluxCorrection::outer faces : %6.3f \n" , res[13]);
            printf( "---> FluxCorrection::correct     : %6.3f \n" , res[14]);
            printf( "---> FluxCorrection::            : %6.3f \n" , res[15]);
            printf( "---> FluxCorrection::            : %6.3f \n" , res[16]);
            printf( "---> FluxCorrection::            : %6.3f \n" , res[17]);
            printf( "---> FluxCorrection::            : %6.3f \n" , res[18]);


            std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
        }

        MPI_Barrier(MPI_COMM_WORLD);

        _deallocAll();
    }




    virtual void _deallocAll() override //called in class destructor
    {
        TGrid::m_blocks.clear();
        TGrid::m_vInfo.clear();
        for (int m=0; m<TGrid::levelMax; m++)
        {
            for (int n=0; n<TGrid::NX*TGrid::NY*TGrid::NZ*pow(pow(2,m),3); n++)
            {
                if (TGrid::BlockInfoAll[m][n].TreePos==Exists && TGrid::BlockInfoAll[m][n].myrank == myrank) 
                {
                    allocator <Block> alloc;
                    alloc.deallocate((Block*)TGrid::BlockInfoAll[m][n].ptrBlock,1);
                }
            }
        }    
    }



    std::vector<BlockInfo>& getBlocksInfo() override
    {
        cached_blockinfo = TGrid::getBlocksInfo();
        return cached_blockinfo;
    }

    const std::vector<BlockInfo>& getBlocksInfo() const override
    {
        return TGrid::getBlocksInfo();//cached_blockinfo;
    }

    std::vector<BlockInfo>& getResidentBlocksInfo()
    {
        return TGrid::getBlocksInfo();
    }

    const std::vector<BlockInfo>& getResidentBlocksInfo() const
    {
        return TGrid::getBlocksInfo();
    }

    virtual bool avail(int ix, int iy, int iz, int m) const override
    {
        int n = TGrid::getZforward(m,ix,iy,iz);
        if (TGrid::BlockInfoAll[m][n].myrank == myrank) return true;
        else                                            return false;
    }


    virtual void UpdateBlockInfoAll_States(bool GlobalUpdate = true) 
    {
    
    double started = MPI_Wtime();
    
    if (GlobalUpdate) //stupid global update of everyting
    {    
        int rank,size;
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        MPI_Comm_size(MPI_COMM_WORLD,&size); 


        int myLength = 3*TGrid::m_vInfo.size();
        std::vector <int> myData(myLength);
        for (int i=0; i<myLength; i+=3)
        {
            myData[i  ] = TGrid::m_vInfo[i/3].level;
            myData[i+1] = TGrid::m_vInfo[i/3].Z    ;


            if (TGrid::m_vInfo[i/3].state == Leave)
                myData[i+2] = 0;
            else if (TGrid::m_vInfo[i/3].state == Compress)
                myData[i+2] = 1;
            else if (TGrid::m_vInfo[i/3].state == Refine)
                myData[i+2] = 2;
        }

        //2.Gather lengths of all processes and use them to allocate memory on each process
        std::vector <int> AllLengths(size,0);
        AllLengths[rank] = myLength;
        MPI_Allgather(&myLength, 1, MPI_INT, &AllLengths[0], 1, MPI_INT,MPI_COMM_WORLD);


        std::vector< std::vector<int> > AllData(size);
        for (int i=0;i<size;i++)
            AllData[i].resize(AllLengths[i]);
        

        std::vector<int> displacement(size);
        displacement[0] = 0;
        for (int i=1;i<size;i++)
        {
            displacement[i] = &AllData[i][0]-&AllData[0][0];    
        }
    

        MPI_Allgatherv(&myData[0], myLength, MPI_INT,
                       &AllData[0][0], &AllLengths[0],
                       &displacement[0], MPI_INT, MPI_COMM_WORLD);

 
        for (int r=0 ; r<size; r++)
        for (int index = 0; index < (int)AllData[r].size(); index += 3)
        {
            int level = AllData[r][index  ];
            int Z     = AllData[r][index+1];
            
            //State s   = (State)AllData[r][index+2];
     
            TGrid::BlockInfoAll[level][Z].myrank  = r;
            TGrid::BlockInfoAll[level][Z].TreePos = Exists;
            

            if (AllData[r][index+2] == 0)
            TGrid::BlockInfoAll[level][Z].state   = Leave;

            else if (AllData[r][index+2] == 1)
            TGrid::BlockInfoAll[level][Z].state   = Compress;

            else if (AllData[r][index+2] == 2)
            TGrid::BlockInfoAll[level][Z].state   = Refine;





            int p[3] = {TGrid::BlockInfoAll[level][Z].index[0],
                        TGrid::BlockInfoAll[level][Z].index[1],
                        TGrid::BlockInfoAll[level][Z].index[2]};
           
            if (level<TGrid::levelMax -1)
                for (int k=0; k<2; k++ )
                for (int j=0; j<2; j++ )
                for (int i=0; i<2; i++ )
                {      
                    int nc = TGrid::getZforward(level+1,2*p[0]+i,2*p[1]+j,2*p[2]+k);
                    TGrid::BlockInfoAll[level+1][nc].TreePos = CheckCoarser;
                    TGrid::BlockInfoAll[level+1][nc].myrank  = -1;
                }
            if (level>0)
            {
                int nf = TGrid::getZforward(level-1,p[0]/2,p[1]/2,p[2]/2);
                TGrid::BlockInfoAll[level-1][nf].TreePos = CheckFiner;
                TGrid::BlockInfoAll[level-1][nf].myrank  = -1;
            }
        }
    }    
    else //update blocks on the boundary of process
    {    
        int rank,size;
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        MPI_Comm_size(MPI_COMM_WORLD,&size); 
     
        std::vector < std::vector <int> > send_buffer(size);
    
        auto blocksPerDim = TGrid::getMaxBlocks(); 
      
        for (auto & info: TGrid::m_vInfo)
        {
            int aux = pow(2,info.level);

            const bool xskin = info.index[0]==0 || info.index[0]==blocksPerDim[0]*aux-1;
            const bool yskin = info.index[1]==0 || info.index[1]==blocksPerDim[1]*aux-1;
            const bool zskin = info.index[2]==0 || info.index[2]==blocksPerDim[2]*aux-1;
            const int  xskip = info.index[0]==0 ? -1 : 1;
            const int  yskip = info.index[1]==0 ? -1 : 1;
            const int  zskip = info.index[2]==0 ? -1 : 1;

            bool isInner = true;
            

            std::set   <int> receivers;
          
            for(int icode=0; icode<27; icode++)
            {
                if (icode == 1*1 + 3*1 + 9*1) continue;
               
                const int code[3] = { icode%3-1, (icode/3)%3-1, (icode/9)%3-1};
               
                if (!TGrid::xperiodic && code[0] == xskip && xskin) continue;
                if (!TGrid::yperiodic && code[1] == yskip && yskin) continue;
                if (!TGrid::zperiodic && code[2] == zskip && zskin) continue; 
                
                BlockInfo & infoNei = TGrid::getBlockInfoAll(info.level,info.Znei_(code[0],code[1],code[2]));   

                if (infoNei.TreePos == Exists && infoNei.myrank != rank)
                {
                  isInner = false;           
                  receivers.insert(infoNei.myrank);
                }
                else if (infoNei.TreePos == CheckCoarser)
                {
                    int nCoarse = TGrid::getZforward(infoNei.level-1,infoNei.index[0]/2,infoNei.index[1]/2,infoNei.index[2]/2);
                    BlockInfo & infoNeiCoarser = TGrid::getBlockInfoAll(infoNei.level-1,nCoarse);
                    if (infoNeiCoarser.myrank != rank)
                    {
                        isInner = false;                 
                        receivers.insert(infoNeiCoarser.myrank);
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
                        int nFine = TGrid::getZforward(infoNei.level+1,2*info.index[0] + max(code[0],0) +code[0]  + (B%2)*max(0, 1 - abs(code[0])),
                                                                       2*info.index[1] + max(code[1],0) +code[1]  + temp *max(0, 1 - abs(code[1])),
                                                                       2*info.index[2] + max(code[2],0) +code[2]  + (B/2)*max(0, 1 - abs(code[2])));
                        BlockInfo & infoNeiFiner = TGrid::getBlockInfoAll(infoNei.level+1,nFine);
                        if (infoNeiFiner.myrank != rank)
                        {
                            isInner = false; 
                            receivers.insert(infoNeiFiner.myrank);
                        }
                    }
                }
            }//icode = 0,...,26  

            if (!isInner)
            {
                std::set<int>::iterator it = receivers.begin();
                while (it != receivers.end())
                {
                    int temp;
                    if (info.state == Leave)
                        temp = 0;
                    else if (info.state == Compress)
                        temp = 1;
                    else 
                        temp = 2;

                    send_buffer[*it].push_back(info.level);
                    send_buffer[*it].push_back(info.Z    );
                    send_buffer[*it].push_back(temp      );
                    it++;
                }
            }
        }


        std::vector<MPI_Request> send_requests;
        std::vector<MPI_Request> recv_requests;
   

        

        std::set <int> receivers;

        for (int r=0; r<size; r++) if (send_buffer[r].size() != 0)
        {
            receivers.insert(r);

            send_requests.resize(send_requests.size()+1);
            MPI_Isend(&send_buffer[r][0], send_buffer[r].size(), MPI_INT, r, 123 , MPI_COMM_WORLD, &send_requests[send_requests.size()-1]);
        }

          
        std::vector < std::vector <int> > recv_buffer(size);

        std::set<int>::iterator it = receivers.begin();
        while (it != receivers.end())
        {   
            int r = *it;         
            int recv_size;
            MPI_Status status;          
            MPI_Probe(r, 123, MPI_COMM_WORLD,&status);
            MPI_Get_count(&status, MPI_INT, &recv_size);
            recv_buffer[r].resize(recv_size);
            recv_requests.resize(recv_requests.size()+1);
            MPI_Irecv(&recv_buffer[r][0], recv_buffer[r].size(), MPI_INT, r, 123 , MPI_COMM_WORLD, &recv_requests[recv_requests.size()-1]);
        
            it++;
        }

        MPI_Waitall(send_requests.size(), &send_requests[0], MPI_STATUSES_IGNORE);     
        MPI_Waitall(recv_requests.size(), &recv_requests[0], MPI_STATUSES_IGNORE);
       













        for (int r=0 ; r<size; r++)
        for (int index = 0; index < (int)recv_buffer[r].size(); index += 3)
        {
            int level = recv_buffer[r][index  ];
            int Z     = recv_buffer[r][index+1];
           
            TGrid::BlockInfoAll[level][Z].myrank  = r;
            TGrid::BlockInfoAll[level][Z].TreePos = Exists;

            if (recv_buffer[r][index+2] == 0)
            TGrid::BlockInfoAll[level][Z].state   = Leave;

            else if (recv_buffer[r][index+2] == 1)
            TGrid::BlockInfoAll[level][Z].state   = Compress;

            else if (recv_buffer[r][index+2] == 2)
            TGrid::BlockInfoAll[level][Z].state   = Refine;


            int p[3] = {TGrid::BlockInfoAll[level][Z].index[0],
                        TGrid::BlockInfoAll[level][Z].index[1],
                        TGrid::BlockInfoAll[level][Z].index[2]};
           
            if (level<TGrid::levelMax -1)
                for (int k=0; k<2; k++ )
                for (int j=0; j<2; j++ )
                for (int i=0; i<2; i++ )
                {      
                    int nc = TGrid::getZforward(level+1,2*p[0]+i,2*p[1]+j,2*p[2]+k);
                    TGrid::BlockInfoAll[level+1][nc].TreePos = CheckCoarser;
                    TGrid::BlockInfoAll[level+1][nc].myrank  = -1;
                }
            if (level>0)
            {
                int nf = TGrid::getZforward(level-1,p[0]/2,p[1]/2,p[2]/2);
                TGrid::BlockInfoAll[level-1][nf].TreePos = CheckFiner;
                TGrid::BlockInfoAll[level-1][nf].myrank  = -1;
            }
        }

           #if 1 //update all blocks in process boundary
           #else //update only the blocks that changed
           #endif
    }

    double done = MPI_Wtime();
    TIMINGS [0] += done - started;
    
    };




    template<typename Processing>
    SynchronizerMPIType * sync(Processing& p)
    {

        auto started = MPI_Wtime();

        bool per [3] = {TGrid::xperiodic,TGrid::yperiodic,TGrid::zperiodic};

        //temporarily hardcoded Cstencil
        StencilInfo Cstencil = p.stencil;
        Cstencil.sx = -1; Cstencil.sy = -1; Cstencil.sz = -1;
        Cstencil.ex =  2; Cstencil.ey =  2; Cstencil.ez =  2;
        Cstencil.tensorial = true;


        auto blockperDim = TGrid::getMaxBlocks();     
        const StencilInfo stencil = p.stencil;
        assert(stencil.isvalid());

        SynchronizerMPIType * queryresult = NULL;
             

        typename std::map<StencilInfo, SynchronizerMPIType*>::iterator itSynchronizerMPI = SynchronizerMPIs.find(stencil);

        if (itSynchronizerMPI == SynchronizerMPIs.end())
        {
            queryresult = new SynchronizerMPIType(p.stencil, Cstencil, worldcomm, per, TGrid::getlevelMax(),
                                                Block::sizeX,
                                                Block::sizeY,
                                                Block::sizeZ,
                                                blockperDim[0],
                                                blockperDim[1],
                                                blockperDim[2],
                                                TGrid::getBlocksInfo(),TGrid::getBlockInfoAll());

            SynchronizerMPIs[stencil] = queryresult;
            queryresult->_Setup(TGrid::getBlocksInfo(),TGrid::getBlockInfoAll());
        }
        else
        {
           queryresult = itSynchronizerMPI->second;
        }  
    
        queryresult->sync(sizeof(typename Block::element_type)/sizeof(Real), sizeof(Real)>4 ? MPI_DOUBLE : MPI_FLOAT, timestamp) ;//, TGrid::getBlocksInfo(),TGrid::getBlockInfoAll());
        timestamp = (timestamp + 1) % 32768;

        auto done = MPI_Wtime();

        TIMINGS [1] += (done-started);

        return queryresult;
    }




    template<typename Processing>
    const SynchronizerMPIType& get_SynchronizerMPI(Processing& p) const
    {
        assert((SynchronizerMPIs.find(p.stencil) != SynchronizerMPIs.end()));

        return *SynchronizerMPIs.find(p.stencil)->second;
    }

    int getResidentBlocksPerDimension(int idim) const
    {
        //assert(false);
        assert(idim>=0 && idim<3);
        return 1;//mybpd[idim];
    }


    int rank() const override {return myrank;}


    virtual void FillPos() override
    { 

        auto started = MPI_Wtime();

        TGrid::m_blocks.clear();
        TGrid::m_vInfo.clear();
        for (int m=0; m<TGrid::levelMax; m++)
        {
            for (int n=0; n<TGrid::NX*TGrid::NY*TGrid::NZ*pow(pow(2,m),3); n++)
            {

                if (TGrid::BlockInfoAll[m][n].TreePos == Exists && TGrid::BlockInfoAll[m][n].myrank == myrank) 
                {
                    TGrid::m_vInfo.push_back(TGrid::BlockInfoAll[m][n]);
                    TGrid::m_blocks.push_back((Block*)TGrid::BlockInfoAll[m][n].ptrBlock);
                }
            }

        }  

        auto done = MPI_Wtime();

        TIMINGS [2] += done-started; 
    }







    int getBlocksPerDimension(int idim) const override
    {
        //assert(false);
        assert(idim>=0 && idim<3);
        return 1;//mybpd[idim]*pesize[idim];
    }

    void peindex(int _mypeindex[3]) const
    {
        //assert(false);
        for(int i=0; i<3; ++i)
            _mypeindex[i] = 0;// mypeindex[i];
    }

    size_t getTimeStamp() const
    {
        return timestamp;
    }

    MPI_Comm getCartComm() const
    {
        return cartcomm;
    }

    MPI_Comm getWorldComm() const
    {
        return worldcomm;
    }

    void getSubdomainLow(double low[3]) const {
        low[0] = subdomain_low[0];
        low[1] = subdomain_low[1];
        low[2] = subdomain_low[2];
    }

    void getSubdomainHigh(double high[3]) const {
        high[0] = subdomain_high[0];
        high[1] = subdomain_high[1];
        high[2] = subdomain_high[2];
    }
};

CUBISM_NAMESPACE_END
