#pragma once

#include "BlockInfo.h"
#include "GridMPI.h"
#include <omp.h>
#include <cstring>
#include <string>
#include <algorithm>

namespace cubism
{

template<typename TGrid>
class LoadBalancer
{

public:
    typedef typename TGrid::Block BlockType;   
    typedef typename TGrid::Block::ElementType ElementType;
    bool movedBlocks;
  
protected:
    TGrid * m_refGrid;
    int rank,size;
    MPI_Datatype MPI_BLOCK;
    struct MPI_Block
    {
        int mn [2]; 
        Real data [sizeof(BlockType) / sizeof(Real)];        
        MPI_Block(BlockInfo & info, bool Fillptr = true)
        {
            mn [0] = info.level;
            mn [1] = info.Z;
            if (Fillptr)
            {
                Real * aux = &((BlockType *)info.ptrBlock)->data[0][0][0].alpha1rho1;
                std::memcpy(&data[0],aux,sizeof(BlockType));
            }
        }
        MPI_Block(){}
    };

public:
    LoadBalancer(TGrid & grid) 
    {
        MPI_Comm_size(MPI_COMM_WORLD,&size);
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        m_refGrid = &grid;
        movedBlocks = false;
        MPI_Block dummy;
        int array_of_blocklengths[2] =  {2,sizeof(BlockType)/sizeof(Real)};
        MPI_Aint array_of_displacements[2] = {0,2*sizeof(int)};
        MPI_Datatype array_of_types[2] = {MPI_INT,MPI_DOUBLE};
        MPI_Type_create_struct(2,array_of_blocklengths,array_of_displacements,array_of_types,&MPI_BLOCK);        
        MPI_Type_commit(&MPI_BLOCK);
    }

    ~LoadBalancer()
    {
        MPI_Type_free(&MPI_BLOCK);
    }

    void PrepareCompression()
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) std::cout << "PrepareCompression checkpoint 1\n";

        std::vector <BlockInfo> & I = m_refGrid->getBlocksInfo();
        std::vector < std::vector < MPI_Block > > send_blocks(size);
        std::vector < std::vector < MPI_Block > > recv_blocks(size); 

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) std::cout << "PrepareCompression checkpoint 1.5\n";


        for ( auto & b: I) 
        {  
            assert(b.level >=0);
            assert(b.index[0] >=0);
            assert(b.index[1] >=0);
            assert(b.index[2] >=0);


            int nBlock = m_refGrid->getZforward(b.level, 2*(b.index[0]/2),2*(b.index[1]/2),2*(b.index[2]/2) );
         
            assert(nBlock >=0);
            assert(b.Z >=0);


            BlockInfo & base  = m_refGrid->getBlockInfoAll(b.level,nBlock);          
            BlockInfo & bCopy = m_refGrid->getBlockInfoAll(b.level,b.Z);          
  
            assert (b.TreePos == Exists);

            if (b.Z != nBlock && base.state==Compress)
            {
                if (base.myrank != rank && b.myrank == rank)
                {
                    if (base.myrank < 0) std::cout << "Base.myrank = " << base.myrank << "\n";
                    assert(base.TreePos == Exists);
                    assert(base.myrank >=0);
                    #if 0
                    base.state = Leave;
                    bCopy.state= Leave;
                    b.state    = Leave;
                    #else
                    send_blocks[base.myrank].push_back({bCopy});            
                    b.myrank     = base.myrank;
                    bCopy.myrank = base.myrank;
                    #endif
                }
            }
            else if (b.Z == nBlock && base.state==Compress)
            {
                for (int k=0; k<2; k++ )
                for (int j=0; j<2; j++ )
                for (int i=0; i<2; i++ )
                {
                  int n   = m_refGrid->getZforward(b.level,b.index[0]+i,b.index[1]+j,b.index[2]+k); 
                  
                  if (n == nBlock) continue;
                    BlockInfo & temp = m_refGrid->getBlockInfoAll(b.level,n);
                  
                    if (temp.myrank != rank)
                    {
                        assert(base.TreePos == Exists);
                        assert(base.myrank >=0);
                        assert(temp.myrank >=0);
                        #if 0
                        base.state = Leave;
                        bCopy.state= Leave;
                        b.state    = Leave;
                        #else
                        recv_blocks[temp.myrank].push_back({temp,false});
                        temp.myrank = base.myrank;
                        #endif
                    }
                }          
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) std::cout << "PrepareCompression checkpoint 2\n";


        #if 1
        MPI_Barrier(MPI_COMM_WORLD);
        for (int r = 0 ; r < size; r ++ ) if (r!=rank)
        {
            for (auto & b : recv_blocks[r])
                std::cout << "Rank " << rank << " receives (" 
                          << b.mn[0] << "," << b.mn[1] <<") from " << r << "\n";

            for (auto & b : send_blocks[r])
                std::cout << "Rank " << rank << " sends (" 
                          << b.mn[0] << "," << b.mn[1] <<") to " << r << "\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
      
        #endif



        int BlockBytes = sizeof(BlockType);

        std::vector <MPI_Request> requests;
        
        for (int r = 0 ; r < size; r ++ ) if (r!=rank)
        {
            if (recv_blocks[r].size()!=0) 
            {
                MPI_Request req;
                requests.push_back(req);            
                MPI_Irecv(&recv_blocks[r][0], recv_blocks[r].size(), MPI_BLOCK , r, 123450 , MPI_COMM_WORLD, &requests.back() );
            }
            if (send_blocks[r].size()!=0) 
            {
                MPI_Request req;
                requests.push_back(req);
                MPI_Isend(&send_blocks[r][0], send_blocks[r].size(), MPI_BLOCK, r, 123450 , MPI_COMM_WORLD, &requests.back());
                
            }    
        }
      
        if (requests.size()!=0)
        {
            int err = 123;
            //MPI_Abort(MPI_COMM_WORLD,err);
            std::cout << "Rank " << rank <<" is preparing compression...\n";
            movedBlocks = true;
            MPI_Waitall(requests.size(), &requests[0], MPI_STATUSES_IGNORE);        
        }

        for (int r=0; r<size; r++)
            for (int i=0; i<(int)send_blocks[r].size(); i++)
            {
                m_refGrid->_dealloc(send_blocks[r][i].mn[0],send_blocks[r][i].mn[1]);
            }

        for (int r=0; r<size; r++)
        {    
            for (int i=0; i<(int)recv_blocks[r].size(); i++)
            {
                int level = recv_blocks[r][i].mn[0];
                int Z     = recv_blocks[r][i].mn[1];

                m_refGrid->_alloc(level,Z);
    
                BlockInfo info = m_refGrid->getBlockInfoAll(level,Z);
                BlockType * b1 =  (BlockType *)info.ptrBlock;
                assert(b1!=NULL);
                Real * a1 = & b1->data[0][0][0].alpha1rho1;
                std::memcpy( a1 ,recv_blocks[r][i].data,BlockBytes);

                info.changed = true;
            }
        }
    }

    void Balance_Diffusion()
    { 
        int right  = (rank == size-1) ?  MPI_PROC_NULL : rank + 1;
        int left   = (rank == 0     ) ?  MPI_PROC_NULL : rank - 1;  
        int my_blocks = m_refGrid->getBlocksInfo().size();
        int right_blocks,left_blocks;

        std::vector<MPI_Request> reqs(4);
        MPI_Irecv(& left_blocks, 1, MPI_INT,  left, 123, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(&right_blocks, 1, MPI_INT, right, 456, MPI_COMM_WORLD, &reqs[1]);
        MPI_Isend(&my_blocks   , 1, MPI_INT,  left, 456, MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend(&my_blocks   , 1, MPI_INT, right, 123, MPI_COMM_WORLD, &reqs[3]);
        MPI_Waitall(4, &reqs[0], MPI_STATUSES_IGNORE);
   

        int nu = 2;
        int flux_left  = ((my_blocks -  left_blocks) / nu); 
        int flux_right = ((my_blocks - right_blocks) / nu); 

        if (rank == size-1) flux_right = 0;
        if (rank == 0     ) flux_left  = 0;


        std::vector <BlockInfo> SortedInfos = m_refGrid->getBlocksInfo();

        if (flux_right!=0  || flux_left !=0 ) 
            std::sort(SortedInfos.begin(),SortedInfos.end());

        std::vector <MPI_Block> send_left; 
        std::vector <MPI_Block> recv_left; 
        std::vector <MPI_Block> send_right; 
        std::vector <MPI_Block> recv_right; 


        int BlockBytes = sizeof(BlockType);
        std::vector<MPI_Request> request;

        if (flux_left > 0) //then I will send blocks to my left rank
        {
            for (int i=0; i< flux_left; i++)
                send_left.push_back({SortedInfos[i]});
            MPI_Request req;
            request.push_back(req);
            MPI_Isend(&send_left[0], send_left.size(), MPI_BLOCK, left, 7890 , MPI_COMM_WORLD, &request.back());

        }
        else if (flux_left < 0) //then I will receive blocks from my left rank
        {
            recv_left.resize( abs(flux_left)  );            
            MPI_Request req;
            request.push_back(req);
            MPI_Irecv(&recv_left[0], recv_left.size(), MPI_BLOCK, left, 4560 , MPI_COMM_WORLD, &request.back());
        }   


        if (flux_right > 0) //then I will send blocks to my right rank
        {
            for (int i=0; i< flux_right; i++)
                send_right.push_back({SortedInfos[my_blocks-i-1]});
            MPI_Request req;
            request.push_back(req);
            MPI_Isend(&send_right[0],send_right.size(), MPI_BLOCK, right, 4560 , MPI_COMM_WORLD, &request.back());
        }
        else if (flux_right < 0) //then I will receive blocks from my right rank
        {
            recv_right.resize( abs(flux_right) );
            MPI_Request req;
            request.push_back(req);
            MPI_Irecv(&recv_right[0], recv_right.size(), MPI_BLOCK, right, 7890 , MPI_COMM_WORLD,&request.back());
        }   

        if (request.size() != 0)
        {
            movedBlocks = true;      
            MPI_Waitall(request.size(), &request[0], MPI_STATUSES_IGNORE);
        }

        for (int i=0; i<flux_right; i++)
        {
            BlockInfo & info = SortedInfos[my_blocks-i-1];
            m_refGrid->_dealloc(info.level,info.Z);
            BlockInfo & info1 = m_refGrid->getBlockInfoAll(info.level,info.Z);
            info1.myrank = right;           
        }

        for (int i=0; i<flux_left; i++)
        {
            BlockInfo & info = SortedInfos[i];
            m_refGrid->_dealloc(info.level,info.Z);
            BlockInfo & info1 = m_refGrid->getBlockInfoAll(info.level,info.Z);
            info1.myrank = left;    
        }

        for (int i=0; i<-flux_left; i++)
        {
            int level =  recv_left[i].mn[0];
            int Z     =  recv_left[i].mn[1];
            m_refGrid->_alloc(level,Z);
            BlockInfo & info = m_refGrid->getBlockInfoAll(level,Z);
            info.TreePos = Exists;
            BlockType * b1 =  (BlockType *)info.ptrBlock;
            Real * a1 = & b1->data[0][0][0].alpha1rho1;
            std::memcpy( a1 ,recv_left[i].data,BlockBytes);            
            info.myrank  = rank;
            info.changed = true;


            assert (m_refGrid->getBlockInfoAll(level,Z).myrank == rank );

            #if 0
            int p[3] = {info.index[0], info.index[1], info.index[2]};    
            if (level<m_refGrid->getlevelMax() -1)
                for (int k=0; k<2; k++ )
                for (int j=0; j<2; j++ )
                for (int ii=0; ii<2; ii++ )
                {      
                    int nc = m_refGrid->getZforward(level+1,2*p[0]+ii,2*p[1]+j,2*p[2]+k);
                    
                    BlockInfo & iC = m_refGrid->getBlockInfoAll(level+1,nc);
                    iC.TreePos = CheckCoarser;
                    iC.myrank  = -1;
                }
             if (level>0)
             {
                int nf = m_refGrid->getZforward(level-1,p[0]/2,p[1]/2,p[2]/2);
                BlockInfo & iF = m_refGrid->getBlockInfoAll(level-1,nf);
                iF.TreePos = CheckFiner;
                iF.myrank  = -1;
             }
             #endif


        }
      
        for (int i=0; i<-flux_right; i++)
        {
            int level =  recv_right[i].mn[0];
            int Z     =  recv_right[i].mn[1];
            m_refGrid->_alloc(level,Z);
            BlockInfo & info = m_refGrid->getBlockInfoAll(level,Z);
            info.TreePos = Exists;
            BlockType * b1 =  (BlockType *)info.ptrBlock;
            assert(b1!=NULL);
            Real * a1 = & b1->data[0][0][0].alpha1rho1;
            std::memcpy( a1 ,recv_right[i].data,BlockBytes);
            info.myrank  = rank;
            info.changed = true;


            assert (m_refGrid->getBlockInfoAll(level,Z).myrank == rank );
            #if 0
            int p[3] = {info.index[0], info.index[1], info.index[2]};    
            if (level<m_refGrid->getlevelMax() -1)
                for (int k=0; k<2; k++ )
                for (int j=0; j<2; j++ )
                for (int ii=0; ii<2; ii++ )
                {      
                    int nc = m_refGrid->getZforward(level+1,2*p[0]+ii,2*p[1]+j,2*p[2]+k);
                    
                    BlockInfo & iC = m_refGrid->getBlockInfoAll(level+1,nc);
                    iC.TreePos = CheckCoarser;
                    iC.myrank  = -1;
                }
             if (level>0)
             {
                int nf = m_refGrid->getZforward(level-1,p[0]/2,p[1]/2,p[2]/2);
                BlockInfo & iF = m_refGrid->getBlockInfoAll(level-1,nf);
                iF.TreePos = CheckFiner;
                iF.myrank  = -1;
             }
             #endif
        }
   

        //MPI_Allreduce(MPI_IN_PLACE,&movedBlocks,1,MPI_LOGICAL, MPI_LAND, MPI_COMM_WORLD);
        int temp = movedBlocks ? 1:0;
        
        MPI_Allreduce(MPI_IN_PLACE,&temp,1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        movedBlocks = (temp == 1);

        {
            int b = m_refGrid->getBlocksInfo().size();
            std::vector<int> all_b(size);
            MPI_Gather(&b, 1, MPI_INT, &all_b[0], 1, MPI_INT, 0, MPI_COMM_WORLD);
            
            if (rank==0)
            {
                std::cout << "&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~\n";
                std::cout << " Distribution of blocks among ranks: \n";       
                for (int r=0; r<size; r++)
                    std::cout << all_b[r] << " | ";
                std::cout << "\n";
                std::cout << "&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~&~\n";
            }
        }


        for (auto & info: m_refGrid->getBlocksInfo())
        {
            if (info.TreePos != Exists) abort();
            info.TreePos = Exists;
            info.myrank = rank;
            assert(info.TreePos == Exists);
        }
























    }
};

}//namespace AMR_CUBISM
