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

public:

    LoadBalancer(TGrid & grid) 
    {
        MPI_Comm_size(MPI_COMM_WORLD,&size);
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        m_refGrid = &grid;
        movedBlocks = false;
    }

    ~LoadBalancer()
    {
    }


    void PrepareCompression()
    {
        std::vector <BlockInfo> & I = m_refGrid->getBlocksInfo();


        std::vector < std::vector < BlockInfo * > > send_infos(size);
        std::vector < std::vector < BlockInfo * > > recv_infos(size); 


        std::vector< std::vector<int> > send_buffer2(size);
        std::vector< std::vector<int> > recv_buffer2(size);


        
        for ( auto & b: I) 
        {                  	

            int nBlock = m_refGrid->getZforward(b.level, 2*(b.index[0]/2),2*(b.index[1]/2),2*(b.index[2]/2) );
         
            BlockInfo & base  = m_refGrid->getBlockInfoAll(b.level,nBlock);          
			BlockInfo & bCopy = m_refGrid->getBlockInfoAll(b.level,b.Z);          
  
            if (b.Z != nBlock && base.state==Compress)
            {
                if (base.myrank != rank && b.myrank == rank)
                {
            
                    send_infos[base.myrank].push_back(&bCopy);
                    send_buffer2[base.myrank].push_back(b.level);
                    send_buffer2[base.myrank].push_back(b.Z    );
                    b.myrank     = base.myrank;
                    bCopy.myrank = base.myrank;
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
            
                        recv_infos[temp.myrank].push_back(&temp);
                        recv_buffer2[temp.myrank].push_back(-1);
                        recv_buffer2[temp.myrank].push_back(-1);
                        temp.myrank = base.myrank;
                    }
                }          
            }






        }
  
        int BlockBytes = sizeof(BlockType);

        std::vector< std::vector<Real> > send_buffer(size);
        std::vector< std::vector<Real> > recv_buffer(size);
             
        for (int r=0; r<size; r++)
        {
            send_buffer[r].resize(send_infos[r].size()*BlockBytes/sizeof(Real));
            recv_buffer[r].resize(recv_infos[r].size()*BlockBytes/sizeof(Real));
            int d = 0;
            for (int i=0; i<(int)send_infos[r].size(); i++)
            {
                BlockInfo & info = *send_infos[r][i];
                BlockType * b1 =  (BlockType *)info.ptrBlock;
                Real * a1 = & b1->data[0][0][0].alpha1rho1;
                std::memcpy(&send_buffer[r][d], a1,BlockBytes);
                d += BlockBytes/sizeof(Real);
            }
        }





        std::vector <MPI_Request> send_requests;
        std::vector <MPI_Request> recv_requests;
        
        for (int r = 0 ; r < size; r ++ ) if (r!=rank)
        {
            if (recv_infos[r].size()!=0) 
            {
                MPI_Request req;
                recv_requests.push_back(req);            
                MPI_Irecv(&recv_buffer[r][0], recv_buffer[r].size(), MPI_DOUBLE, r, 12345 , MPI_COMM_WORLD, &recv_requests.back() );//&recv_requests[(int)recv_requests.size()-1]);
                movedBlocks = true;
            }
            if (send_infos[r].size()!=0) 
            {
                MPI_Request req;
                send_requests.push_back(req);
                MPI_Isend(&send_buffer[r][0], send_buffer[r].size(), MPI_DOUBLE, r, 12345 , MPI_COMM_WORLD, &send_requests.back());//&send_requests[(int)send_requests.size()-1]);
                movedBlocks = true;
            }    
        }
      
        if (recv_requests.size()!=0)
            MPI_Waitall(recv_requests.size(), &recv_requests[0], MPI_STATUSES_IGNORE);
        
        if (send_requests.size()!=0)
            MPI_Waitall(send_requests.size(), &send_requests[0], MPI_STATUSES_IGNORE);


        std::vector <MPI_Request> send_requests2;
        std::vector <MPI_Request> recv_requests2;
        
        for (int r = 0 ; r < size; r ++ ) if (r!=rank)
        {
            if (recv_infos[r].size()!=0) 
            {
                MPI_Request req;
                recv_requests2.push_back(req);            
                MPI_Irecv(&recv_buffer2[r][0], recv_buffer2[r].size(), MPI_INT, r, 1234567 , MPI_COMM_WORLD, &recv_requests2.back());//&recv_requests2[recv_requests2.size()-1]);
                movedBlocks = true;
            }
            if (send_infos[r].size()!=0) 
            {
                MPI_Request req;
                send_requests2.push_back(req);
                MPI_Isend(&send_buffer2[r][0], send_buffer2[r].size(), MPI_INT, r, 1234567 , MPI_COMM_WORLD, &send_requests2.back());// &send_requests2[send_requests2.size()-1]);
                movedBlocks = true;
            }    
        }

        if (recv_requests2.size()!=0)
            MPI_Waitall(recv_requests2.size(), &recv_requests2[0], MPI_STATUSES_IGNORE);
        
        if (send_requests2.size()!=0)
            MPI_Waitall(send_requests2.size(), &send_requests2[0], MPI_STATUSES_IGNORE);




        for (int r=0; r<size; r++)
        {          
            for (int i=0; i<(int)send_infos[r].size(); i++)
            {
                BlockInfo & info = *send_infos[r][i];
                m_refGrid->_dealloc(info.level,info.Z);
            }
        }





        for (int r=0; r<size; r++)
        {    
            int d =0;
            for (int i=0; i<(int)recv_infos[r].size(); i++)
            {
                int level = recv_buffer2[r][2*i  ];
                int Z     = recv_buffer2[r][2*i+1];

                m_refGrid->_alloc(level,Z);
    
                BlockInfo info = m_refGrid->getBlockInfoAll(level,Z);
                BlockType * b1 =  (BlockType *)info.ptrBlock;
                assert(b1!=NULL);
                Real * a1 = & b1->data[0][0][0].alpha1rho1;
                std::memcpy( a1 ,&recv_buffer[r][d],BlockBytes);
                d += BlockBytes/sizeof(Real);
            }
        }

        //m_refGrid->FillPos(true);
        //m_refGrid->UpdateBlockInfoAll_States();
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

   		int flux_left  = (my_blocks -  left_blocks) / nu; //divide by two following stability criterion for diffusion equation
   		int flux_right = (my_blocks - right_blocks) / nu; //divide by two following stability criterion for diffusion equation


        if (rank == size-1) flux_right = 0;
        if (rank == 0     ) flux_left  = 0;


   		std::vector <BlockInfo> SortedInfos = m_refGrid->getBlocksInfo();

   		std::sort(SortedInfos.begin(),SortedInfos.end());



		std::vector <Real> send_buffer_left; 
  		std::vector <Real> recv_buffer_left; 

		std::vector <Real> send_buffer_right; 
  		std::vector <Real> recv_buffer_right; 


        std::vector<int> send_buffer_left2;
        std::vector<int> recv_buffer_left2;
	    std::vector<int> send_buffer_right2;
        std::vector<int> recv_buffer_right2;

 
   		int BlockBytes = sizeof(BlockType);
   		std::vector<MPI_Request> request;
		if (flux_left > 0) //then I will send blocks to my left rank
		{
            movedBlocks = true;
       
			send_buffer_left.resize(flux_left * BlockBytes / sizeof(Real) );
	    
        	int d = 0;
            for (int i=0; i< flux_left; i++)
            {
                BlockInfo & info = SortedInfos[i];
                BlockType * b1 =  (BlockType *)info.ptrBlock;
                Real * a1 = & b1->data[0][0][0].alpha1rho1;
                std::memcpy(&send_buffer_left[d], a1,BlockBytes);
                d += BlockBytes/sizeof(Real);
            
                send_buffer_left2.push_back(info.level);
                send_buffer_left2.push_back(info.Z    );
            }

            MPI_Request req;
            request.push_back(req);
            MPI_Isend(&send_buffer_left[0], send_buffer_left.size(), MPI_DOUBLE, left, 123 , MPI_COMM_WORLD, &request.back());

            MPI_Request req2;
            request.push_back(req2);
            MPI_Isend(&send_buffer_left2[0],send_buffer_left2.size(), MPI_INT, left, 789 , MPI_COMM_WORLD, &request.back());
        }
		else if (flux_left < 0) //then I will receive blocks from my left rank
		{

            movedBlocks = true;
       

			recv_buffer_left.resize( abs(flux_left) * BlockBytes / sizeof(Real) );

			recv_buffer_left2.resize( abs(flux_left) * 2 );

            
            MPI_Request req;
            request.push_back(req);
      		MPI_Irecv(&recv_buffer_left[0], recv_buffer_left.size(), MPI_DOUBLE, left, 456 , MPI_COMM_WORLD, &request.back());//&request[request.size()-1]);

    	    MPI_Request req2;
            request.push_back(req2);
      		MPI_Irecv(&recv_buffer_left2[0], recv_buffer_left2.size(), MPI_INT, left, 101112 , MPI_COMM_WORLD, &request.back());//&request[request.size()-1]);
		}	


		if (flux_right > 0) //then I will send blocks to my right rank
		{
            movedBlocks = true;
       

			send_buffer_right.resize(flux_right * BlockBytes / sizeof(Real) );
	    
        	int d = 0;
            for (int i=0; i< flux_right; i++)
            {
                BlockInfo & info = SortedInfos[my_blocks-i-1];
                BlockType * b1 =  (BlockType *)info.ptrBlock;
                Real * a1 = & b1->data[0][0][0].alpha1rho1;
                std::memcpy(&send_buffer_right[d], a1,BlockBytes);
                d += BlockBytes/sizeof(Real);

                send_buffer_right2.push_back(info.level);
                send_buffer_right2.push_back(info.Z    );
            }

            MPI_Request req;
            request.push_back(req);
            MPI_Isend(&send_buffer_right[0], send_buffer_right.size(), MPI_DOUBLE, right, 456 , MPI_COMM_WORLD, &request.back());//&request[request.size()-1]);

            MPI_Request req2;
            request.push_back(req2);
            MPI_Isend(&send_buffer_right2[0],send_buffer_right2.size(), MPI_INT, right, 101112 , MPI_COMM_WORLD, &request.back());//&request[request.size()-1]);
        }
		else if (flux_right < 0) //then I will receive blocks from my right rank
		{
            movedBlocks = true;      

			recv_buffer_right.resize( abs(flux_right) * BlockBytes / sizeof(Real) );

			recv_buffer_right2.resize( abs(flux_right) * 2 );
            
            MPI_Request req;
            request.push_back(req);
      		MPI_Irecv(&recv_buffer_right[0], recv_buffer_right.size(), MPI_DOUBLE, right, 123 , MPI_COMM_WORLD,&request.back());// &request[request.size()-1]);
	
	   	    MPI_Request req2;
            request.push_back(req2);
      		MPI_Irecv(&recv_buffer_right2[0], recv_buffer_right2.size(), MPI_INT, right, 789 , MPI_COMM_WORLD, &request.back());//&request[request.size()-1]);
		}	


		if (request.size() != 0)
			MPI_Waitall(request.size(), &request[0], MPI_STATUSES_IGNORE);


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


        int d =0;
        for (int i=0; i<-flux_left; i++)
        {
            int level = recv_buffer_left2[2*i  ];
            int Z     = recv_buffer_left2[2*i+1];
            m_refGrid->_alloc(level,Z);
            BlockInfo & info = m_refGrid->getBlockInfoAll(level,Z);
            info.TreePos = Exists;
            BlockType * b1 =  (BlockType *)info.ptrBlock;
            assert(b1!=NULL);
            Real * a1 = & b1->data[0][0][0].alpha1rho1;
            std::memcpy( a1 ,&recv_buffer_left[d],BlockBytes);
            d += BlockBytes/sizeof(Real);
            info.myrank  = rank;

            assert(info.TreePos == Exists);
        }
      

        d =0;
        for (int i=0; i<-flux_right; i++)
        {
            int level = recv_buffer_right2[2*i  ];
            int Z     = recv_buffer_right2[2*i+1];
            m_refGrid->_alloc(level,Z);
            BlockInfo & info = m_refGrid->getBlockInfoAll(level,Z);
            info.TreePos = Exists;
            BlockType * b1 =  (BlockType *)info.ptrBlock;
            assert(b1!=NULL);
            Real * a1 = & b1->data[0][0][0].alpha1rho1;
            std::memcpy( a1 ,&recv_buffer_right[d],BlockBytes);
            d += BlockBytes/sizeof(Real);
            info.myrank  = rank;

            assert(info.TreePos == Exists);
        }
   


        //m_refGrid->FillPos();
        //m_refGrid->UpdateBlockInfoAll_States();

        MPI_Allreduce(MPI_IN_PLACE,&movedBlocks,1,MPI_LOGICAL, MPI_LAND, MPI_COMM_WORLD);



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

   

       




    }
};


        

}//namespace AMR_CUBISM
