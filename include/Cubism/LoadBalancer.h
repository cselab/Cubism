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
  
protected:
    TGrid * m_refGrid;
    int rank,size;

public:

    LoadBalancer(TGrid & grid) 
    {
        MPI_Comm_size(MPI_COMM_WORLD,&size);
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        m_refGrid = &grid;
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


        
        for ( auto & bb: I) 
        {       
            int nBlock = m_refGrid->getZforward(bb.level, 2*(bb.index[0]/2),2*(bb.index[1]/2),2*(bb.index[2]/2) ); 
            
            BlockInfo & b    = m_refGrid->getBlockInfoAll(bb.level,bb.Z);
            BlockInfo & base = m_refGrid->getBlockInfoAll(b.level,nBlock);

            if (b.Z != nBlock && base.state==Compress)
            {
                if (base.myrank != rank && b.myrank == rank)
                {
                    //std::cout << "Rank " << rank << " will send block " << b.level << " " << b.Z << " to rank " << base.myrank << "\n";
            
                    send_infos[base.myrank].push_back(&b);
                    send_buffer2[base.myrank].push_back(b.level);
                    send_buffer2[base.myrank].push_back(b.Z    );
                    b.myrank = base.myrank;
                }
            }
            else if (b.Z == nBlock && base.state==Compress)
            {
                for (int n = nBlock+1; n<nBlock+8; n++)
                {
                    BlockInfo & temp = m_refGrid->getBlockInfoAll(b.level,n);
                  
                    if (temp.myrank != rank)
                    {
                        //std::cout << "Rank " << rank << " will receive block " << temp.level << " " << temp.Z << " from rank " << temp.myrank << "\n";
            
                        recv_infos[temp.myrank].push_back(&temp);
                        recv_buffer2[temp.myrank].push_back(-1);
                        recv_buffer2[temp.myrank].push_back(-1);
                        temp.myrank = base.myrank;
                    }
                }

              
            }
        }

       

        


    #if 0


        for (int r=0; r<size; r++)
        {          
            for (int i=0; i<recv_infos[r].size(); i++)
            {
                BlockInfo & info = *recv_infos[r][i];
                m_refGrid->_alloc(info.level,info.Z);
            }
        }



        int BlockBytes = sizeof(BlockType)/sizeof(Real);
        std::vector <MPI_Datatype> send_datatype(size);
        std::vector <MPI_Datatype> recv_datatype(size);

        for (int r=0; r<size; r++)
        {
            std::vector<int> send_displacement(send_infos[r].size());
            for (int i=0; i<send_infos[r].size(); i++)
            {
                BlockInfo & info = *send_infos[r][i];
                

                BlockType * b1 =  (BlockType *)info.ptrBlock;
                BlockType * b2 =  (BlockType *)(*recv_infos[r][0]).ptrBlock;


                Real * a1 = & b1->data[0][0][0].alpha1rho1;
                Real * a2 = & b2->data[0][0][0].alpha1rho1;


                send_displacement[i] = a1-a2;//((Real*) info.ptrBlock - (Real*)(*send_infos[r][0]).ptrBlock);// *sizeof(Real);
            }
            MPI_Type_create_indexed_block(send_infos[r].size(), BlockBytes, &send_displacement[0], MPI_DOUBLE, &send_datatype[r]);
            MPI_Type_commit(&send_datatype[r]);
      

            std::vector<int> recv_displacement(recv_infos[r].size());
            for (int i=0; i<recv_infos[r].size(); i++)
            {
                BlockInfo & info = *recv_infos[r][i];

                assert(info.ptrBlock != NULL);

                BlockType * b1 =  (BlockType *)info.ptrBlock;
                BlockType * b2 =  (BlockType *)(*recv_infos[r][0]).ptrBlock;


                Real * a1 = & b1->data[0][0][0].alpha1rho1;
                Real * a2 = & b2->data[0][0][0].alpha1rho1;

                // recv_displacement[i] = ((Real*) info.ptrBlock - (Real*)(*recv_infos[r][0]).ptrBlock) ;
            
                recv_displacement[i] = a1-a2;
            

                std::cout << "recv_displacement =" << recv_displacement[i] << "   for rank " << rank <<"\n"; 
            }
            MPI_Type_create_indexed_block(recv_infos[r].size(), BlockBytes, &recv_displacement[0], MPI_DOUBLE, &recv_datatype[r]);
            MPI_Type_commit(&recv_datatype[r]);
        }


        for (int r=0; r<size; r++)
        {          
            if (r!=rank) // && (send_infos[r].size()!=0 || recv_infos[r].size()!=0))
            std::cout << "Rank " << rank << " will send/receive " << send_infos[r].size() << "/" << recv_infos[r].size() << " blocks to rank " << r << "\n";
        }   
        MPI_Barrier(MPI_COMM_WORLD);



        std::vector <MPI_Request> send_requests;
        std::vector <MPI_Request> recv_requests;


        std::vector <std::vector<double>> HugeBuffer(size);
        
        for (int r = 0 ; r < size; r ++ ) if (r!=rank)
        {
            if (recv_infos[r].size()!=0) 
            {
                HugeBuffer[r].resize(8*8*8*8*2*10);


                MPI_Request req;
                recv_requests.push_back(req);
                //MPI_Irecv(&recv_infos[r][0]->ptrBlock, 1, recv_datatype[r], r, 12345 , MPI_COMM_WORLD, &recv_requests[recv_requests.size()-1]);
                
                MPI_Irecv(&HugeBuffer[r][0], recv_infos[r].size()*8*8*8*8*2, MPI_DOUBLE, r, 12345 , MPI_COMM_WORLD, &recv_requests[recv_requests.size()-1]);
      
            }
            if (send_infos[r].size()!=0) 
            {
                MPI_Request req;
                send_requests.push_back(req);
                MPI_Isend(&send_infos[r][0]->ptrBlock, 1, send_datatype[r], r, 12345 , MPI_COMM_WORLD, &send_requests[send_requests.size()-1]);
            }    
        }

        if (recv_requests.size()!=0)
            MPI_Waitall(recv_requests.size(), &recv_requests[0], MPI_STATUSES_IGNORE);
        
        if (send_requests.size()!=0)
            MPI_Waitall(send_requests.size(), &send_requests[0], MPI_STATUSES_IGNORE);


    #else

        int BlockBytes = sizeof(BlockType);

        std::vector< std::vector<Real> > send_buffer(size);
        std::vector< std::vector<Real> > recv_buffer(size);
       


      
        for (int r=0; r<size; r++)
        {
            send_buffer[r].resize(send_infos[r].size()*BlockBytes/sizeof(Real));
            recv_buffer[r].resize(recv_infos[r].size()*BlockBytes/sizeof(Real));
            int d = 0;
            for (int i=0; i<send_infos[r].size(); i++)
            {
                BlockInfo & info = *send_infos[r][i];
                BlockType * b1 =  (BlockType *)info.ptrBlock;
                Real * a1 = & b1->data[0][0][0].alpha1rho1;
                std::memcpy(&send_buffer[r][d], a1,BlockBytes);
                d += BlockBytes/sizeof(Real);
            }
        }


        for (int r=0; r<size; r++)
        {          
        //    if (r!=rank) // && (send_infos[r].size()!=0 || recv_infos[r].size()!=0))
        //    std::cout << "Rank " << rank << " will send/receive " << send_infos[r].size() << "/" << recv_infos[r].size() << " blocks to rank " << r << "\n";
        //    std::cout << "Rank " << rank << " will send/receive " << send_buffer2[r].size() << "/" << recv_buffer2[r].size() << " blocks to rank " << r << "\n";
  
        }   









        std::vector <MPI_Request> send_requests;
        std::vector <MPI_Request> recv_requests;
        
        for (int r = 0 ; r < size; r ++ ) if (r!=rank)
        {
            if (recv_infos[r].size()!=0) 
            {
                MPI_Request req;
                recv_requests.push_back(req);            
                MPI_Irecv(&recv_buffer[r][0], recv_buffer[r].size(), MPI_DOUBLE, r, 12345 , MPI_COMM_WORLD, &recv_requests[recv_requests.size()-1]);
      
            }
            if (send_infos[r].size()!=0) 
            {
                MPI_Request req;
                send_requests.push_back(req);
                MPI_Isend(&send_buffer[r][0], send_buffer[r].size(), MPI_DOUBLE, r, 12345 , MPI_COMM_WORLD, &send_requests[send_requests.size()-1]);
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
                MPI_Irecv(&recv_buffer2[r][0], recv_buffer2[r].size(), MPI_INT, r, 1234567 , MPI_COMM_WORLD, &recv_requests2[recv_requests2.size()-1]);
      
            }
            if (send_infos[r].size()!=0) 
            {
                MPI_Request req;
                send_requests2.push_back(req);
                MPI_Isend(&send_buffer2[r][0], send_buffer2[r].size(), MPI_INT, r, 1234567 , MPI_COMM_WORLD, &send_requests2[send_requests2.size()-1]);
            }    
        }

        if (recv_requests2.size()!=0)
            MPI_Waitall(recv_requests2.size(), &recv_requests2[0], MPI_STATUSES_IGNORE);
        
        if (send_requests2.size()!=0)
            MPI_Waitall(send_requests2.size(), &send_requests2[0], MPI_STATUSES_IGNORE);


        #endif




        for (int r=0; r<size; r++)
        {          
            for (int i=0; i<send_infos[r].size(); i++)
            {
                BlockInfo & info = *send_infos[r][i];
                m_refGrid->_dealloc(info.level,info.Z);
            }
        }




//        //CRASHES HERE
//        //   |
//        //   |
//        //   |
//        //   V
//
//		for (int r=0; r<size; r++)
//		{
//			for (int i=0; i<send_buffer2[r].size(); i++)
//			{
//				std::cout << send_buffer2[r][i] << " | " ;
//			}
//			std::cout<<"\n";
//		}
//        
//
//
//
//		for (int r=0; r<size; r++)
//		{
//			for (int i=0; i<recv_buffer2[r].size(); i++)
//			{
//				std::cout << recv_buffer2[r][i] << " " ;
//			}
//			std::cout<<"\n";
//		}
        





        for (int r=0; r<size; r++)
        {    
            int d =0;
            for (int i=0; i<recv_infos[r].size(); i++)
            {
                //std::cout << recv_buffer2[r].size() << " " << 2*i << "\n"; 
                int level = recv_buffer2[r][2*i  ];
                int Z     = recv_buffer2[r][2*i+1];



                //std::cout << "Rank " << rank << " received block " << level << " " << Z <<" from rank " << r << "\n";



                m_refGrid->_alloc(level,Z);
                BlockInfo info = m_refGrid->getBlockInfoAll(level,Z);
                BlockType * b1 =  (BlockType *)info.ptrBlock;
                assert(b1!=NULL);
                Real * a1 = & b1->data[0][0][0].alpha1rho1;
                std::memcpy( a1 ,&recv_buffer[r][d],BlockBytes);
                d += BlockBytes/sizeof(Real);
            }
        }









        m_refGrid->FillPos();
        m_refGrid->UpdateBlockInfoAll_States();


       // MPI_Barrier(MPI_COMM_WORLD);
       // int err = 123456789;
       // MPI_Abort(MPI_COMM_WORLD,err);



    }










};


        

}//namespace AMR_CUBISM
