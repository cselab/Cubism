/*
 *  HDF5Dumper_MPI.h
 *  Cubism
 *
 *  Created by Babak Hejazialhosseini on 5/24/09.
 *  Copyright 2009 CSE Lab, ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <cassert>
#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <mpi.h>

#include "HDF5Dumper.h"


CUBISM_NAMESPACE_BEGIN

// The following requirements for the data TStreamer are required:
// TStreamer::NCHANNELS        : Number of data elements (1=Scalar, 3=Vector, 9=Tensor)
// TStreamer::operate          : Data access methods for read and write
// TStreamer::getAttributeName : Attribute name of the date ("Scalar", "Vector", "Tensor")
template<typename TStreamer, typename hdf5Real, typename TGrid>
void DumpHDF5_MPI(const TGrid &grid,
                  const int iCounter,
                  const typename TGrid::Real absTime,
                  const std::string &fname,
                  const std::string &dpath = ".",
                  const bool bXMF = true)
{
    typedef typename TGrid::BlockType B;

    // fname is the base filepath tail without file type extension and additional identifiers
    std::ostringstream filename;
    std::ostringstream fullpath;
    filename << fname;
    fullpath << dpath << "/" << filename.str();

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank,size;
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);

    std::vector<B *      > MyBlocks = grid.GetBlocks();
    std::vector<BlockInfo> MyInfos  = grid.getBlocksInfo();
    const int Ngrids = MyBlocks.size();
    const int nX  = B::sizeX;
    const int nY  = B::sizeY;
    const int nZ  = B::sizeZ;
    const int NCHANNELS = TStreamer::NCHANNELS;
 
    std::cout<<" ---> Rank " << rank << " is dumping " << Ngrids << " Blocks.\n";

    //herr_t status;
    hid_t file_id, dataset_id, fspace_id; //fapl_id 
    hsize_t dims[4] = {nZ, nY, nX, NCHANNELS}; 
    
    H5open();

    //1.Set up file access property list with parallel I/O access
    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL);

    //2.Create a new file collectively and release property list identifier.
    file_id = H5Fcreate((fullpath.str()+".h5").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    H5Pclose(plist_id);
    
    //3.All ranks need to create datasets dset* that correspond to each block
    int TotalGrids;
    MPI_Allreduce(&Ngrids, &TotalGrids, 1, MPI_INT, MPI_SUM,comm);
    for (int m=0; m<TotalGrids; m++)
    {            
        std::stringstream name_ss;
        name_ss <<"dset" << std::setfill('0') << std::setw(10) << m ;
        std::string name = name_ss.str();

        fspace_id = H5Screate_simple(4, dims, NULL);
        dataset_id = H5Dcreate2(file_id, name.c_str(), H5T_NATIVE_DOUBLE, fspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dclose(dataset_id);
        H5Sclose(fspace_id);       
    }
    

    //4.Each rank now dumps its own blocks to the corresponding dset 
    int grid_base = 0;
    MPI_Exscan(&Ngrids, &grid_base, 1, MPI_INT,MPI_SUM,comm);

    hdf5Real * array_block = new hdf5Real[nX * nY * nZ * NCHANNELS];
    for (int m=0; m<Ngrids; m++)
    {       
        B & block = *MyBlocks[m];
        for(int iz=0; iz<nZ; iz++)
        for(int iy=0; iy<nY; iy++)
        for(int ix=0; ix<nX; ix++)
        {
            hdf5Real output[NCHANNELS];
            for(int j=0; j<NCHANNELS; ++j)
                output[j] = 0;
        
            TStreamer::operate(block, ix, iy, iz, (hdf5Real*)output);
                
            int base = NCHANNELS*(ix + nX * (iy + nY * iz));
            for(int j=0; j<NCHANNELS; ++j)
                array_block[ base + j ] = output[j];
        }

        int mm = m + grid_base;
        std::stringstream name_ss;
        name_ss <<"dset" << std::setfill('0') << std::setw(10) << mm ;
        std::string name = name_ss.str();

        //Access existing dataset and write to it
        dataset_id = H5Dopen2(file_id, name.c_str(), H5P_DEFAULT);
        H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, array_block);
        H5Dclose(dataset_id);
    }

    //5.Close hdf5 file
    H5Fclose(file_id);


    H5close();




    //6.Write grid meta-data
    if (bXMF)
    {
        MPI_File xmf;
        
        //delete the xmf file is it exists; no worries if it doesn't
        MPI_File_delete((fullpath.str()+".xmf").c_str(), MPI_INFO_NULL);
       

        MPI_File_open(comm,(fullpath.str()+".xmf").c_str(),MPI_MODE_WRONLY|MPI_MODE_CREATE,MPI_INFO_NULL,&xmf);

        std::stringstream s_head;

        s_head << "<?xml version=\"1.0\" ?>\n";
        s_head << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
        s_head << "<Xdmf Version=\"2.0\">\n";
        s_head << " <Domain>\n";
        s_head << "   <Grid Name=\"OctTree\" GridType=\"Collection\">\n";
        s_head << "     <Time Value=\""<<std::setprecision(10)<<std::setw(10)<<absTime<<"\"/>\n\n";

        
        MPI_Offset offset = 0;
        if (rank == 0)
            MPI_File_write_at(xmf, offset, (s_head.str()).c_str()  , (s_head.str()).length(), MPI_CHAR, MPI_STATUS_IGNORE);

        
        const int width = 15;
        for (int m = 0; m < Ngrids; m++)
        {
            int mm = m + grid_base;
            BlockInfo I = MyInfos[m];
        
        
            std::stringstream s;
            s << "   <Grid GridType=\"Uniform\">\n";
            s << "     <Topology TopologyType=\"3DCoRectMesh\" Dimensions=\" " <<nX + 1 << " " <<nY + 1 << " " <<nZ + 1 << "\"/>\n\n";
            s << "       <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n";
            s << "          <DataItem Dimensions=\"3\" NumberType=\"Double\" Precision=\"8\" Format=\"XML\">\n";
            /*ViSit*/ 
            //s << "                 "<<std::setprecision(10) <<std::setw(width) << I.origin[0] << " " <<std::setw(width) << I.origin[1] << " " <<std::setw(width) << I.origin[2] << "\n";
            /*Paraview*/
            s << "                 "<<std::setprecision(10)<<std::setw(width) << I.origin[2] << " " << std::setw(width)<<I.origin[1] << " " << std::setw(width)<<I.origin[0] << "\n";
            s << "          </DataItem>\n";      
            s << "          <DataItem Dimensions=\"3\" NumberType=\"Double\" Precision=\"8\" Format=\"XML\">\n";
            s << "                 "<<std::setprecision(10) <<std::setw(width)<< I.h << " " << std::setw(width)<<I.h << " " << std::setw(width)<<I.h << "\n";
            s << "          </DataItem>\n";      
            s << "     </Geometry>\n\n";
            s << "     <Attribute Name=\"data\" AttributeType=\" "<< TStreamer::getAttributeName() << "\" Center=\"Cell\">\n";
            s << "       <DataItem Dimensions=\" " <<nX << " " << nY << " " << nZ << " " << std::setw(10)<<NCHANNELS <<"\" NumberType=\"Float\" Precision=\" " <<   (int)sizeof(hdf5Real)  << "\" Format=\"HDF\">\n";
            

            std::stringstream name_ss;
            name_ss <<"dset" << std::setfill('0') << std::setw(10) << mm ;
            std::string tmp = name_ss.str();

            s << "        "<<(filename.str()+".h5").c_str()<<":/"<<tmp<<"\n";        
            s << "       </DataItem>\n";
            s << "     </Attribute>\n";
            s << "   </Grid>\n";
        
            std::string st = s.str();
            
            if (m==0)
            {
                offset = grid_base * (s.str()).length() + (s_head.str()).length()  ; 
                MPI_File_set_view(xmf,offset,MPI_CHAR,MPI_CHAR,"native", MPI_INFO_NULL);
            }
            
            MPI_File_write(xmf, st.c_str()  , st.length(), MPI_CHAR, MPI_STATUS_IGNORE);            
        }
        std::stringstream s_tail;
        s_tail <<  "   </Grid>\n";
        s_tail <<  " </Domain>\n";
        s_tail <<  "</Xdmf>\n";
            
        if (rank == size -1)
            MPI_File_write(xmf, (s_tail.str()).c_str()  , (s_tail.str()).length() , MPI_CHAR, MPI_STATUS_IGNORE);
       
        MPI_File_close(&xmf);
    }


    //    MPI_Barrier(MPI_COMM_WORLD);
    //    if (rank == 0) std::cout << "===================================> Aborting manually after calling HDF5Dumper_MPI!\n";
    //    int err = 0 ;
    //    MPI_Abort(MPI_COMM_WORLD,err);
}

template<typename TStreamer, typename hdf5Real, typename TGrid>
void ReadHDF5_MPI(TGrid &grid, const std::string& fname, const std::string& dpath=".")
{
    std::cout<<"mike: ReadHDF5_MPI skipped! \n"; return;

    #if 0 //mike
#ifdef CUBISM_USE_HDF
    typedef typename TGrid::BlockType B;

    int rank;

    // fname is the base filepath tail without file type extension and
    // additional identifiers
    std::ostringstream filename;
    std::ostringstream fullpath;
    filename << fname;
    fullpath << dpath << "/" << filename.str();

    herr_t status;
    hid_t file_id, dataset_id, fspace_id, fapl_id, mspace_id;

    MPI_Comm comm = grid.getCartComm();
    MPI_Comm_rank(comm, &rank);

    int coords[3];
    grid.peindex(coords);

    const unsigned int NX = static_cast<unsigned int>(grid.getResidentBlocksPerDimension(0))*B::sizeX;
    const unsigned int NY = static_cast<unsigned int>(grid.getResidentBlocksPerDimension(1))*B::sizeY;
    const unsigned int NZ = static_cast<unsigned int>(grid.getResidentBlocksPerDimension(2))*B::sizeZ;
    const unsigned int NCHANNELS = TStreamer::NCHANNELS;

    hdf5Real * array_all = new hdf5Real[NX * NY * NZ * NCHANNELS];

    std::vector<BlockInfo> vInfo_local = grid.getResidentBlocksInfo();

    hsize_t count[4] = {NZ, NY, NX, NCHANNELS};
    hsize_t offset[4] = {
        static_cast<unsigned int>(coords[2]) * NZ,
        static_cast<unsigned int>(coords[1]) * NY,
        static_cast<unsigned int>(coords[0]) * NX,
        0
    };

    H5open();
    fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    status = H5Pset_fapl_mpio(fapl_id, comm, MPI_INFO_NULL); if(status<0) H5Eprint1(stdout);
    file_id = H5Fopen((fullpath.str()+".h5").c_str(), H5F_ACC_RDONLY, fapl_id);
    status = H5Pclose(fapl_id); if(status<0) H5Eprint1(stdout);

    dataset_id = H5Dopen2(file_id, "data", H5P_DEFAULT);
    fapl_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(fapl_id, H5FD_MPIO_COLLECTIVE);

    fspace_id = H5Dget_space(dataset_id);
    H5Sselect_hyperslab(fspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);

    mspace_id = H5Screate_simple(4, count, NULL);
    status = H5Dread(dataset_id, get_hdf5_type<hdf5Real>(), mspace_id, fspace_id, fapl_id, array_all);
    if (status < 0) H5Eprint1(stdout);

#pragma omp parallel for
    for(size_t i=0; i<vInfo_local.size(); i++)
    {
        BlockInfo& info = vInfo_local[i];
        const int idx[3] = {info.index[0], info.index[1], info.index[2]};
        B & b = *(B*)info.ptrBlock;

        for(int iz=0; iz<static_cast<int>(B::sizeZ); iz++)
            for(int iy=0; iy<static_cast<int>(B::sizeY); iy++)
                for(int ix=0; ix<static_cast<int>(B::sizeX); ix++)
                {
                    const int gx = idx[0]*B::sizeX + ix;
                    const int gy = idx[1]*B::sizeY + iy;
                    const int gz = idx[2]*B::sizeZ + iz;

                    hdf5Real * const ptr_input = array_all + NCHANNELS*(gx + NX * (gy + NY * gz));
                    TStreamer::operate(b, ptr_input, ix, iy, iz);
                }
    }

    status = H5Pclose(fapl_id); if(status<0) H5Eprint1(stdout);
    status = H5Dclose(dataset_id); if(status<0) H5Eprint1(stdout);
    status = H5Sclose(fspace_id); if(status<0) H5Eprint1(stdout);
    status = H5Sclose(mspace_id); if(status<0) H5Eprint1(stdout);
    status = H5Fclose(file_id); if(status<0) H5Eprint1(stdout);

    H5close();

    delete [] array_all;
#else
    _warn_no_hdf5();
#endif
#endif
}

CUBISM_NAMESPACE_END