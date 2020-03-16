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


//    if (rank == 0) std::cout << "Dumper skipped.\n";
//    return;


    std::vector<B *      > MyBlocks = grid.GetBlocks();
    std::vector<BlockInfo> MyInfos  = grid.getBlocksInfo();
    const int Ngrids = MyBlocks.size();
    const int nX  = B::sizeX;
    const int nY  = B::sizeY;
    const int nZ  = B::sizeZ;
    const int NCHANNELS = TStreamer::NCHANNELS;
 
    std::cout<<" ---> Rank " << rank << " is dumping " << Ngrids << " Blocks.\n";


////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

    hid_t file_id, dataset_id, fspace_id, plist_id; 
    hsize_t dims[4] = {nZ, nY, nX, NCHANNELS}; 

    if (0 == rank)
    {
        H5open();
       
        plist_id = H5Pcreate(H5P_FILE_ACCESS);
        file_id = H5Fcreate((fullpath.str()+".h5").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
        H5Pclose(plist_id);
        H5Fclose(file_id);
       
        H5close();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    
    H5open();

    //1.Set up file access property list with parallel I/O access
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL);

    //2.Create a new file collectively and release property list identifier.
    //file_id = H5Fcreate((fullpath.str()+".h5").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    file_id = H5Fopen((fullpath.str()+".h5").c_str(), H5F_ACC_RDWR, plist_id);
    H5Pclose(plist_id);
    
    //3.All ranks need to create datasets dset* that correspond to each block
    int TotalGrids;
    MPI_Allreduce(&Ngrids, &TotalGrids, 1, MPI_INT, MPI_SUM,comm);
    
    for (int m=0; m< TotalGrids; m++)
    {            
        std::stringstream name_ss;
        name_ss <<"dset" << std::setfill('0') << std::setw(10) << m ;
        std::string name = name_ss.str();

        fspace_id = H5Screate_simple(4, dims, NULL);
        dataset_id = H5Dcreate(file_id, name.c_str(), H5T_NATIVE_DOUBLE, fspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
        H5Dclose(dataset_id);
        H5Sclose(fspace_id);       
    }

    hid_t plist_id1 = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id1, H5FD_MPIO_INDEPENDENT);

    
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
        dataset_id = H5Dopen(file_id, name.c_str(), H5P_DEFAULT);
        H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, plist_id1, array_block);
        //H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, array_block);
        H5Dclose(dataset_id);

    }
     H5Pclose(plist_id1);
   
    //5.Close hdf5 file
    H5Fclose(file_id);
    H5close();

    delete [] array_block;
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////    


    //6.Write grid meta-data
    if (bXMF)
    {
        std::stringstream s;

        if (rank == 0)
        {
            s << "<?xml version=\"1.0\" ?>\n";
            s << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
            s << "<Xdmf Version=\"2.0\">\n";
            s << "<Domain>\n";
            s << " <Grid Name=\"OctTree\" GridType=\"Collection\">\n";
            s << "  <Time Value=\""<<std::scientific<<absTime<<"\"/>\n\n";           
        }
      
        for (int m = 0; m < Ngrids; m++)
        {
            int mm = m + grid_base;
            BlockInfo I = MyInfos[m];     
        
            s << "  <Grid GridType=\"Uniform\">\n";
            s << "   <Topology TopologyType=\"3DCoRectMesh\" Dimensions=\" " <<nX + 1 << " " <<nY + 1 << " " <<nZ + 1 << "\"/>\n";
            s << "   <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n";
            s << "   <DataItem Dimensions=\"3\" NumberType=\"Double\" Precision=\"8\" Format=\"XML\">\n";
            
            /*ViSit*/ 
            //s << "    "<<std::scientific << I.origin[0] << " " << I.origin[1] << " " << I.origin[2] << "\n";
            /*Paraview*/
            s << "    "<<std::scientific << I.origin[2] << " " << I.origin[1] << " " << I.origin[0] << "\n";

            s << "   </DataItem>\n";      
            s << "   <DataItem Dimensions=\"3\" NumberType=\"Double\" Precision=\"8\" Format=\"XML\">\n";
            s << "    "<<std::scientific<< I.h << " " << I.h << " " <<I.h << "\n";
            s << "   </DataItem>\n";      
            s << "   </Geometry>\n";
            s << "   <Attribute Name=\"data\" AttributeType=\" "<< TStreamer::getAttributeName() << "\" Center=\"Cell\">\n";
            s << "   <DataItem Dimensions=\" " <<nX << " " << nY << " " << nZ << " " <<NCHANNELS <<"\" NumberType=\"Float\" Precision=\" " <<   (int)sizeof(hdf5Real)  << "\" Format=\"HDF\">\n";
            
            std::stringstream name_ss;
            name_ss <<"dset" << std::setfill('0') << std::setw(10) << mm ;
            std::string tmp = name_ss.str();

            s << "    "<<(filename.str()+".h5").c_str()<<":/"<<tmp<<"\n";        
            s << "   </DataItem>\n";
            s << "   </Attribute>\n";
            s << "  </Grid>\n\n";
        
        }           
        if (rank == size - 1)
        {
            s<<  " </Grid>\n";
            s<<  "</Domain>\n";
            s<<  "</Xdmf>\n";
        }   

        std::string st = s.str();
        MPI_Offset offset = 0;
        MPI_Offset len = st.size()*sizeof(char);

        MPI_File xmf;
         
        //delete the xmf file is it exists; no worries if it doesn't
        MPI_File_delete((fullpath.str()+".xmf").c_str(), MPI_INFO_NULL);     
        MPI_File_open(comm,(fullpath.str()+".xmf").c_str(),MPI_MODE_WRONLY|MPI_MODE_CREATE,MPI_INFO_NULL,&xmf);

        MPI_Exscan(&len,&offset,1,MPI_OFFSET,MPI_SUM,MPI_COMM_WORLD);

        MPI_File_write_at_all(xmf,offset,st.data(),st.size(),MPI_CHAR,MPI_STATUS_IGNORE);
      
        MPI_File_close(&xmf);
    } 
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
