/*
 *  HDF5Dumper_MPI.h
 *  Cubism
 *
 *  Created by Michalis Chatzimanolakis 
 *  Copyright 2020 CSE Lab, ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <cassert>
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <string>
#include <vector>
#include <fstream>

#include "HDF5Dumper.h"
#include "GridMPI.h"
#include "StencilInfo.h"

CUBISM_NAMESPACE_BEGIN

// The following requirements for the data TStreamer are required:
// TStreamer::NCHANNELS        : Number of data elements (1=Scalar, 3=Vector, 9=Tensor)
// TStreamer::operate          : Data access methods for read and write
// TStreamer::getAttributeName : Attribute name of the date ("Scalar", "Vector", "Tensor")

template <typename TStreamer, typename hdf5Real, typename TGrid, typename LabMPI> 
void DumpHDF5_MPI(TGrid &grid, typename TGrid::Real absTime, const std::string &fname, const std::string &dpath = ".", const bool bXMF = true)
{
    typedef typename TGrid::BlockType B;
    const int nX = B::sizeX;
    const int nY = B::sizeY;
    const int nZ = B::sizeZ;

    MPI_Comm comm = grid.getWorldComm();
    const int rank = grid.myrank;
    const int size = grid.world_size;
    const int NCHANNELS = TStreamer::NCHANNELS;
    std::ostringstream filename;
    std::ostringstream fullpath;
    filename << fname;// fname is the base filepath without file type extension
    fullpath << dpath << "/" << filename.str();

    std::vector<BlockGroup> & MyGroups = grid.MyGroups;
    grid.UpdateMyGroups();

    #if DIMENSION==2
        double hmin = 1e10;
        for (size_t groupID = 0 ; groupID < MyGroups.size() ; groupID ++) hmin = std::min(hmin,MyGroups[groupID].h);
        MPI_Allreduce(MPI_IN_PLACE, &hmin, 1, MPI_DOUBLE, MPI_MIN, comm);
    #endif

    long long mycells = 0;
    for (size_t groupID = 0 ; groupID < MyGroups.size() ; groupID ++)
    {
        mycells += (MyGroups[groupID].NZZ - 1)*(MyGroups[groupID].NYY - 1)*(MyGroups[groupID].NXX - 1);
    }
    hsize_t base_tmp[1] = {0};
    MPI_Exscan(&mycells, &base_tmp[0], 1, MPI_LONG_LONG,  MPI_SUM , comm);

    long long start = 0;
    // Write grid meta-data
    {
        std::ostringstream myfilename;
        myfilename << filename.str();
        std::stringstream s;
        if (rank == 0)
        {
            s << "<?xml version=\"1.0\" ?>\n";
            s << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
            s << "<Xdmf Version=\"2.0\">\n";
            s << "<Domain>\n";
            s << " <Grid Name=\"OctTree\" GridType=\"Collection\">\n";
            s << "  <Time Value=\"" << std::scientific << absTime << "\"/>\n\n";
        }
        for (size_t groupID = 0 ; groupID < MyGroups.size() ; groupID ++)
        {
            const BlockGroup & group = MyGroups[groupID];
            const int nXX = group.NXX;
            const int nYY = group.NYY;
            const int nZZ = group.NZZ;
            s << "  <Grid GridType=\"Uniform\">\n";
            s << "   <Topology TopologyType=\"3DCoRectMesh\" Dimensions=\" " << nZZ << " " << nYY << " " << nXX << "\"/>\n";
            s << "   <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n";
            s << "   <DataItem Dimensions=\"3\" NumberType=\"Double\" Precision=\"8\" " "Format=\"XML\">\n";
            s << "    " << std::scientific << group.origin[2]<< " " << group.origin[1]<< " " << group.origin[0]<< "\n";
            s << "   </DataItem>\n";
            s << "   <DataItem Dimensions=\"3\" NumberType=\"Double\" Precision=\"8\" " "Format=\"XML\">\n";
            #if DIMENSION == 3
              s << "    " << std::scientific <<group.h<<" "<<group.h <<" "<< group.h << "\n";
            #else
              s << "    " << std::scientific <<hmin<<" "<<group.h <<" "<< group.h << "\n";
            #endif
            s << "   </DataItem>\n";
            s << "   </Geometry>\n";
  
            int dd = (nZZ - 1)*(nYY - 1)*(nXX - 1);//*NCHANNELS;
            s << "   <Attribute Name=\"data\" AttributeType=\"" << "Scalar"<< "\" Center=\"Cell\">\n";
            s << "<DataItem ItemType=\"HyperSlab\" Dimensions=\" " << 1 << " " << 1 << " " << dd <<  "\" Type=\"HyperSlab\"> \n";
            s << "<DataItem Dimensions=\"3 1\" Format=\"XML\">\n";
            s << base_tmp[0] + start<<"\n";
            s << 1     <<"\n";
            s << dd    <<"\n";
            s << "</DataItem>\n";
            s << "   <DataItem ItemType=\"Uniform\"  Dimensions=\" " << dd << " " << "\" NumberType=\"Float\" Precision=\" " << (int)sizeof(hdf5Real) << "\" Format=\"HDF\">\n";
            s << "    " << (myfilename.str() + ".h5").c_str() << ":/" << "dset" << "\n";
            s << "   </DataItem>\n";
            s << "   </DataItem>\n";
            s << "   </Attribute>\n";
            start += dd;
            s << "  </Grid>\n\n";
        }
        if (rank == size - 1)
        {
            s << " </Grid>\n";
            s << "</Domain>\n";
            s << "</Xdmf>\n";
        }
        std::string st    = s.str();
        MPI_Offset offset = 0;
        MPI_Offset len    = st.size() * sizeof(char);
        MPI_File xmf;
        MPI_File_delete((fullpath.str() + ".xmf").c_str(), MPI_INFO_NULL); // delete the xmf file is it exists
        MPI_File_open(comm, (fullpath.str() + ".xmf").c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &xmf);
        MPI_Exscan(&len, &offset, 1, MPI_OFFSET, MPI_SUM, comm);
        MPI_File_write_at_all(xmf, offset, st.data(), st.size(), MPI_CHAR, MPI_STATUS_IGNORE);
        MPI_File_close(&xmf);
    }

    H5open();
    // Write group data to separate hdf5 file
    {
        hid_t file_id,fapl_id;
        hid_t dataset_origins, fspace_origins, mspace_origins;// origin[0],origin[1],origin[2],group.h : doubles
        hid_t dataset_indices, fspace_indices, mspace_indices;// nx,ny,nz,index[0],index[1],index[2],level : integers


        //1.Set up file access property list with parallel I/O access
        fapl_id = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_fapl_mpio(fapl_id, comm, MPI_INFO_NULL);

        //2.Create a new file collectively and release property list identifier.
        file_id = H5Fcreate((fullpath.str()+"-groups.h5").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);        
        H5Pclose(fapl_id);

        //3.Create datasets
        fapl_id = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(fapl_id, H5FD_MPIO_COLLECTIVE);

        long long total = MyGroups.size();//total number of groups
        MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_LONG_LONG, MPI_SUM, comm);

        hsize_t dim_origins = 4*total;
        hsize_t dim_indices = 7*total;
        fspace_origins = H5Screate_simple(1, & dim_origins, NULL);
        fspace_indices = H5Screate_simple(1, & dim_indices, NULL);
        dataset_origins = H5Dcreate (file_id, "origins", H5T_NATIVE_DOUBLE, fspace_origins, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        dataset_indices = H5Dcreate (file_id, "indices", H5T_NATIVE_INT   , fspace_indices, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        std::vector<double> origins(4*MyGroups.size());
        std::vector<int   > indices(7*MyGroups.size());
        for (size_t groupID = 0 ; groupID < MyGroups.size() ; groupID ++)
        {
            const BlockGroup & group = MyGroups[groupID];
            origins[4*groupID  ] = group.origin[0];
            origins[4*groupID+1] = group.origin[1];
            origins[4*groupID+2] = group.origin[2];
            origins[4*groupID+3] = group.h;
            indices[7*groupID  ] = group.NXX-1;
            indices[7*groupID+1] = group.NYY-1;
            indices[7*groupID+2] = group.NZZ-1;
            indices[7*groupID+3] = group.i_min[0];
            indices[7*groupID+4] = group.i_min[1];
            indices[7*groupID+5] = group.i_min[2];
            indices[7*groupID+6] = group.level;
        }

        long long my_size = MyGroups.size();//total number of groups
        hsize_t offset_groups = 0;
        MPI_Exscan(&my_size, &offset_groups, 1, MPI_LONG_LONG,  MPI_SUM , comm);
        hsize_t offset_origins = 4 * offset_groups;
        hsize_t offset_indices = 7 * offset_groups;

        hsize_t count_origins = origins.size();
        hsize_t count_indices = indices.size();
        fspace_origins = H5Dget_space(dataset_origins);
        fspace_indices = H5Dget_space(dataset_indices);
        H5Sselect_hyperslab(fspace_origins, H5S_SELECT_SET, &offset_origins, NULL, &count_origins, NULL);
        H5Sselect_hyperslab(fspace_indices, H5S_SELECT_SET, &offset_indices, NULL, &count_indices, NULL);
        mspace_origins = H5Screate_simple(1, &count_origins, NULL);
        mspace_indices = H5Screate_simple(1, &count_indices, NULL);
        H5Dwrite(dataset_origins, H5T_NATIVE_DOUBLE, mspace_origins, fspace_origins, fapl_id, origins.data());
        H5Dwrite(dataset_indices, H5T_NATIVE_INT   , mspace_indices, fspace_indices, fapl_id, indices.data());

        H5Sclose(mspace_origins);
        H5Sclose(mspace_indices);
        H5Sclose(fspace_origins);
        H5Sclose(fspace_indices);
        H5Dclose(dataset_origins);
        H5Dclose(dataset_indices);
        H5Pclose(fapl_id);
        H5Fclose(file_id);
    }
    
    //fullpath <<  std::setfill('0') << std::setw(10) << rank; //mike
    //Dump data
    hid_t file_id, dataset_id, fspace_id, fapl_id, mspace_id;
    //hid_t dataset_id_ghost, fspace_id_ghost;
    
    //1.Set up file access property list with parallel I/O access
    fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl_id, comm, MPI_INFO_NULL);

    //2.Create a new file collectively and release property list identifier.
    file_id = H5Fcreate((fullpath.str()+".h5").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);        
    H5Pclose(fapl_id);

    //3.Create dataset
    fapl_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(fapl_id, H5FD_MPIO_COLLECTIVE);
    long long total;
    MPI_Allreduce(&start, &total, 1, MPI_LONG_LONG, MPI_SUM, comm);
    //total = start;
    hsize_t dims[1]  = {(hsize_t) total};
    fspace_id        = H5Screate_simple(1, dims, NULL);
    dataset_id       = H5Dcreate (file_id, "dset", get_hdf5_type<hdf5Real>(), fspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    //4.Dump
    long long start1 = 0;
    std::vector<hdf5Real> bigArray(start);
    for (size_t groupID = 0 ; groupID < MyGroups.size() ; groupID ++)
    {
        const BlockGroup & group = MyGroups[groupID];
        const int nX_max = group.NXX-1;
        const int nY_max = group.NYY-1;
        const int nZ_max = group.NZZ-1;
        int dd1 = nX_max * nY_max * nZ_max;// * NCHANNELS;
        std::vector<hdf5Real> array_block( dd1, 0.0);
        for (int kB = group.i_min[2]; kB <= group.i_max[2]; kB++)
        for (int jB = group.i_min[1]; jB <= group.i_max[1]; jB++)
        for (int iB = group.i_min[0]; iB <= group.i_max[0]; iB++)
        {
            #if DIMENSION == 3
              const long long Z = BlockInfo::forward(group.level,iB,jB,kB);
            #else
              const long long Z = BlockInfo::forward(group.level,iB,jB);
            #endif
            const cubism::BlockInfo& I = grid.getBlockInfoAll(group.level,Z);
            const auto & lab = * (B*) (I.ptrBlock);
            for (int iz = 0; iz < nZ; iz++)
            for (int iy = 0; iy < nY; iy++)
            for (int ix = 0; ix < nX; ix++)
            {
                hdf5Real output[NCHANNELS];
                TStreamer::operate(lab,ix,iy,iz,output);
                const int iz_b = (kB-group.i_min[2])*nZ + iz;
                const int iy_b = (jB-group.i_min[1])*nY + iy;
                const int ix_b = (iB-group.i_min[0])*nX + ix;
                const int base = iz_b*nX_max*nY_max + iy_b*nX_max + ix_b;
                if (NCHANNELS > 1)
                {
                  output[0] = output[0]*output[0] + output[1]*output[1] + output[2]*output[2];
                  array_block[base] = sqrt(output[0]);
                }
                else
                {
                  array_block[base] = output[0];
                }
            }
        }
        for (int j = 0 ; j < dd1 ; j ++)
        {
            bigArray[start1 + j] = array_block[j];
        }
        start1 += dd1;
    }
    hsize_t count[1] = {bigArray.size()};

    fspace_id = H5Dget_space(dataset_id);
    mspace_id = H5Screate_simple(1, count, NULL);

    H5Sselect_hyperslab(fspace_id, H5S_SELECT_SET, base_tmp, NULL, count, NULL);
    H5Dwrite(dataset_id, get_hdf5_type<hdf5Real>(), mspace_id, fspace_id, fapl_id, bigArray.data());
    H5Sclose(mspace_id);
    H5Sclose(fspace_id);
    H5Dclose(dataset_id);
    H5Pclose(fapl_id);
    H5Fclose(file_id);
    H5close();
}

template <typename TStreamer, typename hdf5Real, typename TGrid>
void DumpHDF5_MPI(TGrid &grid, const int iCounter, typename TGrid::Real absTime, const std::string &fname, const std::string &dpath = ".", const bool bXMF = true)
{
  DumpHDF5_MPI<TStreamer,hdf5Real,TGrid>(grid,absTime,fname,dpath,bXMF);
}

template <typename TStreamer, typename hdf5Real, typename TGrid>
void ReadHDF5_MPI(TGrid &grid, const std::string &fname, const std::string &dpath = ".")
{
   std::cout << "ReadHDF5_MPI is not implemented!" << std::endl;
   return;
}
CUBISM_NAMESPACE_END