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

struct StencilInfoWrapper
{
    StencilInfo stencil;
    StencilInfoWrapper(int g=1)
    {
        stencil.sx = -g;
        stencil.sy = -g;
        stencil.sz = DIMENSION == 3 ? -g : 0;
        stencil.ex = +g+1;
        stencil.ey = +g+1;
        stencil.ez = DIMENSION == 3 ?  +g+1 : 1;
        stencil.tensorial = true;
    }
};

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
    const int nGhosts = 0;
    const int DIM = B::ElementType::DIM;
    StencilInfoWrapper p(nGhosts > 0 ? nGhosts : 1);
    for (int j = 0 ; j < DIM ; j++)
        p.stencil.selcomponents.push_back(j);
    cubism::SynchronizerMPI_AMR<Real,TGrid>& Synch = *grid.sync(p);
    LabMPI lab;
    lab.prepare(grid, Synch);

    long long mycells = 0;
    for (size_t groupID = 0 ; groupID < MyGroups.size() ; groupID ++)
    {
        const BlockGroup & group = MyGroups[groupID];
        const int nXX = group.NXX;
        const int nYY = group.NYY;
        const int nZZ = group.NZZ;
        int dd = (nZZ - 1 + 2*nGhosts )*(nYY - 1 + 2*nGhosts)*(nXX - 1 + 2*nGhosts);//*NCHANNELS;
        mycells += dd;
    }
    hsize_t base_tmp[1] = {0};
    MPI_Exscan(&mycells, &base_tmp[0], 1, MPI_LONG_LONG,  MPI_SUM , comm);

    long long start = 0;
    // Write grid meta-data
    {
        std::ostringstream myfilename;
        myfilename << filename.str();
        //myfilename <<  std::setfill('0') << std::setw(10) << rank;
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
            s << "   <Topology TopologyType=\"3DCoRectMesh\" Dimensions=\" " << nZZ + 2*nGhosts << " " << nYY + 2*nGhosts << " " << nXX + 2*nGhosts << "\"/>\n";
            s << "   <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n";
            s << "   <DataItem Dimensions=\"3\" NumberType=\"Double\" Precision=\"8\" " "Format=\"XML\">\n";
            s << "    " << std::scientific << group.origin[2] - nGhosts*group.h << " " << group.origin[1] - nGhosts*group.h << " " << group.origin[0] - nGhosts*group.h << "\n";
            s << "   </DataItem>\n";
            s << "   <DataItem Dimensions=\"3\" NumberType=\"Double\" Precision=\"8\" " "Format=\"XML\">\n";
            #if DIMENSION == 3
              s << "    " << std::scientific <<group.h<<" "<<group.h <<" "<< group.h << "\n";
            #else
              s << "    " << std::scientific <<hmin<<" "<<group.h <<" "<< group.h << "\n";
            #endif
            s << "   </DataItem>\n";
            s << "   </Geometry>\n";
  
            int dd = (nZZ - 1 + 2*nGhosts )*(nYY - 1 + 2*nGhosts)*(nXX - 1 + 2*nGhosts);//*NCHANNELS;
            //Ghosts not saved as reading them by Paraview is very slow
            ////////////////////////////////////
            //s << "   <Attribute Name=\"vtkGhostType\" AttributeType=\"" << "unsigned char" << "\" Center=\"Cell\">\n";
            //s << "<DataItem ItemType=\"HyperSlab\" Dimensions=\" " << 1 << " " << 1 << " " << dd <<  "\" Type=\"HyperSlab\"> \n";
            //s << "<DataItem Dimensions=\"3 1\" Format=\"XML\">\n";
            //s << base_tmp[0] + start <<"\n";
            //s << 1     <<"\n";
            //s << dd    <<"\n";
            //s << "</DataItem>\n";
            //s << "   <DataItem ItemType=\"Uniform\"  Dimensions=\" " << dd << " " << "\" NumberType=\"UChar\"  Format=\"HDF\">\n";
            ////s << "    " << (myfilename.str() + ".h5").c_str() << ":/" << "dset_ghost" << "\n";
            //s << "    " << (myfilename.str() + ".h5").c_str() << ":/" << "dset" << "\n";
            //s << "   </DataItem>\n";
            //s << "   </DataItem>\n";
            //s << "   </Attribute>\n";
            //////////////////////////////////
            ////////////////////////////////////
            //s << "   <Attribute Name=\"data\" AttributeType=\"" << TStreamer::getAttributeName() << "\" Center=\"Cell\">\n";
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
            //////////////////////////////////
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
    
    //fullpath <<  std::setfill('0') << std::setw(10) << rank; //mike
    //Dump data
    hid_t file_id, dataset_id, fspace_id, fapl_id, mspace_id;
    //hid_t dataset_id_ghost, fspace_id_ghost;
    
    H5open();

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
    //fspace_id_ghost  = H5Screate_simple(1, dims, NULL);
    dataset_id       = H5Dcreate (file_id, "dset"      , get_hdf5_type<hdf5Real>(), fspace_id      , H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    //dataset_id_ghost = H5Dcreate (file_id, "dset_ghost", H5T_NATIVE_UCHAR         , fspace_id_ghost, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    //4.Dump
    long long start1 = 0;
    std::vector<hdf5Real> bigArray(start);
    //std::vector<unsigned char> bigArray_ghost(start);
    std::vector<cubism::BlockInfo*> avail0 = Synch.avail_inner();
    std::vector<cubism::BlockInfo*> avail1 = Synch.avail_halo();
    for (size_t groupID = 0 ; groupID < MyGroups.size() ; groupID ++)
    {
        const BlockGroup & group = MyGroups[groupID];
        const int nX_max = group.NXX-1;
        const int nY_max = group.NYY-1;
        const int nZ_max = group.NZZ-1;
        int dd1 = (nX_max+2*nGhosts) * (nY_max+2*nGhosts) * (nZ_max+2*nGhosts);// * NCHANNELS;
        //int dd1 = 2 * (nX_max+2*nGhosts) * (nY_max+2*nGhosts) * (nZ_max+2*nGhosts);// * NCHANNELS;
        std::vector<hdf5Real> array_block( dd1, 0.0);
        //std::vector<unsigned char> array_block_ghost( dd1, 0);
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
            lab.load(I, 0);
            for (int iz = 0 - nGhosts; iz < nZ + nGhosts; iz++)
            for (int iy = 0 - nGhosts; iy < nY + nGhosts; iy++)
            for (int ix = 0 - nGhosts; ix < nX + nGhosts; ix++)
            {
                hdf5Real output[NCHANNELS];
                TStreamer::operate(lab,ix,iy,iz,output);
                const int iz_b = (kB-group.i_min[2])*nZ + iz + nGhosts;
                const int iy_b = (jB-group.i_min[1])*nY + iy + nGhosts;
                const int ix_b = (iB-group.i_min[0])*nX + ix + nGhosts;
                const int base = iz_b *((nX_max+2*nGhosts)*(nY_max+2*nGhosts)) + iy_b *((nX_max+2*nGhosts)) + ix_b;               
                //for (int j = 0; j < NCHANNELS; ++j) array_block[NCHANNELS*base+j] = output[j];
                if (NCHANNELS > 1)
                {
                  output[0] = output[0]*output[0] + output[1]*output[1] + output[2]*output[2];
                  array_block[base] = sqrt(output[0]);
                }
                else
                {
                  array_block[base] = output[0];
                }
                //if (iz_b <  nGhosts || iy_b <  nGhosts || ix_b <  nGhosts
                //    || iz_b >= nZ_max + nGhosts
                //    || iy_b >= nY_max + nGhosts
                //    || ix_b >= nX_max + nGhosts)
                //    //array_block_ghost[base] = 1;
                //    array_block[base] = 1;
            }
        }
        for (int j = 0 ; j < dd1 ; j ++)
        {
            bigArray      [start1 + j] = array_block[j];
            //bigArray_ghost[start1 + j] = array_block_ghost[j];
        }
        start1 += dd1;
    }
    hsize_t count[1] = {bigArray.size()};

    fspace_id       = H5Dget_space(dataset_id);
    //fspace_id_ghost = H5Dget_space(dataset_id_ghost);
    H5Sselect_hyperslab(fspace_id      , H5S_SELECT_SET, base_tmp, NULL, count, NULL);
    //H5Sselect_hyperslab(fspace_id_ghost, H5S_SELECT_SET, base_tmp, NULL, count, NULL);
    mspace_id = H5Screate_simple(1, count, NULL);
    H5Dwrite(dataset_id, get_hdf5_type<hdf5Real>(), mspace_id, fspace_id      , fapl_id, bigArray.data());
    //H5Dwrite(dataset_id_ghost, H5T_NATIVE_UCHAR   , mspace_id, fspace_id_ghost, fapl_id, bigArray_ghost.data());

    H5Sclose(mspace_id);
    H5Sclose(fspace_id);
    H5Dclose(dataset_id);
    //H5Sclose(fspace_id_ghost);
    //H5Dclose(dataset_id_ghost);
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
