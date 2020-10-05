//
//  HDF5Dumper.h
//  Cubism
//
//  Created by Babak Hejazialhosseini on 1/23/12.
//  Copyright 2011 ETH Zurich. All rights reserved.
//
#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <utility>


#include <cassert>
#include <cstdio>
#include <fstream>
#include <iomanip>      // std::setfill, std::setw



#ifdef _USE_HDF_
#warning  _USE_HDF_ is deprecated, use CUBISM_USE_HDF instead.
#define CUBISM_USE_HDF
#endif

#ifdef CUBISM_USE_HDF
#include <hdf5.h>

// Function to retrieve HDF5 type (hid_t) for a given real type.
// If using custom types, the user should specialize this function.
template <typename T> hid_t get_hdf5_type();
template <> inline hid_t get_hdf5_type<float>() { return H5T_NATIVE_FLOAT; }
template <> inline hid_t get_hdf5_type<double>() { return H5T_NATIVE_DOUBLE; }
#endif

#include "BlockInfo.h"
#include "MeshMap.h"

CUBISM_NAMESPACE_BEGIN

inline void _warn_no_hdf5(void) {
    fprintf(stderr, "USE OF HDF WAS DISABLED AT COMPILE TIME\n");
}


// The following requirements for the data TStreamer are required:
// TStreamer::NCHANNELS        : Number of data elements (1=Scalar, 3=Vector, 9=Tensor)
// TStreamer::operate          : Data access methods for read and write
// TStreamer::getAttributeName : Attribute name of the date ("Scalar", "Vector", "Tensor")
template<typename TStreamer, typename hdf5Real, typename TGrid>
void DumpHDF5(const TGrid &grid,
              //const int iCounter,
              const typename TGrid::Real absTime,
              const std::string &fname,
              const std::string &dpath = ".",
              const bool bXMF = true)
{
#if 1
    #ifdef CUBISM_USE_HDF

   int rank = 0;
   int size = 1;


   typedef typename TGrid::BlockType B;
   static const unsigned int nX = B::sizeX;
   static const unsigned int nY = B::sizeY;
   static const unsigned int nZ = B::sizeZ;

   // fname is the base filepath without file type extension
   std::ostringstream filename;
   std::ostringstream fullpath;
   filename << fname;
   fullpath << dpath << "/" << filename.str();

   std::vector<B *> MyBlocks      = grid.GetBlocks();
   std::vector<BlockInfo> MyInfos = grid.getBlocksInfo();

   const unsigned int Ngrids    = MyBlocks.size();
   const unsigned int NCHANNELS = TStreamer::NCHANNELS;

   std::cout << " ---> Rank " << rank << " is dumping " << Ngrids << " Blocks.\n";

   hid_t file_id, dataset_id, fspace_id, plist_id;

   H5open();

   // 1.Set up file access property list with parallel I/O access
   plist_id = H5Pcreate(H5P_FILE_ACCESS);
   //H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL);

   // 2.Create a new file collectively and release property list identifier.
   file_id = H5Fcreate((fullpath.str() + ".h5").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
   H5Pclose(plist_id);

   // 3.All ranks need to create datasets dset*
   hsize_t dims1[4] = {nZ, nY, nX, NCHANNELS * Ngrids};
   fspace_id        = H5Screate_simple(4, dims1, NULL);
   std::stringstream name;
   name << "dset" << std::setfill('0') << std::setw(10) << 0;
   dataset_id = H5Dcreate(file_id, (name.str()).c_str(), get_hdf5_type<hdf5Real>(), fspace_id,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dclose(dataset_id);
   H5Sclose(fspace_id);

   // 4.Each rank now dumps its own blocks to the corresponding dset
   std::vector<hdf5Real> array_block(Ngrids * nX * nY * nZ * NCHANNELS, 0.0);
   size_t count = 0;
   for (unsigned int iz = 0; iz < nZ; iz++)
      for (unsigned int iy = 0; iy < nY; iy++)
         for (unsigned int ix = 0; ix < nX; ix++)
         {
            for (unsigned int m = 0; m < Ngrids; m++) // loop order inefficient AF but works
            {
               B &block = *MyBlocks[m];
               hdf5Real output[NCHANNELS];
               TStreamer::operate(block, ix, iy, iz, (hdf5Real *)output);
               for (unsigned int j = 0; j < NCHANNELS; ++j)
               {
                  array_block[count] = output[j];
                  count++;
               }
            }
         }

   dataset_id = H5Dopen(file_id, (name.str()).c_str(), H5P_DEFAULT);

   H5Dwrite(dataset_id, get_hdf5_type<hdf5Real>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, array_block.data());
   //H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, array_block.data());
   H5Dclose(dataset_id);

   // 5.Close hdf5 file
   H5Fclose(file_id);
   H5close();

   // 6.Write grid meta-data
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
         s << "  <Time Value=\"" << std::scientific << absTime << "\"/>\n\n";
      }

      for (unsigned int m = 0; m < Ngrids; m++)
      {
         BlockInfo I = MyInfos[m];

         s << "  <Grid GridType=\"Uniform\">\n";
         s << "   <Topology TopologyType=\"3DCoRectMesh\" Dimensions=\" " << nZ + 1 << " " << nY + 1
           << " " << nX + 1 << "\"/>\n";
         s << "   <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n";
         s << "   <DataItem Dimensions=\"3\" NumberType=\"Double\" Precision=\"8\" "
              "Format=\"XML\">\n";

         /*ViSit*/
         // s << "    "<<std::scientific << I.origin[0] << " " << I.origin[1] << " " << I.origin[2]
         // << "\n";
         /*Paraview*/
         s << "    " << std::scientific << I.origin[2] << " " << I.origin[1] << " " << I.origin[0]
           << "\n";

         s << "   </DataItem>\n";
         s << "   <DataItem Dimensions=\"3\" NumberType=\"Double\" Precision=\"8\" "
              "Format=\"XML\">\n";
         s << "    " << std::scientific << I.h_gridpoint << " " << I.h_gridpoint << " " << I.h_gridpoint << "\n";
         s << "   </DataItem>\n";
         s << "   </Geometry>\n";

         ////////////////////////////////////
         s << "   <Attribute Name=\"data\" AttributeType=\"" << TStreamer::getAttributeName()
           << "\" Center=\"Cell\">\n";

         s << "<DataItem ItemType=\"HyperSlab\" Dimensions=\" " << nZ << " " << nY << " " << nX
           << " " << NCHANNELS << "\" Type=\"HyperSlab\"> \n";

         s << "<DataItem Dimensions=\"3 4\" Format=\"XML\">\n";
         s << 0 << " " << 0 << " " << 0 << " " << m * NCHANNELS << "\n";
         s << 1 << " " << 1 << " " << 1 << " " << 1 << "\n";
         s << nZ << " " << nY << " " << nX << " " << NCHANNELS << "\n";

         s << "</DataItem>\n";

         s << "   <DataItem ItemType=\"Uniform\"  Dimensions=\" " << nZ << " " << nY << " " << nX
           << " " << Ngrids * NCHANNELS << " "
           << "\" NumberType=\"Float\" Precision=\" " << (int)sizeof(hdf5Real)
           << "\" Format=\"HDF\">\n";

         s << "    " << (filename.str() + ".h5").c_str() << ":/" << name.str() << "\n";
         s << "   </DataItem>\n";
         s << "   </DataItem>\n";
         s << "   </Attribute>\n";
         //////////////////////////////////

         s << "  </Grid>\n\n";
      }
      if (rank == size - 1)
      {
         s << " </Grid>\n";
         s << "</Domain>\n";
         s << "</Xdmf>\n";
      }

      std::string st    = s.str();
      
      FILE *xmf = 0;
      xmf = fopen((fullpath.str()+".xmf").c_str(), "w");
      fprintf(xmf, st.c_str());
      fclose(xmf);

      //MPI_Offset offset = 0;
      //MPI_Offset len    = st.size() * sizeof(char);
      //MPI_File xmf;
      //// delete the xmf file is it exists; no worries if it doesn't
      //MPI_File_delete((fullpath.str() + ".xmf").c_str(), MPI_INFO_NULL);
      //MPI_File_open(comm, (fullpath.str() + ".xmf").c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE,
      //              MPI_INFO_NULL, &xmf);
      //MPI_Exscan(&len, &offset, 1, MPI_OFFSET, MPI_SUM, comm);
      //MPI_File_write_at_all(xmf, offset, st.data(), st.size(), MPI_CHAR, MPI_STATUS_IGNORE);
      //MPI_File_close(&xmf);



   }
#else
   _warn_no_hdf5();
#endif
#endif




#if 0


#ifdef CUBISM_USE_HDF
    typedef typename TGrid::BlockType B;

    // fname is the base filepath tail without file type extension and
    // additional identifiers
    std::ostringstream filename;
    std::ostringstream fullpath;
    filename << fname;
    fullpath << dpath << "/" << filename.str();

    std::vector<BlockInfo> vInfo_local = grid.getBlocksInfo();

    herr_t status;
    hid_t file_id, dataset_id, fspace_id, fapl_id, mspace_id;

    ///////////////////////////////////////////////////////////////////////////
    // startup file
    H5open();
    fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    file_id = H5Fcreate((fullpath.str()+".h5").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
    status = H5Pclose(fapl_id); if(status<0) H5Eprint1(stdout);

    ///////////////////////////////////////////////////////////////////////////
    // write mesh
    std::vector<int> mesh_dims;
    std::vector<std::string> dset_name;
    dset_name.push_back("/vx");
    dset_name.push_back("/vy");
    dset_name.push_back("/vz");
    for (size_t i = 0; i < 3; ++i)
    {
        const MeshMap<B>& m = grid.getMeshMap(i);
        std::vector<double> vertices(m.ncells()+1, m.start());
        mesh_dims.push_back(vertices.size());

        for (size_t j = 0; j < m.ncells(); ++j)
            vertices[j+1] = vertices[j] + m.cell_width(j);

        hsize_t dim[1] = {vertices.size()};
        fspace_id = H5Screate_simple(1, dim, NULL);
#ifndef CUBISM_ON_FERMI
        dataset_id = H5Dcreate(file_id, dset_name[i].c_str(), H5T_NATIVE_DOUBLE, fspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
#else
        dataset_id = H5Dcreate2(file_id, dset_name[i].c_str(), H5T_NATIVE_DOUBLE, fspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
#endif
        status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, vertices.data());
        status = H5Sclose(fspace_id);
        status = H5Dclose(dataset_id);
    }

    ///////////////////////////////////////////////////////////////////////////
    // write data
    const unsigned int NX = static_cast<unsigned int>(grid.getBlocksPerDimension(0))*B::sizeX;
    const unsigned int NY = static_cast<unsigned int>(grid.getBlocksPerDimension(1))*B::sizeY;
    const unsigned int NZ = static_cast<unsigned int>(grid.getBlocksPerDimension(2))*B::sizeZ;
    const unsigned int NCHANNELS = TStreamer::NCHANNELS;

    std::cout << "Allocating " << (NX * NY * NZ * NCHANNELS * sizeof(hdf5Real))/(1024.*1024.*1024.) << " GB of HDF5 data" << std::endl;
    hdf5Real * array_all = new hdf5Real[NX * NY * NZ * NCHANNELS];

    hsize_t count[4]  = {NZ, NY, NX, NCHANNELS};
    hsize_t dims[4]   = {NZ, NY, NX, NCHANNELS};
    hsize_t offset[4] = {0, 0, 0, 0};

#pragma omp parallel for
    for(size_t i=0; i<vInfo_local.size(); i++)
    {
        BlockInfo& info = vInfo_local[i];
        const unsigned int idx[3] = {(unsigned int)info.index[0], (unsigned int)info.index[1], (unsigned int)info.index[2]};
        B & b = *(B*)info.ptrBlock;

        for(int iz=0; iz<static_cast<int>(B::sizeZ); iz++)
            for(int iy=0; iy<static_cast<int>(B::sizeY); iy++)
                for(int ix=0; ix<static_cast<int>(B::sizeX); ix++)
                {
                    hdf5Real output[NCHANNELS];
                    for(unsigned int j=0; j<NCHANNELS; ++j)
                        output[j] = 0;

                    TStreamer::operate(b, ix, iy, iz, (hdf5Real*)output);

                    const unsigned int gx = idx[0]*B::sizeX + ix;
                    const unsigned int gy = idx[1]*B::sizeY + iy;
                    const unsigned int gz = idx[2]*B::sizeZ + iz;

                    hdf5Real * const ptr = array_all + NCHANNELS*(gx + NX * (gy + NY * gz));

                    for(unsigned int j=0; j<NCHANNELS; ++j)
                        ptr[j] = output[j];
                }
    }

    fapl_id = H5Pcreate(H5P_DATASET_XFER);

    fspace_id = H5Screate_simple(4, dims, NULL);
#ifndef CUBISM_ON_FERMI
    dataset_id = H5Dcreate(file_id, "data", get_hdf5_type<hdf5Real>(), fspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
#else
    dataset_id = H5Dcreate2(file_id, "data", get_hdf5_type<hdf5Real>(), fspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
#endif

    fspace_id = H5Dget_space(dataset_id);

    H5Sselect_hyperslab(fspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);

    mspace_id = H5Screate_simple(4, count, NULL);

    status = H5Dwrite(dataset_id, get_hdf5_type<hdf5Real>(), mspace_id, fspace_id, fapl_id, array_all);
    if (status < 0) H5Eprint1(stdout);

    status = H5Sclose(mspace_id); if(status<0) H5Eprint1(stdout);
    status = H5Sclose(fspace_id); if(status<0) H5Eprint1(stdout);
    status = H5Dclose(dataset_id); if(status<0) H5Eprint1(stdout);
    status = H5Pclose(fapl_id); if(status<0) H5Eprint1(stdout);
    status = H5Fclose(file_id); if(status<0) H5Eprint1(stdout);
    H5close();

    delete [] array_all;

    if (bXMF)
    {
        FILE *xmf = 0;
        xmf = fopen((fullpath.str()+".xmf").c_str(), "w");
        fprintf(xmf, "<?xml version=\"1.0\" ?>\n");
        fprintf(xmf, "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n");
        fprintf(xmf, "<Xdmf Version=\"2.0\">\n");
        fprintf(xmf, " <Domain>\n");
        fprintf(xmf, "   <Grid GridType=\"Uniform\">\n");
        fprintf(xmf, "     <Time Value=\"%e\"/>\n\n", absTime);
        fprintf(xmf, "     <Topology TopologyType=\"3DRectMesh\" Dimensions=\"%d %d %d\"/>\n\n", mesh_dims[2], mesh_dims[1], mesh_dims[0]);
        fprintf(xmf, "     <Geometry GeometryType=\"VxVyVz\">\n");
        fprintf(xmf, "       <DataItem Name=\"mesh_vx\" Dimensions=\"%d\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n", mesh_dims[0]);
        fprintf(xmf, "        %s:/vx\n",(filename.str()+".h5").c_str());
        fprintf(xmf, "       </DataItem>\n");
        fprintf(xmf, "       <DataItem Name=\"mesh_vy\" Dimensions=\"%d\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n", mesh_dims[1]);
        fprintf(xmf, "        %s:/vy\n",(filename.str()+".h5").c_str());
        fprintf(xmf, "       </DataItem>\n");
        fprintf(xmf, "       <DataItem Name=\"mesh_vz\" Dimensions=\"%d\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n", mesh_dims[2]);
        fprintf(xmf, "        %s:/vz\n",(filename.str()+".h5").c_str());
        fprintf(xmf, "       </DataItem>\n");
        fprintf(xmf, "     </Geometry>\n\n");
        fprintf(xmf, "     <Attribute Name=\"data\" AttributeType=\"%s\" Center=\"Cell\">\n", TStreamer::getAttributeName());
        fprintf(xmf, "       <DataItem Dimensions=\"%d %d %d %d\" NumberType=\"Float\" Precision=\"%d\" Format=\"HDF\">\n", (int)dims[0], (int)dims[1], (int)dims[2], (int)dims[3], (int)sizeof(hdf5Real));
        fprintf(xmf, "        %s:/data\n",(filename.str()+".h5").c_str());
        fprintf(xmf, "       </DataItem>\n");
        fprintf(xmf, "     </Attribute>\n");
        fprintf(xmf, "   </Grid>\n");
        fprintf(xmf, " </Domain>\n");
        fprintf(xmf, "</Xdmf>\n");
        fclose(xmf);
    }
#else
    _warn_no_hdf5();
#endif
#endif
}


template<typename TStreamer, typename hdf5Real, typename TGrid>
void ReadHDF5(TGrid &grid, const std::string& fname, const std::string& dpath=".")
{
    std::cout<<"mike: ReadHDF5 skipped! \n"; return;

#ifdef CUBISM_USE_HDF
    typedef typename TGrid::BlockType B;

    // fname is the base filepath tail without file type extension and
    // additional identifiers
    std::ostringstream filename;
    std::ostringstream fullpath;
    filename << fname;
    fullpath << dpath << "/" << filename.str();

    herr_t status;
    hid_t file_id, dataset_id, fspace_id, fapl_id, mspace_id;

    const unsigned int NX = static_cast<unsigned int>(grid.getBlocksPerDimension(0))*B::sizeX;
    const unsigned int NY = static_cast<unsigned int>(grid.getBlocksPerDimension(1))*B::sizeY;
    const unsigned int NZ = static_cast<unsigned int>(grid.getBlocksPerDimension(2))*B::sizeZ;
    const unsigned int NCHANNELS = TStreamer::NCHANNELS;

    hdf5Real * array_all = new hdf5Real[NX * NY * NZ * NCHANNELS];

    std::vector<BlockInfo> vInfo_local = grid.getBlocksInfo();

    hsize_t count[4] = {NZ, NY, NX, NCHANNELS};
    hsize_t offset[4] = {0, 0, 0, 0};

    H5open();
    fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    file_id = H5Fopen((fullpath.str()+".h5").c_str(), H5F_ACC_RDONLY, fapl_id);
    status = H5Pclose(fapl_id); if(status<0) H5Eprint1(stdout);

    dataset_id = H5Dopen2(file_id, "data", H5P_DEFAULT);
    fapl_id = H5Pcreate(H5P_DATASET_XFER);

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
}

CUBISM_NAMESPACE_END
