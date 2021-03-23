//
//  HDF5Dumper.h
//  Cubism
//
//  Created by Michalis Chatzimanolakis on 20.10.2020
//  Copyright 2020 ETH Zurich. All rights reserved.
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
#include <iomanip> // std::setfill, std::setw


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
              const typename TGrid::Real absTime,
              const std::string &fname,
              const std::string &dpath = ".",
              const bool bXMF = true)
{
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

   // std::cout << " ---> Rank " << rank << " is dumping " << Ngrids << " Blocks.\n";

   hid_t file_id, dataset_id, fspace_id, plist_id;

   H5open();

   // 1.Set up file access property list with parallel I/O access
   plist_id = H5Pcreate(H5P_FILE_ACCESS);

   // 2.Create a new file collectively and release property list identifier.
   file_id = H5Fcreate((fullpath.str() + ".h5").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
   H5Pclose(plist_id);

   // 3.All ranks need to create datasets dset*
#if DIMENSION == 2
   hsize_t dims1[3] = {nY, nX, NCHANNELS * Ngrids};
   fspace_id        = H5Screate_simple(3, dims1, NULL);
#else
   hsize_t dims1[4] = {nZ, nY, nX, NCHANNELS * Ngrids};
   fspace_id        = H5Screate_simple(4, dims1, NULL);
#endif
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
               hdf5Real output[NCHANNELS]={0.0};//initialize to silence warning
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
#if DIMENSION == 2
         s << "   <Topology TopologyType=\"2DCoRectMesh\" Dimensions=\" " << nY + 1 << " " << nX + 1 << "\"/>\n";
         s << "   <Geometry GeometryType=\"ORIGIN_DXDY\">\n";
         s << "   <DataItem Dimensions=\"2\" NumberType=\"Double\" Precision=\"8\" " "Format=\"XML\">\n";
         s << "    " << std::scientific << I.origin[1] << " " << I.origin[0] << "\n";
         s << "   </DataItem>\n";
         s << "   <DataItem Dimensions=\"2\" NumberType=\"Double\" Precision=\"8\" " "Format=\"XML\">\n";
         s << "    " << std::scientific << I.h_gridpoint << " " << I.h_gridpoint << "\n";
         s << "   </DataItem>\n";
         s << "   </Geometry>\n";
#else
         s << "   <Topology TopologyType=\"2DCoRectMesh\" Dimensions=\" " << nZ + 1 << " " << nY + 1 << " " << nX + 1 << "\"/>\n";
         s << "   <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n";
         s << "   <DataItem Dimensions=\"3\" NumberType=\"Double\" Precision=\"8\" " "Format=\"XML\">\n";
         s << "    " << std::scientific << I.origin[2] << " " << I.origin[1] << " " << I.origin[0] << "\n";
         s << "   </DataItem>\n";
         s << "   <DataItem Dimensions=\"3\" NumberType=\"Double\" Precision=\"8\" " "Format=\"XML\">\n";
         s << "    " << std::scientific << I.h_gridpoint << " " << I.h_gridpoint << " " << I.h_gridpoint << "\n";
         s << "   </DataItem>\n";
         s << "   </Geometry>\n";
#endif


         ////////////////////////////////////
         s << "   <Attribute Name=\"data\" AttributeType=\"" << TStreamer::getAttributeName()
           << "\" Center=\"Cell\">\n";

#if DIMENSION == 2
         s << "<DataItem ItemType=\"HyperSlab\" Dimensions=\" " << nY << " " << nX
           << " " << NCHANNELS << "\" Type=\"HyperSlab\"> \n";
         s << "<DataItem Dimensions=\"3 3\" Format=\"XML\">\n";
         s <<  0 << " " << 0  << " " << m * NCHANNELS << "\n";
         s <<  1 << " " << 1  << " " << 1             << "\n";
         s << nY << " " << nX << " " << NCHANNELS     << "\n";
#else
         s << "<DataItem ItemType=\"HyperSlab\" Dimensions=\" " << nZ << " " << nY << " " << nX
           << " " << NCHANNELS << "\" Type=\"HyperSlab\"> \n";
         s << "<DataItem Dimensions=\"3 3\" Format=\"XML\">\n";
         s << 0 << " " <<  0 << " " << 0  << " " << m * NCHANNELS << "\n";
         s << 1 << " " <<  1 << " " << 1  << " " << 1             << "\n";
         s << nZ<< " " << nY << " " << nX << " " << NCHANNELS     << "\n";
#endif

         s << "</DataItem>\n";

#if DIMENSION == 2
         s << "   <DataItem ItemType=\"Uniform\"  Dimensions=\" " << nY << " " << nX
           << " " << Ngrids * NCHANNELS << " "
#else
         s << "   <DataItem ItemType=\"Uniform\"  Dimensions=\" " << nZ << " " << nY << " " << nX
           << " " << Ngrids * NCHANNELS << " "
#endif
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
   }
#else
   _warn_no_hdf5();
#endif
}

//Used when iCounter is passed, in Cubism-MPCF
template<typename TStreamer, typename hdf5Real, typename TGrid>
void DumpHDF5(const TGrid &grid,
              const int iCounter,
              const typename TGrid::Real absTime,
              const std::string &fname,
              const std::string &dpath = ".",
              const bool bXMF = true)
{
  DumpHDF5<TStreamer,hdf5Real,TGrid>(grid,absTime,fname,dpath,bXMF);
}


template<typename TStreamer, typename hdf5Real, typename TGrid>
void ReadHDF5(TGrid &grid, const std::string& fname, const std::string& dpath=".")
{
    std::cout<<"mike: ReadHDF5 skipped! \n"; return;
#if 0
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
#endif
}

CUBISM_NAMESPACE_END
