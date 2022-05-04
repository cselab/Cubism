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

#include <hdf5.h>

// Function to retrieve HDF5 type (hid_t) for a given real type.
// If using custom types, the user should specialize this function.
template <typename T> hid_t get_hdf5_type();
template <> inline hid_t get_hdf5_type<long long>() { return H5T_NATIVE_LLONG;}
template <> inline hid_t get_hdf5_type<short int>() { return H5T_NATIVE_SHORT;}
template <> inline hid_t get_hdf5_type<int>      () { return H5T_NATIVE_INT;  }
template <> inline hid_t get_hdf5_type<float>    () { return H5T_NATIVE_FLOAT;}
template <> inline hid_t get_hdf5_type<double>   () { return H5T_NATIVE_DOUBLE;}

#include "BlockInfo.h"

namespace cubism {

template<typename TStreamer, typename hdf5Real, typename TGrid>
void DumpHDF5_uniform(const TGrid &grid, const typename TGrid::Real absTime, const std::string &fname, const std::string &dpath = ".")
{
  //only for 2D!

  typedef typename TGrid::BlockType B;
  const unsigned int nX = B::sizeX;
  const unsigned int nY = B::sizeY;
  // const unsigned int nZ = B::sizeZ;

  // fname is the base filepath without file type extension
  std::ostringstream filename;
  std::ostringstream fullpath;
  filename << fname;
  fullpath << dpath << "/" << filename.str();
  std::vector<B *> MyBlocks      = grid.GetBlocks();
  std::vector<BlockInfo> MyInfos = grid.getBlocksInfo();
  const int levelMax = grid.getlevelMax();
  std::array<int, 3> bpd = grid.getMaxBlocks();
  const unsigned int unx = bpd[0]*(1<<(levelMax-1))*nX;
  const unsigned int uny = bpd[1]*(1<<(levelMax-1))*nY;
  //const int unz = bpd[2]*(1<<(levelMax-1))*nZ;
  const unsigned int NCHANNELS = TStreamer::NCHANNELS;
  double hmin = 1e10;
  for (size_t i = 0 ; i < MyInfos.size() ; i ++) hmin = std::min(hmin,MyInfos[i].h);
  const double h = hmin;

  // TODO: Refactor, move the interpolation logic into a separate function at
  // the level of a Grid, see copyToUniformNoInterpolation for reference.

  std::vector <float> uniform_mesh(uny*unx*NCHANNELS);
  for (size_t i = 0 ; i < MyInfos.size() ; i ++)
  {
    const BlockInfo & info = MyInfos[i];
    const int level = info.level;

    for (unsigned int y = 0; y < nY; y++)
    for (unsigned int x = 0; x < nX; x++)
    {
      B & block = * (B*)info.ptrBlock;

      float output[NCHANNELS]={0.0};
      float dudx  [NCHANNELS]={0.0};
      float dudy  [NCHANNELS]={0.0};
      TStreamer::operate(block, x, y, 0, (float *)output);

      if (x!= 0 && x!= nX-1)
      {
        float output_p [NCHANNELS]={0.0};
        float output_m [NCHANNELS]={0.0};
        TStreamer::operate(block, x+1, y, 0, (float *)output_p);
        TStreamer::operate(block, x-1, y, 0, (float *)output_m);
        for (unsigned int j = 0; j < NCHANNELS; ++j)
          dudx[j] = 0.5*(output_p[j]-output_m[j]);
      }
      else if (x==0)
      {
        float output_p [NCHANNELS]={0.0};
        TStreamer::operate(block, x+1, y, 0, (float *)output_p);
        for (unsigned int j = 0; j < NCHANNELS; ++j)
          dudx[j] = output_p[j]-output[j];        
      }
      else
      {
        float output_m [NCHANNELS]={0.0};
        TStreamer::operate(block, x-1, y, 0, (float *)output_m);
        for (unsigned int j = 0; j < NCHANNELS; ++j)
          dudx[j] = output[j]-output_m[j];        
      }

      if (y!= 0 && y!= nY-1)
      {
        float output_p [NCHANNELS]={0.0};
        float output_m [NCHANNELS]={0.0};
        TStreamer::operate(block, x, y+1, 0, (float *)output_p);
        TStreamer::operate(block, x, y-1, 0, (float *)output_m);
        for (unsigned int j = 0; j < NCHANNELS; ++j)
          dudy[j] = 0.5*(output_p[j]-output_m[j]);
      }
      else if (y==0)
      {
        float output_p [NCHANNELS]={0.0};
        TStreamer::operate(block, x, y+1, 0, (float *)output_p);
        for (unsigned int j = 0; j < NCHANNELS; ++j)
          dudy[j] = output_p[j]-output[j];        
      }
      else
      {
        float output_m [NCHANNELS]={0.0};
        TStreamer::operate(block, x, y-1, 0, (float *)output_m);
        for (unsigned int j = 0; j < NCHANNELS; ++j)
          dudy[j] = output[j]-output_m[j];        
      }

      int iy_start = (info.index[1]*nY + y)*(1<< ( (levelMax-1)-level ) );
      int ix_start = (info.index[0]*nX + x)*(1<< ( (levelMax-1)-level ) );

      const int points = 1<< ( (levelMax-1)-level ); 
      const double dh = 1.0/points;

      for (int iy = iy_start; iy< iy_start + (1<< ( (levelMax-1)-level ) ); iy++)
      for (int ix = ix_start; ix< ix_start + (1<< ( (levelMax-1)-level ) ); ix++)
      {
        double cx = (ix - ix_start - points/2 + 1 - 0.5)*dh;
        double cy = (iy - iy_start - points/2 + 1 - 0.5)*dh;
        for (unsigned int j = 0; j < NCHANNELS; ++j)
          uniform_mesh[iy*NCHANNELS*unx+ix*NCHANNELS+j] = output[j]+ cx*dudx[j]+ cy*dudy[j];
      }
    }

  }

  hid_t file_id, dataset_id, fspace_id, plist_id;
  H5open();
  // 1.Set up file access property list with parallel I/O access
  // 2.Create a new file collectively and release property list identifier.
  // 3.All ranks need to create datasets dset*
  hsize_t dims[4] = {1, uny, unx, NCHANNELS};

  plist_id   = H5Pcreate(H5P_FILE_ACCESS);
  file_id    = H5Fcreate((fullpath.str() + "uniform.h5").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
  fspace_id  = H5Screate_simple(4, dims, NULL);
  dataset_id = H5Dcreate(file_id, "dset",H5T_NATIVE_FLOAT, fspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Pclose(plist_id);
  H5Dclose(dataset_id);
  H5Sclose(fspace_id);

  dataset_id = H5Dopen(file_id, "dset", H5P_DEFAULT);
  H5Dwrite(dataset_id,H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, uniform_mesh.data());
  H5Dclose(dataset_id);

  // 5.Close hdf5 file
  H5Fclose(file_id);
  H5close();

   // 6.Write grid meta-data
   {
     FILE *xmf = 0;
     xmf = fopen((fullpath.str()+"uniform.xmf").c_str(), "w");
     fprintf(xmf, "<?xml version=\"1.0\" ?>\n");
     fprintf(xmf, "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n");
     fprintf(xmf, "<Xdmf Version=\"2.0\">\n");
     fprintf(xmf, " <Domain>\n");
     fprintf(xmf, "   <Grid GridType=\"Uniform\">\n");
     fprintf(xmf, "     <Time Value=\"%e\"/>\n\n", absTime);
     fprintf(xmf, "     <Topology TopologyType=\"3DCoRectMesh\" Dimensions=\"%d %d %d\"/>\n\n", 1+1, uny+1, unx+1);
     fprintf(xmf, "     <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n");
     fprintf(xmf, "       <DataItem Dimensions=\"3\" NumberType=\"Double\" Precision=\"8\" " "Format=\"XML\">\n");
     fprintf(xmf, "        %e %e %e\n",0.0,0.0,0.0);
     fprintf(xmf, "       </DataItem>\n");
     fprintf(xmf, "       <DataItem Dimensions=\"3\" NumberType=\"Double\" Precision=\"8\" " "Format=\"XML\">\n");
     fprintf(xmf, "        %e %e %e\n",h,h,h);
     fprintf(xmf, "       </DataItem>\n");
     fprintf(xmf, "     </Geometry>\n\n");
     fprintf(xmf, "     <Attribute Name=\"dset\" AttributeType=\"%s\" Center=\"Cell\">\n", TStreamer::getAttributeName());
     fprintf(xmf, "       <DataItem Dimensions=\"%d %d %d %d\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n", 1, uny, unx, NCHANNELS);
     fprintf(xmf, "        %s:/dset\n",(filename.str()+"uniform.h5").c_str());
     fprintf(xmf, "       </DataItem>\n");
     fprintf(xmf, "     </Attribute>\n");
     fprintf(xmf, "   </Grid>\n");
     fprintf(xmf, " </Domain>\n");
     fprintf(xmf, "</Xdmf>\n");
     fclose(xmf);
   }
}

// The following requirements for the data TStreamer are required:
// TStreamer::NCHANNELS        : Number of data elements (1=Scalar, 3=Vector, 9=Tensor)
// TStreamer::operate          : Data access methods for read and write
// TStreamer::getAttributeName : Attribute name of the date ("Scalar", "Vector", "Tensor")
template<typename TStreamer, typename hdf5Real, typename TGrid>
void DumpHDF5(const TGrid &grid, const typename TGrid::Real absTime, const std::string &fname, const std::string &dpath = ".", const bool bXMF = true)
{
  typedef typename TGrid::BlockType B;
  const int nX = B::sizeX;
  const int nY = B::sizeY;
  const int nZ = B::sizeZ;

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
  #endif

  int start = 0;
  // Write grid meta-data
  {
    std::ostringstream myfilename;
    myfilename << filename.str();
    std::stringstream s;
    s << "<?xml version=\"1.0\" ?>\n";
    s << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
    s << "<Xdmf Version=\"2.0\">\n";
    s << "<Domain>\n";
    s << " <Grid Name=\"OctTree\" GridType=\"Collection\">\n";
    s << "  <Time Value=\"" << std::scientific << absTime << "\"/>\n\n";
    for (size_t groupID = 0 ; groupID < MyGroups.size() ; groupID ++)
    {
      const BlockGroup & group = MyGroups[groupID];
      const int nXX = group.NXX;
      const int nYY = group.NYY;
      const int nZZ = group.NZZ;

      s << "  <Grid GridType=\"Uniform\">\n";
      s << "   <Topology TopologyType=\"3DCoRectMesh\" Dimensions=\" "<<nZZ<<" "<<nYY<<" "<<nXX<<"\"/>\n";
      s << "   <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n";
      s << "   <DataItem Dimensions=\"3\" NumberType=\"Double\" Precision=\"8\" " "Format=\"XML\">\n";
      s << "    " << std::scientific <<group.origin[2]<<" "<<group.origin[1]<<" "<<group.origin[0]<< "\n";
      s << "   </DataItem>\n";
      s << "   <DataItem Dimensions=\"3\" NumberType=\"Double\" Precision=\"8\" " "Format=\"XML\">\n";
      #if DIMENSION == 3
        s << "    " << std::scientific <<group.h<<" "<<group.h <<" "<< group.h << "\n";
      #else
        s << "    " << std::scientific <<hmin<<" "<<group.h <<" "<< group.h << "\n";
      #endif
      s << "   </DataItem>\n";
      s << "   </Geometry>\n";
      int dd = (nZZ - 1)*(nYY - 1)*(nXX - 1);
      s << "   <Attribute Name=\"data\" AttributeType=\"" << "Scalar"<< "\" Center=\"Cell\">\n";
      s << "<DataItem ItemType=\"HyperSlab\" Dimensions=\" " << 1 << " " << 1 << " " << dd <<  "\" Type=\"HyperSlab\"> \n";
      s << "<DataItem Dimensions=\"3 1\" Format=\"XML\">\n";
      s << start<<"\n";
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
    s << " </Grid>\n";
    s << "</Domain>\n";
    s << "</Xdmf>\n";
    std::string st    = s.str();

    FILE *xmf = 0;
    xmf = fopen((fullpath.str()+".xmf").c_str(), "w");
    fprintf(xmf, st.c_str());
    fclose(xmf);
  }
    
  hid_t file_id, dataset_id, fspace_id, fapl_id;
    
  H5open();

  //1.Set up file access property list
  fapl_id = H5Pcreate(H5P_FILE_ACCESS);

  //2.Create a new file and release property list identifier.
  file_id = H5Fcreate((fullpath.str()+".h5").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);        
  H5Pclose(fapl_id);

  //3.Create dataset
  fapl_id = H5Pcreate(H5P_DATASET_XFER);
  int total = start;
  hsize_t dims[1]  = {(hsize_t) total};
  fspace_id        = H5Screate_simple(1, dims, NULL);
  dataset_id       = H5Dcreate (file_id, "dset", get_hdf5_type<hdf5Real>(), fspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  H5Dclose(dataset_id);
  //4.Dump
  int start1 = 0;
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
      B & block = * (B*)I.ptrBlock;
      for (int iz = 0; iz < nZ; iz++)
      for (int iy = 0; iy < nY; iy++)
      for (int ix = 0; ix < nX; ix++)
      {
        hdf5Real output[NCHANNELS];
        TStreamer::operate(block,ix,iy,iz,output);
        const int iz_b = (kB-group.i_min[2])*nZ + iz;
        const int iy_b = (jB-group.i_min[1])*nY + iy;
        const int ix_b = (iB-group.i_min[0])*nX + ix;
        const int base = iz_b *(nX_max*nY_max) + iy_b *nX_max + ix_b;               
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
    for (int j = 0 ; j < dd1 ; j ++) bigArray[start1 + j] = array_block[j];
    start1 += dd1;
  }

  dataset_id = H5Dopen(file_id, "dset", H5P_DEFAULT);
  H5Dwrite(dataset_id, get_hdf5_type<hdf5Real>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, bigArray.data());
  H5Sclose(fspace_id);
  H5Dclose(dataset_id);
  H5Pclose(fapl_id);
  H5Fclose(file_id);
  H5close();
}

template<typename TStreamer, typename hdf5Real, typename TGrid>
void DumpHDF5_groups(TGrid &grid,const typename TGrid::Real absTime,const std::string &fname,const std::string &dpath = ".",const bool bXMF = true)
{
  DumpHDF5<TStreamer,hdf5Real,TGrid>(grid,absTime,fname,dpath,bXMF);
}

template<typename TStreamer, typename hdf5Real, typename TGrid>
void ReadHDF5(TGrid &grid, const std::string& fname, const std::string& dpath=".")
{
  std::cout<<"ReadHDF5 is only implemented for MPI (no serial version). \n"; return;
}

}//namespace cubism
