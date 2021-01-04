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

#if 1
#include <vtkCell.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkDoubleArray.h>
#include <vtkNonOverlappingAMR.h>
#include <vtkUniformGrid.h>
#include <vtkXMLPUniformGridAMRWriter.h>
#include <vtkXMLUniformGridAMRWriter.h>
#endif

CUBISM_NAMESPACE_BEGIN

// The following requirements for the data TStreamer are required:
// TStreamer::NCHANNELS        : Number of data elements (1=Scalar, 3=Vector, 9=Tensor)
// TStreamer::operate          : Data access methods for read and write
// TStreamer::getAttributeName : Attribute name of the date ("Scalar", "Vector", "Tensor")
template <typename TStreamer, typename hdf5Real, typename TGrid> 
void DumpHDF5_MPI(TGrid &grid, typename TGrid::Real absTime, const std::string &fname, const std::string &dpath = ".", const bool bXMF = true)
{
    #if 1 // VTK AMR-dataset
    std::ostringstream filename;
    std::ostringstream fullpath;
    filename << fname;// fname is the base filepath without file type extension
    fullpath << dpath << "/" << filename.str();

    MPI_Comm comm = grid.getWorldComm();
    const unsigned int NCHANNELS = TStreamer::NCHANNELS;
    typedef typename TGrid::BlockType B;
    static const unsigned int nX = B::sizeX;
    static const unsigned int nY = B::sizeY;
    static const unsigned int nZ = B::sizeZ;

    std::vector<B *> MyBlocks      = grid.GetBlocks();
    std::vector<BlockInfo> MyInfos = grid.getBlocksInfo();

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    int numberOfLevels = grid.getlevelMax();

    std::vector<BlockGroup> & MyGroups = grid.MyGroups;
    grid.UpdateMyGroups();

    std::vector<int> blocksPerLevel(numberOfLevels);
    for (unsigned int m = 0; m < MyGroups.size(); m++)
        blocksPerLevel[MyGroups[m].level] ++;

    std::vector<int> TotalblocksPerLevel(numberOfLevels);
    MPI_Allreduce(blocksPerLevel.data(), TotalblocksPerLevel.data(), numberOfLevels, MPI_INT, MPI_SUM,comm);

    std::vector<int> BaseblocksPerLevel(numberOfLevels);  
    MPI_Exscan(blocksPerLevel.data(), BaseblocksPerLevel.data(), numberOfLevels, MPI_INT,  MPI_SUM , comm);

    std::vector<int> CountblocksPerLevel(numberOfLevels,0);  

    vtkNonOverlappingAMR* amrGrid = vtkNonOverlappingAMR::New();
    amrGrid->Initialize(numberOfLevels, TotalblocksPerLevel.data());

    for (unsigned int m = 0; m < MyGroups.size(); m++)
    {
        BlockGroup & info = MyGroups[m];
        vtkUniformGrid* g = vtkUniformGrid::New();
        g->SetSpacing(info.h, info.h, info.h);
        g->SetOrigin(info.origin[0], info.origin[1], info.origin[2]);
        g->SetExtent(0, info.NXX-1, 0, info.NYY-1, 0, info.NZZ-1);

        vtkSmartPointer<vtkDoubleArray> xyz = vtkSmartPointer<vtkDoubleArray>::New();
        xyz->SetName("data");
        xyz->SetNumberOfComponents(NCHANNELS);
        xyz->SetNumberOfTuples(g->GetNumberOfCells());

        const unsigned int nX_max = info.NXX-1;
        const unsigned int nY_max = info.NYY-1;
        //const unsigned int nZ_max = info.NZZ-1;
        for (int kB = info.i_min[2]; kB <= info.i_max[2]; kB++)
        for (int jB = info.i_min[1]; jB <= info.i_max[1]; jB++)
        for (int iB = info.i_min[0]; iB <= info.i_max[0]; iB++)
        {
            B &block = *grid.avail1(iB,jB,kB,info.level);
            for (unsigned int iz = 0; iz < nZ; iz++)
            for (unsigned int iy = 0; iy < nY; iy++)
            for (unsigned int ix = 0; ix < nX; ix++)
            {
                double output[NCHANNELS];
                TStreamer::operate(block,ix,iy,iz,output);
                const int base = ( ((kB-info.i_min[2])*nZ + iz)*(nX_max*nY_max) + 
                                   ((jB-info.i_min[1])*nY + iy)*(nX_max) + 
                                   ((iB-info.i_min[0])*nX + ix) );
                xyz->SetTuple(base, output);
            }
        }
        g->GetCellData()->AddArray(xyz);
        amrGrid->SetDataSet(info.level, BaseblocksPerLevel[info.level] + CountblocksPerLevel[info.level], g);
        CountblocksPerLevel[info.level]++;
        g->Delete();
    }

    std::vector<int> CountblocksPerLevel1(numberOfLevels,0);

    std::string FNAME = (fullpath.str() + ".vthb").c_str();
    auto writer = vtkSmartPointer<vtkXMLUniformGridAMRWriter>::New();
    writer->SetFileName(FNAME.c_str());
    writer->SetInputData(amrGrid);
    writer->Write();

    if (bXMF)
    {
        std::stringstream s;
        if (rank == 0)
        {
            s << "<?xml version=\"1.0\"?>\n";
            s << "<VTKFile type=\"vtkNonOverlappingAMR\" version=\"1.1\" byte_order=\"LittleEndian\" header_type=\"UInt32\" compressor=\"vtkZLibDataCompressor\">\n";
            s << "  <vtkNonOverlappingAMR>\n";
            s << "  <Time Value=\"" << std::scientific << absTime << "\"/>\n\n";
        }
        for (unsigned int m = 0; m < MyGroups.size(); m++)
        {
            BlockGroup & I = MyGroups[m];
            int nID = BaseblocksPerLevel[I.level] + CountblocksPerLevel1[I.level];
            CountblocksPerLevel1[I.level]++;
            for (int l = 0 ; l < I.level ; l++)
            {
                nID += TotalblocksPerLevel[l];
            }
            s <<"     <Block level=\"" + std::to_string(I.level) + "\"> \n";
            s <<"       <DataSet index=\""+ std::to_string(nID) + "\" file=\"" + filename.str() + "/" +filename.str() + "_" + std::to_string(nID)+ ".vti\"/>\n";
                                                                                 
            s <<"     </Block>\n";
        }
        if (rank == size - 1)
        {
           s << "   </vtkNonOverlappingAMR>\n";
           s << " </VTKFile>\n";
        }
        std::string st    = s.str();
        MPI_Offset offset = 0;
        MPI_Offset len    = st.size() * sizeof(char);
        MPI_File xmf;
        MPI_File_delete((fullpath.str() + ".vthb").c_str(), MPI_INFO_NULL);// delete the xmf file is it exists; no worries if it doesn't
        MPI_File_open(comm, (fullpath.str() + ".vthb").c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE,MPI_INFO_NULL, &xmf);
        MPI_Exscan(&len, &offset, 1, MPI_OFFSET, MPI_SUM, comm);
        MPI_File_write_at_all(xmf, offset, st.data(), st.size(), MPI_CHAR, MPI_STATUS_IGNORE);
        MPI_File_close(&xmf);
    }
    #else //HDF5
    typedef typename TGrid::BlockType B;
    static const unsigned int nX = B::sizeX;
    static const unsigned int nY = B::sizeY;
    static const unsigned int nZ = B::sizeZ;

    MPI_Comm comm = grid.getWorldComm();
    int rank, size;
    rank = grid.myrank;
    size = grid.world_size;
    const unsigned int NCHANNELS = TStreamer::NCHANNELS;
    std::ostringstream filename;
    std::ostringstream fullpath;
    filename << fname;// fname is the base filepath without file type extension
    fullpath << dpath << "/" << filename.str();

    std::vector<BlockGroup> & MyGroups = grid.MyGroups;
    grid.UpdateMyGroups();

    // 2. Write grid meta-data
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
      for (size_t groupID = 0 ; groupID < MyGroups.size() ; groupID ++)
      {
        std::stringstream name;
        name << "dset" << std::setfill('0') << std::setw(10) << rank;
        name << "_"    << std::setfill('0') << std::setw(10) << groupID;
        const BlockGroup & group = MyGroups[groupID];
        const int nXX = group.NXX;
        const int nYY = group.NYY;
        const int nZZ = group.NZZ;
        s << "  <Grid GridType=\"Uniform\">\n";
        s << "   <Topology TopologyType=\"3DCoRectMesh\" Dimensions=\" " << nZZ << " " << nYY << " " << nXX << "\"/>\n";
        s << "   <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n";
        s << "   <DataItem Dimensions=\"3\" NumberType=\"Double\" Precision=\"8\" " "Format=\"XML\">\n";
        /*ViSit*/
        //s << "    " << std::scientific << group.origin[0] << " " << group.origin[1] << " " << group.origin[2] << "\n";
        /*Paraview*/
        s << "    " << std::scientific << group.origin[2] << " " << group.origin[1] << " " << group.origin[0] << "\n";
        s << "   </DataItem>\n";
        s << "   <DataItem Dimensions=\"3\" NumberType=\"Double\" Precision=\"8\" " "Format=\"XML\">\n";
        s << "    " << std::scientific << group.h << " " << group.h << " " << group.h << "\n";
        s << "   </DataItem>\n";
        s << "   </Geometry>\n";
        ////////////////////////////////////
        s << "   <Attribute Name=\"data\" AttributeType=\"" << TStreamer::getAttributeName() << "\" Center=\"Cell\">\n";
        s << "<DataItem ItemType=\"HyperSlab\" Dimensions=\" " << nZZ - 1 << " " << nYY - 1 << " " << nXX - 1 << " " << NCHANNELS <<  "\" Type=\"HyperSlab\"> \n";
        s << "<DataItem Dimensions=\"3 4\" Format=\"XML\">\n";
        s << 0       << " " << 0        << " " << 0       << " " <<  0         << "\n";
        s << 1       << " " << 1        << " " << 1       << " " <<  1         << "\n";
        s << nZZ -1  << " " << nYY - 1  << " " << nXX - 1 << " " <<  NCHANNELS << "\n";
        s << "</DataItem>\n";
        s << "   <DataItem ItemType=\"Uniform\"  Dimensions=\" " << nZZ - 1 << " " << nYY - 1 << " " << nXX - 1 << " " << NCHANNELS << " " << "\" NumberType=\"Float\" Precision=\" " << (int)sizeof(hdf5Real) << "\" Format=\"HDF\">\n";
        
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
      MPI_Offset offset = 0;
      MPI_Offset len    = st.size() * sizeof(char);
      MPI_File xmf;
      MPI_File_delete((fullpath.str() + ".xmf").c_str(), MPI_INFO_NULL); // delete the xmf file is it exists
      MPI_File_open(comm, (fullpath.str() + ".xmf").c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &xmf);
      MPI_Exscan(&len, &offset, 1, MPI_OFFSET, MPI_SUM, comm);
      MPI_File_write_at_all(xmf, offset, st.data(), st.size(), MPI_CHAR, MPI_STATUS_IGNORE);
      MPI_File_close(&xmf);
    }

    //3. Dump data
    hid_t file_id, dataset_id, fspace_id, plist_id;

    H5open();

    // 3a.Set up file access property list with parallel I/O access
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL);

    // 3b.Create a new file collectively and release property list identifier.
    file_id = H5Fcreate((fullpath.str() + ".h5").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    //file_id = H5Fcreate((fullpath.str() + ".h5").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    H5Pclose(plist_id);

    // 3c.All ranks need to create datasets dset*
    std::vector<unsigned int> & Groups_per_rank = grid.Groups_per_rank;
    std::vector<unsigned int> & allN            = grid.allN           ;

    int Npos = 0;
    for (int r = 0; r < size; r++)
    {
      for (size_t groupID = 0 ; groupID < Groups_per_rank[r] ; groupID ++)
      {
        hsize_t cdims[4] = {8,8,8,1};
        hid_t plist_id1 = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(plist_id1, 4, cdims);        
        //H5Pset_deflate(plist_id1, 6);
        hsize_t dims1[4] = {allN[Npos+2] - 1, allN[Npos+1] - 1, allN[Npos] - 1, NCHANNELS};
        Npos += 3;
        fspace_id        = H5Screate_simple(4, dims1, NULL);
        std::stringstream name;
        name << "dset" << std::setfill('0') << std::setw(10) << r;
        name << "_"    << std::setfill('0') << std::setw(10) << groupID;
        //dataset_id = H5Dcreate(file_id, (name.str()).c_str(), get_hdf5_type<hdf5Real>(), fspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        dataset_id = H5Dcreate(file_id, (name.str()).c_str(), get_hdf5_type<hdf5Real>(), fspace_id, H5P_DEFAULT, plist_id1, H5P_DEFAULT);
        H5Dclose(dataset_id);
        H5Sclose(fspace_id);
        H5Pclose(plist_id1);
      }
    }

    //hid_t plist_id2 = H5Pcreate(H5P_DATASET_XFER);
    //                  H5Pset_dxpl_mpio(plist_id2, H5FD_MPIO_COLLECTIVE);

    // 3d.Each rank now dumps its own blocks to the corresponding dset
    for (size_t groupID = 0 ; groupID < MyGroups.size() ; groupID ++)
    {
    //for (int r = 0; r < size; r++)
    //{
    //  if (r == rank)
    //  for (size_t groupID = 0 ; groupID < Groups_per_rank[r] ; groupID ++)
    //  {
      std::stringstream name;
      name << "dset" << std::setfill('0') << std::setw(10) << rank;
      name << "_"    << std::setfill('0') << std::setw(10) << groupID;
      const BlockGroup & group = MyGroups[groupID];
      const unsigned int nX_max = group.NXX-1;
      const unsigned int nY_max = group.NYY-1;
      const unsigned int nZ_max = group.NZZ-1;
      std::vector<hdf5Real> array_block(nX_max * nY_max * nZ_max * NCHANNELS, 0.0);
      for (int kB = group.i_min[2]; kB <= group.i_max[2]; kB++)
      for (int jB = group.i_min[1]; jB <= group.i_max[1]; jB++)
      for (int iB = group.i_min[0]; iB <= group.i_max[0]; iB++)
      {
        B &block = *grid.avail1(iB,jB,kB,group.level);
        for (unsigned int iz = 0; iz < nZ; iz++)
        for (unsigned int iy = 0; iy < nY; iy++)
        for (unsigned int ix = 0; ix < nX; ix++)
        {
          hdf5Real output[NCHANNELS];
          TStreamer::operate(block,ix,iy,iz,(hdf5Real *)output);
          const int base = ( ((kB-group.i_min[2])*nZ + iz)*(nX_max*nY_max) + 
                             ((jB-group.i_min[1])*nY + iy)*(nX_max) + 
                             ((iB-group.i_min[0])*nX + ix) )*NCHANNELS;
          for (unsigned int j = 0; j < NCHANNELS; ++j)
              array_block[base+j] = output[j];
        }
      }
      dataset_id = H5Dopen(file_id, (name.str()).c_str(), H5P_DEFAULT);
      H5Dwrite(dataset_id, get_hdf5_type<hdf5Real>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, array_block.data());
      //H5Dwrite(dataset_id, get_hdf5_type<hdf5Real>(), H5S_ALL, H5S_ALL, plist_id2, array_block.data());
      H5Dclose(dataset_id);         
    //}
    //else
    //{
    //  for (size_t groupID = 0 ; groupID < Groups_per_rank[r] ; groupID ++)
    //  {
    //    std::stringstream name;
    //    name << "dset" << std::setfill('0') << std::setw(10) << r;
    //    name << "_"    << std::setfill('0') << std::setw(10) << groupID;
    //    dataset_id = H5Dopen(file_id, (name.str()).c_str(), H5P_DEFAULT);
    //    H5Dwrite(dataset_id, get_hdf5_type<hdf5Real>(), H5S_ALL, H5S_ALL, plist_id2, NULL);
    //    H5Dclose(dataset_id);
    //  }
    //}
    }
   //H5Pclose(plist_id2);
   // 5.Close hdf5 file
   H5Fclose(file_id);
   H5close();
   #endif
}

template <typename TStreamer, typename hdf5Real, typename TGrid>
void DumpHDF5_MPI(/*const*/ TGrid &grid, const int iCounter, /*const*/ typename TGrid::Real absTime,
                  const std::string &fname, const std::string &dpath = ".", const bool bXMF = true)
{
  DumpHDF5_MPI<TStreamer,hdf5Real,TGrid>(grid,absTime,fname,dpath,bXMF);
}

template <typename TStreamer, typename hdf5Real, typename TGrid>
void ReadHDF5_MPI(TGrid &grid, const std::string &fname, const std::string &dpath = ".")
{
   std::cout << "mike: ReadHDF5_MPI skipped! \n";
   return;
   #if 0 // mike
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

#if 0
// The following requirements for the data TStreamer are required:
// TStreamer::NCHANNELS        : Number of data elements (1=Scalar, 3=Vector, 9=Tensor)
// TStreamer::operate          : Data access methods for read and write
// TStreamer::getAttributeName : Attribute name of the date ("Scalar", "Vector", "Tensor")
template <typename TStreamer, typename hdf5Real, typename TGrid>
void DumpHDF5_MPI(const TGrid &grid, const typename TGrid::Real absTime,
                  const std::string &fname, const std::string &dpath = ".", const bool bXMF = true)
{
#ifdef CUBISM_USE_HDF

   MPI_Comm comm = grid.getWorldComm();
   int rank, size;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &size);

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

   std::cout << " ---> Rank " << rank << " is dumping " << Ngrids << " Blocks." << std::endl;;

   hid_t file_id, dataset_id, fspace_id, plist_id;

   H5open();

   // 1.Set up file access property list with parallel I/O access
   plist_id = H5Pcreate(H5P_FILE_ACCESS);
   H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL);

   // 2.Create a new file collectively and release property list identifier.
   file_id = H5Fcreate((fullpath.str() + ".h5").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
   H5Pclose(plist_id);

   // 3.All ranks need to create datasets dset*
   std::vector<unsigned int> Block_per_rank(size);
   MPI_Allgather(&Ngrids, 1, MPI_UNSIGNED, &Block_per_rank[0], 1, MPI_UNSIGNED, comm);
   for (int r = 0; r < size; r++)
   {
      hsize_t dims1[4] = {nZ, nY, nX, NCHANNELS * Block_per_rank[r]};
      fspace_id        = H5Screate_simple(4, dims1, NULL);
      std::stringstream name;
      name << "dset" << std::setfill('0') << std::setw(10) << r;
      dataset_id = H5Dcreate(file_id, (name.str()).c_str(), get_hdf5_type<hdf5Real>(), fspace_id,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dclose(dataset_id);
      H5Sclose(fspace_id);
   }

   // 4.Each rank now dumps its own blocks to the corresponding dset
   std::vector<hdf5Real> array_block(Block_per_rank[rank] * nX * nY * nZ * NCHANNELS, 0.0);
#if 1
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
#else
  for (unsigned int m = 0; m < Ngrids; m++)
  {
    B &block = *MyBlocks[m];
    for (unsigned int iz = 0; iz < nZ; iz++)
      for (unsigned int iy = 0; iy < nY; iy++)
        for (unsigned int ix = 0; ix < nX; ix++)
        {
          hdf5Real output[NCHANNELS];
          TStreamer::operate(block, ix, iy, iz, (hdf5Real *)output);
          const unsigned int idx_base = m * NCHANNELS + ix * (NCHANNELS*Ngrids) + 
                                                        iy * (NCHANNELS*Ngrids*nX) + 
                                                        iz * (NCHANNELS*Ngrids*nX*nY); 
          for (unsigned int j = 0; j < NCHANNELS; ++j)
            array_block[idx_base + j] = output[j];
        }
  }
#endif   
   std::stringstream name;
   name << "dset" << std::setfill('0') << std::setw(10) << rank;
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
         s << "    " << std::scientific << I.h << " " << I.h << " " << I.h << "\n";
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
      MPI_Offset offset = 0;
      MPI_Offset len    = st.size() * sizeof(char);

      MPI_File xmf;

      // delete the xmf file is it exists; no worries if it doesn't
      MPI_File_delete((fullpath.str() + ".xmf").c_str(), MPI_INFO_NULL);

      MPI_File_open(comm, (fullpath.str() + ".xmf").c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE,
                    MPI_INFO_NULL, &xmf);

      MPI_Exscan(&len, &offset, 1, MPI_OFFSET, MPI_SUM, comm);

      MPI_File_write_at_all(xmf, offset, st.data(), st.size(), MPI_CHAR, MPI_STATUS_IGNORE);

      MPI_File_close(&xmf);
   }
#else
   _warn_no_hdf5();
#endif
}
#endif

CUBISM_NAMESPACE_END
