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

#ifdef USE_VTK
 #include <vtkCell.h>
 #include <vtkCellData.h>
 #include <vtkDataArray.h>
 #include <vtkDoubleArray.h>
 #include <vtkNonOverlappingAMR.h>
 #include <vtkUniformGrid.h>
 #include <vtkXMLUniformGridAMRWriter.h>
 #include <vtkXMLPUniformGridAMRWriter.h>
 #include <vtkFloatArray.h>
 #include <vtkUnsignedCharArray.h>
 #include <vtkIntArray.h>
 #include <vtkAMRBox.h>
 #include <vtkOverlappingAMR.h>
#endif

CUBISM_NAMESPACE_BEGIN

struct StencilInfoWrapper
{
    StencilInfo stencil;
    StencilInfoWrapper(int g=1)
    {
        stencil.sx = -g;
        stencil.sy = -g;
        stencil.sz = -g;
        stencil.ex = +g+1;
        stencil.ey = +g+1;
        stencil.ez = +g+1;
        stencil.selcomponents.push_back(0);
        stencil.selcomponents.push_back(1);
        stencil.selcomponents.push_back(2);
        stencil.selcomponents.push_back(3);
        stencil.selcomponents.push_back(4);
        stencil.selcomponents.push_back(5);
        stencil.selcomponents.push_back(6);
        stencil.selcomponents.push_back(7);
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
    static const int nX = B::sizeX;
    static const int nY = B::sizeY;
    static const int nZ = B::sizeZ;

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

    #ifdef USE_VTK //VTK AMR-dataset

    const int numberOfLevels = grid.getlevelMax();

    std::vector<int> blocksPerLevel (numberOfLevels);
    std::vector<int> blocksPerLevel1(numberOfLevels);
    for (size_t m = 0; m < MyGroups.size(); m++)
        blocksPerLevel[MyGroups[m].level] ++;
    for (int l = numberOfLevels-1 ; l >=0 ; l--)
    {
      blocksPerLevel1[l] += blocksPerLevel[l];
      if (l>0) blocksPerLevel1[l-1] += blocksPerLevel1[l];
    }

    std::vector<int> TotalblocksPerLevel (numberOfLevels);
    std::vector<int> TotalblocksPerLevel1(numberOfLevels);
    MPI_Allreduce(blocksPerLevel.data(), TotalblocksPerLevel.data(), numberOfLevels, MPI_INT, MPI_SUM,comm);
    MPI_Allreduce(blocksPerLevel1.data(), TotalblocksPerLevel1.data(), numberOfLevels, MPI_INT, MPI_SUM,comm);

    std::vector<int> BaseblocksPerLevel(numberOfLevels);  
    std::vector<int> BaseblocksPerLevel1(numberOfLevels);  
    MPI_Exscan(blocksPerLevel.data(), BaseblocksPerLevel.data(), numberOfLevels, MPI_INT,  MPI_SUM , comm);
    MPI_Exscan(blocksPerLevel1.data(), BaseblocksPerLevel1.data(), numberOfLevels, MPI_INT,  MPI_SUM , comm);

    std::vector<int> CountblocksPerLevel(numberOfLevels,0);  

    vtkNonOverlappingAMR* amrGrid = vtkNonOverlappingAMR::New();
    amrGrid->Initialize(numberOfLevels, TotalblocksPerLevel.data());

    const int nGhosts = 2;
    const StencilInfoWrapper p(nGhosts > 0 ? nGhosts : 1);
    cubism::SynchronizerMPI_AMR<Real,TGrid>& Synch = *grid.sync(p);
    LabMPI lab;
    lab.prepare(grid, Synch);
    std::vector<cubism::BlockInfo*> avail0 = Synch.avail_inner();
    std::vector<cubism::BlockInfo*> avail1 = Synch.avail_halo();

    const double globalOrigin [3] ={0,0,0};
    for (unsigned int m = 0; m < MyGroups.size(); m++)
    {
        BlockGroup & info = MyGroups[m];
        
        const double spacing[3] ={info.h,info.h,info.h};
        double spacing1[3]      ={info.h,info.h,info.h};
        const int dimensions[3] ={info.NXX+2*nGhosts,info.NYY+2*nGhosts,info.NZZ+2*nGhosts};
        
        double origin [3] = {info.origin[0]-nGhosts*info.h,
                             info.origin[1]-nGhosts*info.h,
                             info.origin[2]-nGhosts*info.h}; 

        const vtkAMRBox myBox(origin,dimensions,spacing,globalOrigin);

        vtkUniformGrid* g = vtkUniformGrid::New();
        if (nGhosts == 0)
           g->Initialize(&myBox,origin,spacing1);
        else
           g->Initialize(&myBox,origin,spacing1,nGhosts);

        vtkSmartPointer<vtkFloatArray> xyz = vtkSmartPointer<vtkFloatArray>::New();
        xyz->SetName("data");
        //xyz->SetNumberOfComponents(NCHANNELS);
        xyz->SetNumberOfComponents(1);
        xyz->SetNumberOfTuples(g->GetNumberOfCells());

        const unsigned int nX_max = info.NXX-1;
        const unsigned int nY_max = info.NYY-1;
        for (int kB = info.i_min[2]; kB <= info.i_max[2]; kB++)
        for (int jB = info.i_min[1]; jB <= info.i_max[1]; jB++)
        for (int iB = info.i_min[0]; iB <= info.i_max[0]; iB++)
        {
            int Z = BlockInfo::forward(info.level,iB,jB,kB);
            const cubism::BlockInfo& I = grid.getBlockInfoAll(info.level,Z);
            lab.load(I, 0);
            for (int iz = 0 - nGhosts; iz < nZ + nGhosts; iz++)
            for (int iy = 0 - nGhosts; iy < nY + nGhosts; iy++)
            for (int ix = 0 - nGhosts; ix < nX + nGhosts; ix++)
            {
                float output[NCHANNELS];
                TStreamer::operate(lab,ix,iy,iz,output);
                const int base = ( ((kB-info.i_min[2])*nZ + iz)*((nX_max+2*nGhosts)*(nY_max+2*nGhosts)) + 
                                   ((jB-info.i_min[1])*nY + iy)*((nX_max+2*nGhosts)) + 
                                   ((iB-info.i_min[0])*nX + ix) )
                                   +nGhosts*(nX_max+2*nGhosts)*(nY_max+2*nGhosts)
                                   +nGhosts*(nX_max+2*nGhosts)
                                   +nGhosts*1;
                if (NCHANNELS > 1)
                    output[0] = sqrt(output[0]*output[0]+output[1]*output[1]+output[2]*output[2]);
                xyz->SetTuple(base, output);
            }
        }
        g->GetCellData()->AddArray(xyz);
        amrGrid->SetDataSet(info.level, BaseblocksPerLevel[info.level] + CountblocksPerLevel[info.level], g);
        CountblocksPerLevel[info.level]++;
        g->Delete();
    }

    //Write data
    std::string FNAME = (fullpath.str() + ".vthb").c_str();
    auto writer = vtkSmartPointer<vtkXMLPUniformGridAMRWriter>::New();
    writer->SetFileName(FNAME.c_str());
    writer->SetInputData(amrGrid);
    //writer->SetWriteMetaFile(false);
    writer->Write();

    //Overlapping AMR 
    {
        std::vector<int> CountblocksPerLevel1(numberOfLevels,0);
        std::vector<int> CountblocksPerLevel_new(numberOfLevels,0);
        std::stringstream s;
        if (rank == 0)
        {
            s << "<?xml version=\"1.0\"?>\n";
            s << "<VTKFile type=\"vtkOverlappingAMR\" version=\"1.1\" byte_order=\"LittleEndian\" header_type=\"UInt32\" compressor=\"vtkZLibDataCompressor\">\n";
            s << "  <vtkOverlappingAMR origin=\"0 0 0\" grid_description=\"XYZ\"> " << "\n";
            s << "  <Time Value=\"" << std::scientific << absTime << "\"/>\n\n";
        }
        for (unsigned int m = 0; m < MyGroups.size(); m++)
        {
            BlockGroup & I = MyGroups[m];

            //Add box from current level
            int nID = BaseblocksPerLevel[I.level] + CountblocksPerLevel1[I.level];
            int nID_new = BaseblocksPerLevel1[I.level] + CountblocksPerLevel_new[I.level];
            CountblocksPerLevel1[I.level]++;
            CountblocksPerLevel_new[I.level]++;
            for (int l = 0 ; l < I.level ; l++)
            {
                nID += TotalblocksPerLevel[l];
            }
            int s0 = (I.origin[0] + 0.5*I.h)/I.h;
            int s1 = (I.origin[1] + 0.5*I.h)/I.h;
            int s2 = (I.origin[2] + 0.5*I.h)/I.h;
            int e0 = s0 + I.NXX - 1 -1;
            int e1 = s1 + I.NYY - 1 -1;
            int e2 = s2 + I.NZZ - 1 -1;
            s <<"     <Block level=\"" + std::to_string(I.level) +  "\" spacing=\"" << std::scientific << I.h << " " << I.h << " " << I.h << "\" > \n";
            s <<"       <DataSet index=\""+ std::to_string(nID_new) + 
                "\" amr_box=\"" << s0 << " " << e0 <<  " " << s1 << " " << e1 << " " << s2 << " " << e2 << "\" file=\"" + filename.str() + "/" +filename.str() + "_" + std::to_string(nID)+ ".vti\"/>\n";
            s <<"     </Block>\n";

            //Add boxes from lower levels
            int aux = 1 << I.level;
            for (int l = 0 ; l <= I.level - 1; l++)
            {
                nID_new = BaseblocksPerLevel1[l] + CountblocksPerLevel_new[l];
                CountblocksPerLevel_new[l]++;
                s0 = ( (I.origin[0] + 0.5*I.h)/I.h ) / aux;
                s1 = ( (I.origin[1] + 0.5*I.h)/I.h ) / aux;
                s2 = ( (I.origin[2] + 0.5*I.h)/I.h ) / aux;
                e0 = s0 + (I.NXX-1)/aux - 1;
                e1 = s1 + (I.NYY-1)/aux - 1;
                e2 = s2 + (I.NZZ-1)/aux - 1;
                s <<"     <Block level=\"" + std::to_string(l) +  "\" spacing=\"" << std::scientific << I.h*aux << " " << I.h*aux << " " << I.h*aux << "\" > \n";
                s <<"       <DataSet index=\""+ std::to_string(nID_new) + "\" amr_box=\"" << s0 << " " << e0 <<  " " << s1 << " " << e1 << " " << s2 << " " << e2 << "\" />\n";
                s <<"     </Block>\n";
                aux /=2;
            }
        }
        if (rank == size - 1)
        {
           s << "   </vtkOverlappingAMR>\n";
           s << " </VTKFile>\n";
        }
        std::string st    = s.str();
        MPI_Offset offset = 0;
        MPI_Offset len    = st.size() * sizeof(char);
        MPI_File xmf;
        MPI_File_delete((fullpath.str() + "_over.vthb").c_str(), MPI_INFO_NULL);// delete the xmf file is it exists; no worries if it doesn't
        MPI_File_open(comm, (fullpath.str() + "_over.vthb").c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE,MPI_INFO_NULL, &xmf);
        MPI_Exscan(&len, &offset, 1, MPI_OFFSET, MPI_SUM, comm);
        MPI_File_write_at_all(xmf, offset, st.data(), st.size(), MPI_CHAR, MPI_STATUS_IGNORE);
        MPI_File_close(&xmf);
    }
    //Non-overlapping AMR
    {
        std::vector<int> CountblocksPerLevel2(numberOfLevels,0);
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
            int nID = BaseblocksPerLevel[I.level] + CountblocksPerLevel2[I.level];
            CountblocksPerLevel2[I.level]++;
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
    //Save mesh
    {
        std::stringstream s;
        if (rank == 0)
        {
           s << "<?xml version=\"1.0\" ?>\n";
           s << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
           s << "<Xdmf Version=\"2.0\">\n";
           s << "<Domain>\n";
           s << " <Grid Name=\"OctTree\" GridType=\"Collection\">\n";
           //s << "  <Time Value=\"" << std::scientific << absTime << "\"/>\n\n";
        }
        for (unsigned int m = 0; m < MyGroups.size(); m++)
        {
           BlockGroup & I = MyGroups[m];
           s << "  <Grid GridType=\"Uniform\">\n";
           s << "   <Topology TopologyType=\"3DCoRectMesh\" Dimensions=\" " << (I.NZZ-1)/nZ + 1 << " " << (I.NYY-1)/nY + 1 << " " << (I.NXX-1)/nX + 1<< "\"/>\n";
           s << "   <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n";
           s << "   <DataItem Dimensions=\"3\" NumberType=\"Double\" Precision=\"8\" " "Format=\"XML\">\n";
           s << "    " << std::scientific << I.origin[2] << " " << I.origin[1] << " " << I.origin[0] << "\n";
           s << "   </DataItem>\n";
           s << "   <DataItem Dimensions=\"3\" NumberType=\"Double\" Precision=\"8\" " "Format=\"XML\">\n";
           s << "    " << std::scientific << I.h*nZ << " " << I.h*nY << " " << I.h*nX << "\n";
           s << "   </DataItem>\n";
           s << "   </Geometry>\n";
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
        MPI_File_delete((fullpath.str() + "_grid.xmf").c_str(), MPI_INFO_NULL);
        MPI_File_open(comm, (fullpath.str() + "_grid.xmf").c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &xmf);
        MPI_Exscan(&len, &offset, 1, MPI_OFFSET, MPI_SUM, comm);
        MPI_File_write_at_all(xmf, offset, st.data(), st.size(), MPI_CHAR, MPI_STATUS_IGNORE);
        MPI_File_close(&xmf);
    }

    #else //HDF5

    // 2. Write grid meta-data
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
          s << "   <DataItem ItemType=\"Uniform\"  Dimensions=\" " << nZZ - 1 << " " << nYY - 1<< " " << nXX - 1<< " " << NCHANNELS << " " << "\" NumberType=\"Float\" Precision=\" " << (int)sizeof(hdf5Real) << "\" Format=\"HDF\">\n";
          
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
    H5open();
    hid_t file_id, dataset_id, fspace_id, plist_id;

    // 3a.Set up file access property list with parallel I/O access
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL);

    // 3b.Create a new file collectively and release property list identifier.
    file_id = H5Fcreate((fullpath.str() + ".h5").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    H5Pclose(plist_id);

    // 3c.All ranks need to create datasets dset*
    std::vector<unsigned int> & Groups_per_rank = grid.Groups_per_rank;
    std::vector<unsigned int> & allN            = grid.allN           ;

    int Npos = 0;
    for (int r = 0; r < size; r++)
    for (size_t groupID = 0 ; groupID < Groups_per_rank[r] ; groupID ++)
    {
        hsize_t cdims[4] = {8,8,8,1};
        hid_t plist_id1 = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(plist_id1, 4, cdims);        
        hsize_t dims1[4] = {allN[Npos+2] - 1, allN[Npos+1] - 1, allN[Npos] - 1, NCHANNELS};
        Npos += 3;
        fspace_id        = H5Screate_simple(4, dims1, NULL);
        std::stringstream name;
        name << "dset" << std::setfill('0') << std::setw(10) << r;
        name << "_"    << std::setfill('0') << std::setw(10) << groupID;
        dataset_id = H5Dcreate(file_id, (name.str()).c_str(), get_hdf5_type<hdf5Real>(), fspace_id, H5P_DEFAULT, plist_id1, H5P_DEFAULT);
        H5Dclose(dataset_id);
        H5Sclose(fspace_id);
        H5Pclose(plist_id1);
    }

    // 3d.Each rank now dumps its own blocks to the corresponding dset
    for (size_t groupID = 0 ; groupID < MyGroups.size() ; groupID ++)
    {
        std::stringstream name;
        name << "dset" << std::setfill('0') << std::setw(10) << rank;
        name << "_"    << std::setfill('0') << std::setw(10) << groupID;
        const BlockGroup & group = MyGroups[groupID];
        const int nX_max = group.NXX-1;
        const int nY_max = group.NYY-1;
        const int nZ_max = group.NZZ-1;
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
        H5Dclose(dataset_id);         
    }
    // 4.Close hdf5 file
    H5Fclose(file_id);
    H5close();

    #endif
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