//
//  HDF5SliceDumper.h
//  Cubism
//
//  Created by Fabian Wermelinger 09/27/2016
//  Copyright 2016 ETH Zurich. All rights reserved.
//
#ifndef HDF5SLICEDUMPER_H_QI4Y9HO7
#define HDF5SLICEDUMPER_H_QI4Y9HO7

#include <cassert>
#include <iostream>
#include <vector>
#include <sstream>

#ifdef _USE_HDF_
#include <hdf5.h>
#endif

#ifndef _HDF5_DOUBLE_PRECISION_
#define HDF_REAL H5T_NATIVE_FLOAT
typedef  float hdf5Real;
#else
#define HDF_REAL H5T_NATIVE_DOUBLE
typedef double hdf5Real;
#endif

#include "BlockInfo.h"
#include "MeshMap.h"

///////////////////////////////////////////////////////////////////////////////
// helpers
namespace SliceTypes
{
    template <typename TGrid>
    struct Slice
    {
        typedef TGrid GridType;

        TGrid * grid;
        int id;
        int dir;
        int idx;
        int width, height;
        bool valid;
        Slice() : grid(NULL), id(-1), dir(-1), idx(-1), width(0), height(0), valid(false) {}

        template <typename TSlice>
        static std::vector<TSlice> getEntities(ArgumentParser& parser, TGrid& grid)
        {
            typedef typename TGrid::BlockType B;
            int Dim[3];
            Dim[0] = grid.getBlocksPerDimension(0)*B::sizeX;
            Dim[1] = grid.getBlocksPerDimension(1)*B::sizeY;
            Dim[2] = grid.getBlocksPerDimension(2)*B::sizeZ;

            std::vector<TSlice> slices(0);
            const size_t nSlices = parser("nslices").asInt(0);
            for (size_t i = 0; i < nSlices; ++i)
            {
                TSlice thisOne;
                thisOne.id = i+1;
                thisOne.grid = &grid;
                assert(thisOne.grid != NULL);

                std::ostringstream identifier;
                identifier << "slice" << i+1;
                // fetch direction
                const std::string sDir = identifier.str() + "_direction";
                if (parser.check(sDir)) thisOne.dir = parser(sDir).asInt(0);
                const bool bDirOK = (thisOne.dir >= 0 && thisOne.dir < 3);
                assert(bDirOK);

                // compute index
                const std::string sIndex = identifier.str() + "_index";
                const std::string sFrac  = identifier.str() + "_fraction";
                if (parser.check(sIndex)) thisOne.idx = parser(sIndex).asInt(0);
                else if (parser.check(sFrac))
                {
                    const double fraction = parser(sFrac).asDouble(0.0);
                    const int idx = static_cast<int>(Dim[thisOne.dir] * fraction);
                    thisOne.idx = (fraction == 1.0) ? Dim[thisOne.dir]-1 : idx;
                }
                const bool bIdxOK = (thisOne.idx >= 0 && thisOne.idx < Dim[thisOne.dir]);
                assert(bIdxOK);

                if (bDirOK && bIdxOK) thisOne.valid = true;
                else
                {
                    std::cerr << "Slice: WARNING: Ill defined slice \"" << identifier.str() << "\"... Skipping this one" << std::endl;
                    thisOne.valid = false;
                    slices.push_back(thisOne);
                    continue;
                }

                // define slice layout
                if (thisOne.dir == 0)
                {
                    thisOne.width  = Dim[2];
                    thisOne.height = Dim[1];
                }
                else if (thisOne.dir == 1)
                {
                    thisOne.width  = Dim[2];
                    thisOne.height = Dim[0];
                }
                else if (thisOne.dir == 2)
                {
                    thisOne.width  = Dim[0];
                    thisOne.height = Dim[1];
                }
                slices.push_back(thisOne);
            }
            return slices;
        }
    };
}


namespace SliceExtractor
{
    template <typename TBlock, typename TStreamer>
    void YZ(const int ix, const int width, std::vector<BlockInfo>& bInfo, hdf5Real * const data)
    {
        const unsigned int NCHANNELS = TStreamer::NCHANNELS;

#pragma omp parallel for
        for(int i = 0; i < (int)bInfo.size(); ++i)
        {
            BlockInfo& info = bInfo[i];
            const int idx[3] = {info.index[0], info.index[1], info.index[2]};
            TBlock& b = *(TBlock*)info.ptrBlock;

            for(unsigned int iz=0; iz<TBlock::sizeZ; ++iz)
                for(unsigned int iy=0; iy<TBlock::sizeY; ++iy)
                {
                    hdf5Real output[NCHANNELS];
                    for(unsigned int k=0; k<NCHANNELS; ++k)
                        output[k] = 0;

                    TStreamer::operate(b, ix, iy, iz, (hdf5Real*)output);

                    const unsigned int gy = idx[1]*TBlock::sizeY + iy;
                    const unsigned int gz = idx[2]*TBlock::sizeZ + iz;

                    hdf5Real * const ptr = data + NCHANNELS*(gz + width * gy);

                    for(unsigned int k=0; k<NCHANNELS; ++k)
                        ptr[k] = output[k];
                }
        }
    }

    template <typename TBlock, typename TStreamer>
    void XZ(const int iy, const int width, std::vector<BlockInfo>& bInfo, hdf5Real * const data)
    {
        const unsigned int NCHANNELS = TStreamer::NCHANNELS;

#pragma omp parallel for
        for(int i = 0; i < (int)bInfo.size(); ++i)
        {
            BlockInfo& info = bInfo[i];
            const int idx[3] = {info.index[0], info.index[1], info.index[2]};
            TBlock& b = *(TBlock*)info.ptrBlock;

            for(unsigned int iz=0; iz<TBlock::sizeZ; ++iz)
                for(unsigned int ix=0; ix<TBlock::sizeX; ++ix)
                {
                    hdf5Real output[NCHANNELS];
                    for(unsigned int k=0; k<NCHANNELS; ++k)
                        output[k] = 0;

                    TStreamer::operate(b, ix, iy, iz, (hdf5Real*)output);

                    const unsigned int gx = idx[0]*TBlock::sizeX + ix;
                    const unsigned int gz = idx[2]*TBlock::sizeZ + iz;

                    hdf5Real * const ptr = data + NCHANNELS*(gz + width * gx);

                    for(unsigned int k=0; k<NCHANNELS; ++k)
                        ptr[k] = output[k];
                }
        }
    }

    template <typename TBlock, typename TStreamer>
    void YX(const int iz, const int width, std::vector<BlockInfo>& bInfo, hdf5Real * const data)
    {
        const unsigned int NCHANNELS = TStreamer::NCHANNELS;

#pragma omp parallel for
        for(int i = 0; i < (int)bInfo.size(); ++i)
        {
            BlockInfo& info = bInfo[i];
            const int idx[3] = {info.index[0], info.index[1], info.index[2]};
            TBlock& b = *(TBlock*)info.ptrBlock;

            for(unsigned int iy=0; iy<TBlock::sizeY; ++iy)
                for(unsigned int ix=0; ix<TBlock::sizeX; ++ix)
                {
                    hdf5Real output[NCHANNELS];
                    for(unsigned int k=0; k<NCHANNELS; ++k)
                        output[k] = 0;

                    TStreamer::operate(b, ix, iy, iz, (hdf5Real*)output);

                    const unsigned int gx = idx[0]*TBlock::sizeX + ix;
                    const unsigned int gy = idx[1]*TBlock::sizeY + iy;

                    hdf5Real * const ptr = data + NCHANNELS*(gx + width * gy);

                    for(unsigned int k=0; k<NCHANNELS; ++k)
                        ptr[k] = output[k];
                }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// Dumpers
//
// The following requirements for the data Streamer are required:
// Streamer::NCHANNELS        : Number of data elements (1=Scalar, 3=Vector, 9=Tensor)
// Streamer::operate          : Data access methods for read and write
// Streamer::getAttributeName : Attribute name of the date ("Scalar", "Vector", "Tensor")

template<typename TSlice, typename TStreamer>
void DumpSliceHDF5(const TSlice& slice, const int stepID, const Real t, const std::string fname, const std::string dpath=".", const bool bXMF=true)
{
#ifdef _USE_HDF_
    typedef typename TSlice::GridType::BlockType B;
    const typename TSlice::GridType& grid = *(slice.grid);

    static const unsigned int NCHANNELS = TStreamer::NCHANNELS;
    const unsigned int width = slice.width;
    const unsigned int height = slice.height;

    std::cout << "Allocating " << (width * height * NCHANNELS * sizeof(hdf5Real))/(1024.*1024.) << " MB of HDF5 slice data" << std::endl;;

    herr_t status;
    hid_t file_id, dataset_id, fspace_id, fapl_id, mspace_id;

    ///////////////////////////////////////////////////////////////////////////
    // startup file
    // fname is the base filename without file type extension
    std::ostringstream filename;
    filename << dpath << "/" << fname << "_slice" << slice.id;
    H5open();
    fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    file_id = H5Fcreate((filename.str()+".h5").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
    status = H5Pclose(fapl_id); if(status<0) H5Eprint1(stdout);

    ///////////////////////////////////////////////////////////////////////////
    // write mesh
    std::vector<int> mesh_dims;
    std::vector<std::string> dset_name;
    dset_name.push_back("/vwidth");
    dset_name.push_back("/vheight");
    int slice_orientation[2];
    if (0 == slice.dir)
    {
        slice_orientation[0] = 2;
        slice_orientation[1] = 1;
    }
    else if (1 == slice.dir)
    {
        slice_orientation[0] = 2;
        slice_orientation[1] = 0;
    }
    else if (2 == slice.dir)
    {
        slice_orientation[0] = 0;
        slice_orientation[1] = 1;
    }

    for (size_t i = 0; i < 2; ++i)
    {
        const MeshMap<B>& m = grid.getMeshMap(slice_orientation[i]);
        std::vector<double> vertices(m.ncells()+1, m.start());
        mesh_dims.push_back(vertices.size());

        for (int j = 0; j < m.ncells(); ++j)
            vertices[j+1] = vertices[j] + m.cell_width(j);

        hsize_t dim[1] = {vertices.size()};
        fspace_id = H5Screate_simple(1, dim, NULL);
#ifndef _ON_FERMI_
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
    hdf5Real * array_all = new hdf5Real[width * height * NCHANNELS];

    std::vector<BlockInfo> bInfo_local = grid.getBlocksInfo();
    std::vector<BlockInfo> bInfo_slice;
    for (size_t i = 0; i < bInfo_local.size(); ++i)
    {
        const int start = bInfo_local[i].index[slice.dir] * _BLOCKSIZE_;
        if (start <= slice.idx && slice.idx < (start+_BLOCKSIZE_))
            bInfo_slice.push_back(bInfo_local[i]);
    }

    hsize_t count[3] = {height, width, NCHANNELS};
    hsize_t dims[3] = {height, width, NCHANNELS};
    hsize_t offset[3] = {0, 0, 0};

    if (0 == slice.dir)
        SliceExtractor::YZ<B,TStreamer>(slice.idx%_BLOCKSIZE_, width, bInfo_slice, array_all);
    else if (1 == slice.dir)
        SliceExtractor::XZ<B,TStreamer>(slice.idx%_BLOCKSIZE_, width, bInfo_slice, array_all);
    else if (2 == slice.dir)
        SliceExtractor::YX<B,TStreamer>(slice.idx%_BLOCKSIZE_, width, bInfo_slice, array_all);

    fapl_id = H5Pcreate(H5P_DATASET_XFER);
    fspace_id = H5Screate_simple(3, dims, NULL);
#ifndef _ON_FERMI_
    dataset_id = H5Dcreate(file_id, "data", HDF_REAL, fspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
#else
    dataset_id = H5Dcreate2(file_id, "data", HDF_REAL, fspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
#endif

    fspace_id = H5Dget_space(dataset_id);
    H5Sselect_hyperslab(fspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);
    mspace_id = H5Screate_simple(3, count, NULL);
    status = H5Dwrite(dataset_id, HDF_REAL, mspace_id, fspace_id, fapl_id, array_all); if(status<0) H5Eprint1(stdout);

    status = H5Sclose(mspace_id); if(status<0) H5Eprint1(stdout);
    status = H5Sclose(fspace_id); if(status<0) H5Eprint1(stdout);
    status = H5Dclose(dataset_id); if(status<0) H5Eprint1(stdout);
    status = H5Pclose(fapl_id); if(status<0) H5Eprint1(stdout);
    status = H5Fclose(file_id); if(status<0) H5Eprint1(stdout);
    H5close();

    delete [] array_all;

    // writing xmf wrapper
    if (bXMF)
    {
        FILE *xmf = 0;
        xmf = fopen((filename.str()+".xmf").c_str(), "w");
        fprintf(xmf, "<?xml version=\"1.0\" ?>\n");
        fprintf(xmf, "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n");
        fprintf(xmf, "<Xdmf Version=\"2.0\">\n");
        fprintf(xmf, " <Domain>\n");
        fprintf(xmf, "   <Grid GridType=\"Uniform\">\n");
        fprintf(xmf, "     <Time Value=\"%e\"/>\n\n", t);
        fprintf(xmf, "     <Topology TopologyType=\"2DRectMesh\" Dimensions=\"%d %d\"/>\n\n", mesh_dims[1], mesh_dims[0]);
        fprintf(xmf, "     <Geometry GeometryType=\"VxVyVz\">\n");
        fprintf(xmf, "       <DataItem Name=\"mesh_vx\" Dimensions=\"1\" NumberType=\"Float\" Precision=\"8\" Format=\"XML\">\n");
        fprintf(xmf, "        %e\n", 0.0);
        fprintf(xmf, "       </DataItem>\n");
        fprintf(xmf, "       <DataItem Name=\"mesh_vy\" Dimensions=\"%d\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n", mesh_dims[0]);
        fprintf(xmf, "        %s:/vwidth\n",(filename.str()+".h5").c_str());
        fprintf(xmf, "       </DataItem>\n");
        fprintf(xmf, "       <DataItem Name=\"mesh_vz\" Dimensions=\"%d\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n", mesh_dims[1]);
        fprintf(xmf, "        %s:/vheight\n",(filename.str()+".h5").c_str());
        fprintf(xmf, "       </DataItem>\n");
        fprintf(xmf, "     </Geometry>\n\n");
        fprintf(xmf, "     <Attribute Name=\"data\" AttributeType=\"%s\" Center=\"Cell\">\n", TStreamer::getAttributeName());
        fprintf(xmf, "       <DataItem Dimensions=\"%d %d %d\" NumberType=\"Float\" Precision=\"%d\" Format=\"HDF\">\n", height, width, NCHANNELS, sizeof(hdf5Real));
        fprintf(xmf, "        %s:/data\n",(filename.str()+".h5").c_str());
        fprintf(xmf, "       </DataItem>\n");
        fprintf(xmf, "     </Attribute>\n");
        fprintf(xmf, "   </Grid>\n");
        fprintf(xmf, " </Domain>\n");
        fprintf(xmf, "</Xdmf>\n");
        fclose(xmf);
    }

#else
#warning USE OF HDF WAS DISABLED AT COMPILE TIME
#endif
}

#endif /* HDF5SLICEDUMPER_H_QI4Y9HO7 */
