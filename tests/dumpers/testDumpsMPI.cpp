// File       : testDumpsMPI.cpp
// Created    : Tue Jul 24 2018 02:05:12 PM (+0200)
// Author     : Fabian Wermelinger
// Description: Test Cubism dumping facilities
// Copyright 2018 ETH Zurich. All Rights Reserved.
#ifdef _DOUBLE_
using MyReal = double;
#else
using MyReal = float;
#define _FLOAT_PRECISION_
#endif /* _DOUBLE_ */

#define _BLOCKSIZE_ 16
#define _BLOCKSIZEX_ _BLOCKSIZE_
#define _BLOCKSIZEY_ _BLOCKSIZE_
#define _BLOCKSIZEZ_ _BLOCKSIZE_

#include <mpi.h>
#include <cstring>
#include <vector>
#include <string>
#include <sstream>
using namespace std;

#include "ArgumentParser.h"
#include "Grid.h"
#include "GridMPI.h"
#include "BlockInfo.h"

// dumpers
#ifndef _HDF5_DOUBLE_PRECISION_
const string prec_string = "4byte";
#else
const string prec_string = "8byte";
#endif

#define _USE_HDF_
#include "HDF5Dumper.h"
#include "HDF5Dumper_MPI.h"

#include "HDF5SliceDumper.h"
#include "HDF5SliceDumperMPI.h"

// #include "ZBinDumper.h" // requires CubismZ
// #include "ZBinDumper_MPI.h" // requires CubismZ

#include "PlainBinDumper_MPI.h"


template <typename TReal>
struct Block
{
    static const size_t sizeX = _BLOCKSIZE_;
    static const size_t sizeY = _BLOCKSIZE_;
    static const size_t sizeZ = _BLOCKSIZE_;

    typedef TReal ElementType;

    TReal data[_BLOCKSIZE_][_BLOCKSIZE_][_BLOCKSIZE_];
    inline void clear() { memset(&data[0][0][0], 0, _BLOCKSIZE_*_BLOCKSIZE_*_BLOCKSIZE_*sizeof(TReal)); }
};

struct MyStreamer
{
    static const int NCHANNELS = 1;

    template <typename TBlock, typename T>
    static inline void operate(const TBlock& b, const int ix, const int iy, const int iz, T output[NCHANNELS])
    {
        output[0] = b.data[iz][iy][ix];
    }

    static const char * getAttributeName() { return "Scalar"; }
};

using MyBlock   = Block<MyReal>;
using MyGrid    = Grid<MyBlock>;
using MyGridMPI = GridMPI<MyGrid>;
using MySlice   = typename SliceCreator::Slice<MyGrid>;
using MySliceMPI= typename SliceCreatorMPI::Slice<MyGridMPI>;
#ifdef _NONUNIFORM_
using MyMeshMap = MeshMap<MyBlock>;
using MyDensity = RandomDensity;
#endif /* _NONUNIFORM_ */


void set_grid_ic(MyGridMPI* grid, const int myrank=0)
{
    vector<BlockInfo> infos = grid->getResidentBlocksInfo();

    const int NX = grid->getResidentBlocksPerDimension(0);
    const int NY = grid->getResidentBlocksPerDimension(1);
    const int NZ = grid->getResidentBlocksPerDimension(2);

    const double blocksize = MyBlock::sizeX*MyBlock::sizeY*MyBlock::sizeZ;
    const double offset = myrank * NX*NY*NZ * blocksize;

#pragma omp parallel for
    for (size_t i = 0; i < infos.size(); ++i)
    {
        BlockInfo info = infos[i];
        MyBlock& b = *(MyBlock*)info.ptrBlock;
        for (size_t iz = 0; iz < MyBlock::sizeZ; ++iz)
            for (size_t iy = 0; iy < MyBlock::sizeY; ++iy)
                for (size_t ix = 0; ix < MyBlock::sizeX; ++ix)
                    b.data[iz][iy][ix] = offset + info.blockID*blocksize + ix + (double)MyBlock::sizeX*(iy + MyBlock::sizeY*iz);
    }
}


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    ArgumentParser parser(argc, argv);

    // blocks
    const int bpdx = parser("bpdx").asInt(1);
    const int bpdy = parser("bpdy").asInt(bpdx);
    const int bpdz = parser("bpdz").asInt(bpdx);

    // processes
    const int ppdx = parser("ppdx").asInt(1);
    const int ppdy = parser("ppdy").asInt(ppdx);
    const int ppdz = parser("ppdz").asInt(ppdx);

#ifdef _NONUNIFORM_
    MyDensity mesh_density;
    MyMeshMap* xmap = new MyMeshMap(0, 1, ppdx * bpdx);
    MyMeshMap* ymap = new MyMeshMap(0, 1, ppdy * bpdy);
    MyMeshMap* zmap = new MyMeshMap(0, 1, ppdz * bpdz);
    xmap->init(&mesh_density);
    ymap->init(&mesh_density);
    zmap->init(&mesh_density);
    MyGridMPI* grid = new MyGridMPI(xmap, ymap, zmap, ppdx, ppdy, ppdz, bpdx, bpdy, bpdz);
#else
    MyGridMPI* grid = new MyGridMPI(ppdx, ppdy, ppdz, bpdx, bpdy, bpdz);
#endif /* _NONUNIFORM_ */

    int myrank;
    const MPI_Comm comm = grid->getCartComm();
    MPI_Comm_rank(comm, &myrank);

    set_grid_ic(grid, myrank);

    ///////////////////////////////////////////////////////////////////////////
    // HDF5 full dumps
    {
        ostringstream fname;
        fname << "serial_rank" << myrank << "_hdf5_" << prec_string;
        DumpHDF5<MyGrid,MyStreamer>(*(MyGrid*)grid, 0, 0, fname.str());
    }

    {
        ostringstream fname;
        fname << "mpi_hdf5_" << prec_string;
        DumpHDF5_MPI<MyGridMPI,MyStreamer>(*grid, 0, 0, fname.str());
    }
    ///////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////
    // HDF5 slice dumps
    {
        // set defaults
        parser("nslices").asInt(3);
        parser("slice1_direction").asInt(0);
        parser("slice1_fraction").asDouble(0.3);
        parser("slice2_direction").asInt(1);
        parser("slice2_fraction").asDouble(0.6);
        parser("slice3_direction").asInt(2);
        parser("slice3_fraction").asDouble(0.8);

        // parser.print_args();

        vector<MySlice> slices = MySlice::getSlices<MySlice>(parser, *(MyGrid*)grid);
        vector<MySliceMPI> slices_mpi = MySliceMPI::getSlices<MySliceMPI>(parser, *grid);

        for (size_t i = 0; i < slices.size(); ++i)
        {
            const MySlice& slice = slices[i];

            ostringstream fname;
            fname << "serial_rank" << myrank << "_hdf5_slice" << (i+1) << "_" << prec_string;
            DumpSliceHDF5<MySlice,MyStreamer>(slice, 0, 0, fname.str());
        }

        for (size_t i = 0; i < slices_mpi.size(); ++i)
        {
            const MySliceMPI& slice = slices_mpi[i];

            ostringstream fname;
            fname << "mpi_hdf5_slice" << (i+1) << "_" << prec_string;
            DumpSliceHDF5MPI<MySliceMPI,MyStreamer>(slice, 0, 0, fname.str());
        }
    }
    ///////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////
    // Plain binary dumper (MPI only)
    {
        ostringstream fname;
        fname << "mpi_plain_bin_" << prec_string;

        BlockInfo info_front = grid->getBlocksInfo().front();
        Real* const src = (Real*)info_front.ptrBlock;

        const int NX = grid->getResidentBlocksPerDimension(0);
        const int NY = grid->getResidentBlocksPerDimension(1);
        const int NZ = grid->getResidentBlocksPerDimension(2);
        const size_t bytes = NX*NY*NZ*sizeof(MyBlock);

        PlainDumpBin_MPI(comm, src, bytes, fname.str());
    }

    MPI_Finalize();
    return 0;
}
