/*
 *  ZBinDumper.h
 *  Cubism
 *
 *  Created by Panos Hadjidoukas on 3/18/14.
 *  Copyright 2014 CSE Lab, ETH Zurich. All rights reserved.
 *
 */
#pragma once

#include <iostream>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>

#include "BlockInfo.h"
#include "LosslessCompression.h"

#define DBG 0

CUBISM_NAMESPACE_BEGIN

typedef struct _header_serial {
    long size[8];
} header_serial;

// The following requirements for the data TStreamer are required:
// TStreamer::NCHANNELS: Number of data elements (1=Scalar, 3=Vector, 9=Tensor)
// TStreamer::operate: Data access methods for read and write
// TStreamer::getAttributeName: Attribute name of the date ("Scalar", "Vector",
// "Tensor")
template <typename TStreamer, typename TGrid>
void DumpZBin(const TGrid &grid,
              const int iCounter,
              const typename TGrid::Real t,
              const std::string &f_name,
              const std::string &dump_path = ".",
              const bool bDummy = false)
{
    std::cout << "DumpZBin skipped.\n"; return;

    typedef typename TGrid::BlockType B;

    // f_name is the base filename without file type extension
    std::ostringstream filename;
    filename << dump_path << "/" << f_name;

    FILE *file_id;

    static const int NCHANNELS = TStreamer::NCHANNELS;
    const int NX = grid.getBlocksPerDimension(0) * B::sizeX;
    const int NY = grid.getBlocksPerDimension(1) * B::sizeY;
    const int NZ = grid.getBlocksPerDimension(2) * B::sizeZ;

    Real memsize = (NX * NY * NZ * sizeof(Real)) / (1024. * 1024. * 1024.);
    std::cout << "Allocating " << memsize << " GB of BIN data" << std::endl;
    Real *array_all = new Real[NX * NY * NZ];

    std::vector<BlockInfo> vInfo_local = grid.getBlocksInfo();

    static const int sX = 0;
    static const int sY = 0;
    static const int sZ = 0;

    static const int eX = B::sizeX;
    static const int eY = B::sizeY;
    static const int eZ = B::sizeZ;

    file_id = fopen((filename.str() + ".zbin").c_str(), "w");

    header_serial tag;
    fseek(file_id, sizeof(tag), SEEK_SET);
    for (int ichannel = 0; ichannel < NCHANNELS; ichannel++) {
#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(vInfo_local.size()); i++) {
            BlockInfo &info = vInfo_local[i];
            const int idx[3] = {info.index[0], info.index[1], info.index[2]};
            B &b = *(B *)info.ptrBlock;

            for (int ix = sX; ix < eX; ix++) {
                const int gx = idx[0] * B::sizeX + ix;
                for (int iy = sY; iy < eY; iy++) {
                    const int gy = idx[1] * B::sizeY + iy;
                    for (int iz = sZ; iz < eZ; iz++) {
                        const int gz = idx[2] * B::sizeZ + iz;

                        assert((gz + NZ * (gy + NY * gx)) < NX * NY * NZ);

                        Real *const ptr =
                            array_all + (gz + NZ * (gy + NY * gx));

                        Real output;
                        TStreamer::operate(b,
                                           ix,
                                           iy,
                                           iz,
                                           &output,
                                           ichannel); // point -> output,
                        ptr[0] = output;
                    }
                }
            }
        }

        //	size_t local_count = NX * NY * NZ * NCHANNELS;
        size_t local_count = NX * NY * NZ * 1;
        size_t local_bytes = local_count * sizeof(Real);

        size_t max = local_bytes;
        //	int layout[4] = {NCHANNELS, NX, NY, NZ};
        int layout[4] = {NX, NY, NZ, 1};
        long compressed_bytes =
            ZZcompress<typename TGrid::Real>((unsigned char *)array_all,
                                             local_bytes,
                                             layout,
                                             &max); // "in place"

        printf("Writing %ld bytes of Compressed data (cr = %.2f)\n",
               compressed_bytes,
               NX * NY * NZ * sizeof(Real) * NCHANNELS * 1.0 /
                   compressed_bytes);

        tag.size[ichannel] = compressed_bytes;
        fwrite(array_all, 1, compressed_bytes, file_id);
    }

    fseek(file_id, 0, SEEK_SET);
    fwrite(&tag.size[0], 1, sizeof(tag), file_id);

    fclose(file_id);

    delete[] array_all;
}

template <typename TStreamer, typename TGrid>
void ReadZBin(TGrid &grid,
              const std::string &f_name,
              const std::string &read_path = ".")
{
    std::cout << "ReadZBin skipped.\n"; return;
    typedef typename TGrid::BlockType B;
    typedef typename TGrid::Real Real;

    // f_name is the base filename without file type extension
    std::ostringstream filename;
    filename << read_path << "/" << f_name;

    FILE *file_id;

    const int NX = grid.getBlocksPerDimension(0) * B::sizeX;
    const int NY = grid.getBlocksPerDimension(1) * B::sizeY;
    const int NZ = grid.getBlocksPerDimension(2) * B::sizeZ;
    static const int NCHANNELS = TStreamer::NCHANNELS;

    Real *array_all = new Real[NX * NY * NZ * NCHANNELS];

    std::vector<BlockInfo> vInfo_local = grid.getBlocksInfo();

    static const int sX = 0;
    static const int sY = 0;
    static const int sZ = 0;

    const int eX = B::sizeX;
    const int eY = B::sizeY;
    const int eZ = B::sizeZ;

    file_id = fopen((filename.str() + ".zbin").c_str(), "rb");

    size_t local_count = NX * NY * NZ * 1;
    size_t local_bytes = local_count * sizeof(Real);

    header_serial tag;
    fread(&tag.size[0], 1, sizeof(tag), file_id);

#if DBG
    printf("HEADER(%d):\n", rank);
    for (int i = 0; i < NCHANNELS; i++) {
        printf("channel %d: %ld\n", i, tag.size[i]);
    }
#endif

    for (unsigned int ichannel = 0; ichannel < NCHANNELS; ichannel++) {
#if DBG
        printf("compr. size = %ld\n", tag.size[ichannel]);
        fflush(0);
#endif

        size_t compressed_bytes = tag.size[ichannel];
#if DBG
        printf("Reading %ld bytes of Compressed data (cr = %.2f)\n",
               compressed_bytes,
               local_bytes * 1.0 / compressed_bytes);
#endif
        unsigned char *tmp = (unsigned char *)malloc(compressed_bytes + 4096);

        fread(tmp, 1, compressed_bytes, file_id);

        int layout[4] = {NX, NY, NZ, 1};
#if DBG
        size_t decompressed_bytes =
#endif
            ZZdecompress<typename TGrid::Real>(tmp,
                                               compressed_bytes,
                                               layout,
                                               (unsigned char *)array_all,
                                               local_bytes);
        free(tmp);
#if DBG
        printf("size = %ld (%ld)\n", decompressed_bytes, local_bytes);
        fflush(0);
#endif

#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(vInfo_local.size()); i++) {
            BlockInfo &info = vInfo_local[i];
            const int idx[3] = {info.index[0], info.index[1], info.index[2]};
            B &b = *(B *)info.ptrBlock;

            for(int ix=sX; ix<eX; ix++)
                for(int iy=sY; iy<eY; iy++)
                    for(int iz=sZ; iz<eZ; iz++)
                    {
                        const int gx = idx[0]*B::sizeX + ix;
                        const int gy = idx[1]*B::sizeY + iy;
                        const int gz = idx[2]*B::sizeZ + iz;

                        Real * const ptr_input = array_all + (gz + NZ * (gy + NY * gx));

                        TStreamer::operate(b, *ptr_input, ix, iy, iz, ichannel);	// output -> point
                    }
        }
    } /* ichannel */

    fclose(file_id);
    delete [] array_all;
}

CUBISM_NAMESPACE_END
