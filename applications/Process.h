/*
 *  Block.h
 *  Cubism
 *
 *  Created by Ivica Kicic on 07/28/17.
 *  Copyright 2017 ETH Zurich. All rights reserved.
 *
 */

#ifndef _CUBISM_INTERNALS_H_
#define _CUBISM_INTERNALS_H_

#include <omp.h>

#include "../source/BlockInfo.h"
#include "../source/StencilInfo.h"

namespace cubism::applications {

/*
 * Process each point separately, or in technical words, each block separately.
 *
 * Arguments:
 *    rhs - Functor of type (const BlockInfo &, Block &) called for each block.
 *    grid - Reference to the grid.
 *
 * As opposed to `process_stencil`, `process_pointwise` does not require
 * BlockLab, i.e. ghost cells.
 */
template<typename Operator, typename Grid>
inline void process_pointwise(Operator rhs, Grid &grid) {
    typedef typename Grid::BlockType Block;
    const int nthreads = omp_get_max_threads();
    std::vector<BlockInfo> &blocks = grid.getBlocksInfo(); // block metadata

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        Operator myrhs = rhs;

        // Applies the kernel "myrhs" given the BlockInfo and pointer to
        // first data element in block i.
        #pragma omp for schedule(dynamic,1)
        for (size_t i = 0; i < blocks.size(); ++i)
            myrhs(blocks[i], *(Block *)blocks[i].ptrBlock);
    }
}


/*
 * Processes a kernel that requires a stencil.
 *
 * TODO: Argument description.
 *
 * Here, a BlockLab type is used, which is a copy of the actual data stored in
 * a block. The BlockLab defines ghost cells, based on the information in
 * StencilInfo, which defines the width of the stencil that is used for the
 * kernel.
 */
template<typename Lab, typename Operator, typename Grid>
inline void process_stencil(const StencilInfo &stencil,
                            Operator rhs,
                            Grid &grid,
                            const Real t = 0) {
    typedef typename Grid::BlockType Block;
    const int nthreads = omp_get_max_threads();
    std::vector<BlockInfo> &block_infos = grid.getBlocksInfo();

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();

        // Each thread works with its own BlockLab.
        Lab mylab;
        Operator myrhs = rhs;

        // Allocated memory for the lab based on the stencil.
        mylab.prepare(grid,
                      stencil.sx,  stencil.ex,
                      stencil.sy,  stencil.ey,
                      stencil.sz,  stencil.ez,
                      stencil.tensorial);

        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < block_infos.size(); ++i) {
            // copy the data from the grid block into the BlockLab.
            mylab.load(block_infos[i], t);

            // Apply the kernel using the BlockLab. Data is written back
            // into the block stored in the grid.
            myrhs(*(Block *)block_infos[i].ptrBlock, mylab, block_infos[i]);
        }
    }
}

}  // Namespace cubism.

#endif
