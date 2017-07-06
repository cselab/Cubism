/*
 *  Block.h
 *  Cubism
 *
 *  Created by Ivica Kicic on 06/28/17.
 *  Copyright 2017 ETH Zurich. All rights reserved.
 *
 */

/*
 * Block processing utility functions for single-node MPI.
 *     process_pointwise - Process each block separately (can be used for
 *                         MPI-Cubism too).
 *     process_stencil   - Process a kernel with a stencil.
 */

#ifndef _CUBISM_PROCESS_SINGLE_NODE_H_
#define _CUBISM_PROCESS_SINGLE_NODE_H_

#include <omp.h>

#include "../source/BlockInfo.h"
#include "../source/StencilInfo.h"

namespace cubism::utils
{

/*
 * Process each point separately, or in technical words, each block separately.
 *
 * Arguments:
 *     kernel - Functor of type (const BlockInfo &, Block &) invoked for
 *              each block.
 *     grid   - Reference to the grid.
 *
 * As opposed to `process_stencil`, `process_pointwise` does not require
 * BlockLab, i.e. ghost cells.
 */
template<typename Kernel, typename Grid>
inline void process_pointwise(Kernel kernel, Grid &grid)
{
    typedef typename Grid::BlockType Block;
    const int nthreads = omp_get_max_threads();
    std::vector<BlockInfo> &infos = grid.getBlocksInfo(); // block metadata

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        Kernel mykernel = kernel;

        // Applies the kernel "mykernel" given the BlockInfo and pointer to
        // first data element in block i.
        #pragma omp for schedule(dynamic,1)
        for (size_t i = 0; i < infos.size(); ++i)
            mykernel(infos[i], *(Block *)infos[i].ptrBlock);
    }
}


/*
 * Process a kernel that requires a stencil.
 *
 * Arguments:
 *     stencil - Stencil information.
 *     kernel  - Functor of type (const Lab &, const BlockInfo &, Block &)
 *               invoked for each block.
 *     grid    - Grid reference.
 *     t       - Argument passed to Lab::load. 0.0 by default. (?).
 *
 * Here, a BlockLab type is used, which is a copy of the actual data stored in
 * a block. The BlockLab defines ghost cells, based on the information in
 * StencilInfo, which defines the width of the stencil that is used for the
 * kernel.
 */
template<typename Lab, typename Kernel, typename Grid>
inline void process_stencil(const StencilInfo &stencil,
                            Kernel kernel,
                            Grid &grid,
                            const Real t = 0)
{
    typedef typename Grid::BlockType Block;
    const int nthreads = omp_get_max_threads();
    std::vector<BlockInfo> &infos = grid.getBlocksInfo();

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();

        // Each thread works with its own BlockLab.
        Lab mylab;
        Kernel mykernel = kernel;

        // Allocated memory for the lab based on the stencil.
        mylab.prepare(grid,
                      stencil.sx,  stencil.ex,
                      stencil.sy,  stencil.ey,
                      stencil.sz,  stencil.ez,
                      stencil.tensorial);

        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < infos.size(); ++i)
        {
            // copy the data from the grid block into the BlockLab.
            mylab.load(infos[i], t);

            // Apply the kernel using the BlockLab. Data is written back
            // into the block stored in the grid.
            mykernel(mylab, infos[i], *(Block *)infos[i].ptrBlock);
        }
    }
}

}  // Namespace cubism.

#endif
