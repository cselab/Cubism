/*
 *  ProcessMPI.h
 *  Cubism
 *
 *  Created by Ivica Kicic 07/06/2017
 *  Copyright 2017 ETH Zurich. All rights reserved.
 *
 */

/*
 * Original file: BlockProcessing_MPI.h (Cubism-MPCF)
 * Created by Fabian Wermelinger 10/13/2016.
 */

#ifndef _CUBISM_PROCESS_MPI_H_
#define _CUBISM_PROCESS_MPI_H_

#include <mpi.h>
#include "../source/StencilInfo.h"
#include "../source/SynchronizerMPI.h"

namespace cubism {
namespace utils {

/*
 * Process a kernel that requires a stencil.
 *
 * Arguments:
 *     kernel - Functor of type (const Lab &, const BlockInfo &, Block &)
 *              that also defines a `.stencil` of type StencilInfo.
 *     grid   - Grid reference.
 *     t      - Argument passed to Lab::load. 0.0 by default. (?).
 *
 * Runs kernel(...) for each block, given a block reference, block info
 * reference and a lab (containing the ghost cells).
 */
template<typename TLab, typename TKernel, typename TGrid>
inline void process_stencil_MPI(TKernel kernel,
                                TGrid& grid,
                                const Real t = 0.0)
{
    typedef typename TGrid::Block Block;
    SynchronizerMPI& Synch = grid.sync(kernel);

    const int nthreads = omp_get_max_threads();

    TLab * const labs = new TLab[nthreads];
    for (int i = 0; i < nthreads; ++i)
        labs[i].prepare(grid, Synch);

    static int rounds = -1;
    static int one_less = 1;
    if (rounds == -1)
    {
        const char * const s = getenv("MYROUNDS");
        if (s != NULL)
            rounds = atoi(s);
        else
            rounds = 0;

        const char * const s2 = getenv("USEMAXTHREADS");
        if (s2 != NULL)
            one_less = !atoi(s2);
    }

    MPI_Barrier(grid.getCartComm());


    std::vector<BlockInfo> avail0 = Synch.avail_inner();
    const int Ninner = (int)avail0.size();
    BlockInfo * const ary0 = avail0.data();

    int nthreads_first = nthreads - one_less;
    if (nthreads_first == 0) nthreads_first = 1;

    int Ninner_first = (nthreads_first)*rounds;
    if (Ninner_first > Ninner) Ninner_first = Ninner;
    int Ninner_rest = Ninner - Ninner_first;

#pragma omp parallel num_threads(nthreads_first)
    {
        int tid = omp_get_thread_num();
        TLab& mylab = labs[tid];

#pragma omp for schedule(dynamic,1)
        for(int i=0; i<Ninner_first; i++)
        {
            mylab.load(ary0[i], t);
            kernel(mylab, ary0[i], *(Block *)ary0[i].ptrBlock);
        }
    }

    std::vector<BlockInfo> avail1 = Synch.avail_halo();
    const int Nhalo = (int)avail1.size();
    BlockInfo * const ary1 = avail1.data();

#pragma omp parallel num_threads(nthreads)
    {
        const int tid = omp_get_thread_num();
        TLab& mylab = labs[tid];

#pragma omp for schedule(dynamic,1)
        for(int i=-Ninner_rest; i<Nhalo; i++)
        {
            if (i < 0)
            {
                const int ii = i + Ninner;
                mylab.load(ary0[ii], t);
                kernel(mylab, ary0[ii], *(Block *)ary0[ii].ptrBlock);
            }
            else
            {
                mylab.load(ary1[i], t);
                kernel(mylab, ary1[i], *(Block *)ary1[i].ptrBlock);
            }
        }
    }

    delete[] labs;

    MPI_Barrier(grid.getCartComm());
}


/*
 * Wrapper for `process_stencil_MPI` below.
 *
 * Instead of taking as a parameter a class defining operator() and .stencil,
 * here we take these two separately and construct the class automatically.
 *
 */

template <typename TLab, typename TKernel, typename TBlock>
struct __Kernel {
    /* Helper struct for process_stencil_MPI(...) below. */
    StencilInfo stencil;
    TKernel kernel;

    inline void operator()(const TLab &lab,
                           const BlockInfo &info,
                           TBlock &block)
    {
        kernel(lab, info, block);
    }
};

template<typename TLab, typename TKernel, typename TGrid>
inline void process_stencil_MPI(const StencilInfo &stencil,
                                TKernel kernel,
                                TGrid& grid,
                                const Real t = 0.0)
{
    typedef typename TGrid::Block Block;
    __Kernel<TLab, TKernel, Block> __kernel{stencil, kernel};
    process_stencil_MPI<TLab>(__kernel, grid, t);
}

}  // Namespace utils.
}  // Namespace cubism.

#endif
