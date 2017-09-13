/*
 *  P2M.h
 *  Cubism
 *
 *  Created by Ivica Kicic on 06/28/17.
 *  Copyright 2017 ETH Zurich. All rights reserved.
 *
 */
#ifndef _CUBISM_UTILS_P2M_H_
#define _CUBISM_UTILS_P2M_H_

#include "Process.h"

namespace cubism {
namespace utils {

/*
 * Particle to mesh algorithm.
 *
 * Update grid at the given locations. For each of the given locations
 * (particle positions), determine which grid points are affected and call
 * `update_func` with the grid point reference and the corresponding weight.
 *
 * Restrictions:
 *   - Assumes point coordinates between 0 and 1 (same as Cubism domain).
 *   - 1st order interpolation.
 *   - Only periodic boundaries.
 *   - Only single node (NO MPI!).
 *
 */
template <int DIM, typename Grid, typename Array, typename UpdateFunc>
void linear_p2m(Grid &grid, const Array &points, UpdateFunc update_func) {
    /* Based on Fabian's Cubism-LAMMPS example. */

    static_assert(DIM == 2 || DIM == 3, "Only 2 and 3 dimensions supported.");
    typedef typename Grid::BlockType Block;

    // BEGIN Particle storage.
    struct Particle {
        double pos[DIM];
        const typename Array::value_type *it;
    };

    const int N[3] = {
        grid.getBlocksPerDimension(0),
        grid.getBlocksPerDimension(1),
        grid.getBlocksPerDimension(2)
    };

    std::vector<std::vector<Particle>> particles(N[0] * N[1] * N[2]);

    const double h = grid.getBlocksInfo()[0].h_gridpoint;
    for (const auto &point : points) {
        const double pos[3] = {
            point[0],
            point[1],
            DIM == 3 ? point[2] : 0
        };
        const int min[3] = {
            int(pos[0] * N[0]),
            int(pos[1] * N[1]),
            int(pos[2] * N[2])
        };
        const int max[3] = {
            int(pos[0] * N[0] + h),
            int(pos[1] * N[1] + h),
            int(pos[2] * N[2] + h)
        };

        auto get_block_periodic = [&N](const int index[3], int *idx) {
            idx[0] = N[0]*(((index[2]+N[2])%N[2])*N[1] + ((index[1]+N[1])%N[1])) + ((index[0]+N[0])%N[0]);
            idx[1] = index[0] < 0 ? 1 : (index[0] < N[0] ? 0 : -1 );
            idx[2] = index[1] < 0 ? 1 : (index[1] < N[1] ? 0 : -1 );
            idx[3] = index[2] < 0 ? 1 : (index[2] < N[2] ? 0 : -1 );
            assert(idx[0] >= 0 && idx[0] < N[0] * N[1] * N[2]);
        };

        for (int iz = min[2]; iz <= max[2]; ++iz)
        for (int iy = min[1]; iy <= max[1]; ++iy)
        for (int ix = min[0]; ix <= max[0]; ++ix) {
            int idx[4];
            const int index[3] = {ix, iy, iz};
            get_block_periodic(index, idx);  // Periodic domain [0, 1].
            Particle p{{}, &point};
            for (int i = 0; i < DIM; ++i)
                p.pos[i] = pos[i] + idx[i + 1];
            particles[idx[0]].push_back(p);
        }
    }
    // END Particle storage.

    // BEGIN Linear P2M.
    auto rhs = [&particles, &update_func, &N](const BlockInfo &info,
                                              auto &block) {
        const int block_index = info.index[0] + N[0] * (
                                info.index[1] + N[1] * info.index[2]);
        const double invh = 1. / info.h_gridpoint;
        for (const auto &part : particles[block_index]) {
            double _idx[DIM];
            for (int i = 0; i < DIM; ++i)
                _idx[i] = invh * (part.pos[i] - info.origin[i]);

            int idx[DIM];
            for (int i = 0; i < DIM; ++i) {
                idx[i] = std::max(0, std::min((int)_idx[i],
                                              Block::sizeArray[i] - 1));
            }

            double alpha[2 * DIM];
            for (int i = 0; i < DIM; ++i) {
                alpha[2 * i] = 1 - (_idx[i] - idx[i]);
                alpha[2 * i + 1] = 1 - alpha[2 * i];
            }
            const int xmin = std::max(idx[0], 0);
            const int ymin = std::max(idx[1], 0);
            const int xmax = std::min(idx[0] + 1, block.sizeX - 1);
            const int ymax = std::min(idx[1] + 1, block.sizeY - 1);
            // TODO: Use C++17 if constexpr here.
            if (DIM == 2) {
                for (int iy = ymin; iy <= ymax; ++iy) {
                    const double ay = alpha[2 + iy - idx[1]];
                    for (int ix = xmin; ix <= xmax; ++ix) {
                        const double weight = ay * alpha[ix - idx[0]];
                        update_func(block(ix, iy), weight, *part.it);
                    }
                }
            } else if (DIM == 3) {
                const int zmin = std::min(idx[2], 0);
                const int zmax = std::min(idx[2] + 2, block.sizeZ - 1);
                for (int iz = zmin; iz <= zmax; ++iz) {
                    const double az = alpha[4 + iz - idx[2]];
                    for (int iy = ymin; iy <= ymax; ++iy) {
                        const double ayz = az * alpha[2 + iy - idx[1]];
                        for (int ix = xmin; ix <= xmax; ++ix) {
                            const double weight = ayz * alpha[ix - idx[0]];
                            update_func(block(ix, iy, iz), weight, *part.it);
                        }
                    }
                }
            }
        }
    };

    cubism::utils::process_pointwise(rhs, grid);
    // END Linear P2M.
}

}  // Namespace utils.
}  // Namespace cubism.

#endif
