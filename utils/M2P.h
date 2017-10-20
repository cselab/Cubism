/*
 *  M2P.h
 *  Cubism
 *
 *  Created by Ivica Kicic on 07/05/17.
 *  Copyright 2017 ETH Zurich. All rights reserved.
 *
 */
#ifndef _CUBISM_UTILS_M2P_H_
#define _CUBISM_UTILS_M2P_H_

#include <vector>

namespace cubism {
namespace utils {

/*
 * Mesh to particle (interpolation) algorithm.
 *
 * For each given point (particle position), interpolate the value of the field.
 *
 * Mandatory template arguments:
 *   - DIM - Dimensionality of the grid (only 2 and 3 supported).
 *
 * Arguments:
 *   - block_processing - Block processing functor.
 *   - grid   - Reference to the grid.
 *   - points - Array of the points. (*)
 *   - getter - Lambda of a single argument (BlockLab), returning the value
 *              to be interpolated (**).
 *   - setter - Lambda of two arguments (point ID, interpolated value).
 *
 * (*) Points should support operator [] for accessing x, y, z coordinates.
 * (**) Returned value type X must support operators <double> * X and X + X.
 */
template <int DIM,  // <-- Must be specified manually.
          typename BlockProcessing,
          typename Grid,
          typename Array,
          typename Getter,
          typename Setter>
void linear_m2p(BlockProcessing block_processing,
                Grid &grid,
                const Array &points,
                Getter getter,
                Setter setter) {
    /* Based on Fabian's Cubism-LAMMPS example. */

    static_assert(DIM == 2 || DIM == 3);
    typedef typename Grid::BlockType Block;

    // BEGIN Particle storage.
    struct Particle {
        int id;
        double pos[DIM];
    };

    const int N[3] = {
        grid.getBlocksPerDimension(0),
        grid.getBlocksPerDimension(1),
        grid.getBlocksPerDimension(2)
    };

    std::vector<std::vector<Particle>> particles(N[0] * N[1] * N[2]);

    // Map particles to CUBISM domain [0, 1] and put them in different blocks.
    for (decltype(points.size()) i = 0; i < points.size(); ++i) {
        const auto &point = points[i];
        // Map position to [0, 1].
        const double pos[3] = {
            point[0],
            point[1],
            DIM == 3 ? point[2] : 0
        };
        // Find block.
        const int index[3] = {
            std::max(std::min(int(pos[0] * N[0]), N[0] - 1), 0),
            std::max(std::min(int(pos[1] * N[1]), N[1] - 1), 0),
            std::max(std::min(int(pos[2] * N[2]), N[2] - 1), 0)
        };
        const int idx = index[0] + N[0] * (index[1] + N[1] * index[2]);
        Particle part;
        part.id = i;
        part.pos[0] = pos[0];
        part.pos[1] = pos[1];
        if (DIM == 3) part.pos[2] = pos[2];
        particles[idx].push_back(part);
    }
    // END Particle storage.

    // BEGIN Linear P2M.
    const auto rhs = [&particles, &N, &getter, &setter](
            const auto &lab,
            const BlockInfo &info,
            Block &o) {
        // Extract relevant block info.
        const int block_index = info.index[0] + N[0] * (
                                info.index[1] + N[1] * info.index[2]);
        const double invh = 1.0 / info.h_gridpoint;

        // Loop over all particles.
        for (const auto &part : particles[block_index]) {
            // Get position in index space within block.
            double ipos[DIM];
            for (int i = 0; i < DIM; ++i)
                ipos[i] = invh * (part.pos[i] - info.origin[i]);

            // Get first index to consider.
            int idx[DIM];
            for (int i = 0; i < DIM; ++i) {
                idx[i] = std::max(0, std::min((int)ipos[i],
                                              Block::sizeArray[i] - 1));
            }
            // Compute 1D weights.
            double w[DIM];
            for (int i = 0; i < DIM; ++i)
                w[i] = ipos[i] - idx[i];

            // Do M2P interpolation.
            if (DIM == 2) {
                const double w00 = (1 - w[0]) * (1 - w[1]);
                const double w01 = (1 - w[0]) * (    w[1]);
                const double w10 = (    w[0]) * (1 - w[1]);
                const double w11 = (    w[0]) * (    w[1]);
                setter(part.id,
                       w00 * getter(lab.read(idx[0]    , idx[1]    ))
                     + w01 * getter(lab.read(idx[0]    , idx[1] + 1))
                     + w10 * getter(lab.read(idx[0] + 1, idx[1]    ))
                     + w11 * getter(lab.read(idx[0] + 1, idx[1] + 1)));
            } else {
                const double w000 = (1 - w[0]) * (1 - w[1]) * (1 - w[2]);
                const double w010 = (1 - w[0]) * (    w[1]) * (1 - w[2]);
                const double w100 = (    w[0]) * (1 - w[1]) * (1 - w[2]);
                const double w110 = (    w[0]) * (    w[1]) * (1 - w[2]);
                const double w001 = (1 - w[0]) * (1 - w[1]) * (    w[2]);
                const double w011 = (1 - w[0]) * (    w[1]) * (    w[2]);
                const double w101 = (    w[0]) * (1 - w[1]) * (    w[2]);
                const double w111 = (    w[0]) * (    w[1]) * (    w[2]);
                setter(part.id,
                       w000 * getter(lab.read(idx[0]    , idx[1]    , idx[2]    ))
                     + w010 * getter(lab.read(idx[0]    , idx[1] + 1, idx[2]    ))
                     + w100 * getter(lab.read(idx[0] + 1, idx[1]    , idx[2]    ))
                     + w110 * getter(lab.read(idx[0] + 1, idx[1] + 1, idx[2]    ))
                     + w001 * getter(lab.read(idx[0]    , idx[1]    , idx[2] + 1))
                     + w011 * getter(lab.read(idx[0]    , idx[1] + 1, idx[2] + 1))
                     + w101 * getter(lab.read(idx[0] + 1, idx[1]    , idx[2] + 1))
                     + w111 * getter(lab.read(idx[0] + 1, idx[1] + 1, idx[2] + 1)));
            }
        }
    };
    // END Linear P2M.

    block_processing(rhs, grid);
}

/*
 * Analogous to `linear_m2p`, except it stores the results into a `std::vector`
 * and returns it.
 */
template <int DIM,  // <-- Must be specified manually.
          typename BlockProcessing,
          typename Grid,
          typename Array,
          typename Getter>
auto linear_m2p_vector(BlockProcessing block_processing,
                       Grid &grid,
                       const Array &points,
                       Getter getter) {
    typedef typename Grid::BlockType block_type;
    typedef typename block_type::element_type element_type;
    typedef decltype(getter(element_type())) result_type;

    std::vector<result_type> result;
    result.resize(points.size());
    const auto setter = [&points, &result](int id, const result_type &value) {
        result[id] = value;
    };

    linear_m2p<DIM>(block_processing, grid, points, getter, setter);
    return result;
}

}  // Namespace utils.
}  // Namespace cubism.

#endif
