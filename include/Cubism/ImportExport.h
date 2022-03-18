#include <vector>

namespace cubism
{

/** Fully refine the AMR grid and store into the given contiguous matrix.

    The output matrix in expected to have xStride == 1,
    yStride == xNumCells and zStride == xNumCells * yNumCells.

    Blocks already at the highest level of refinement are copied without
    interpolation. All other blocks are upscaled using interpolation that is
    quadratic everywhere except at the coarse-fine boundary, where it is
    linear.

    Template arguments:
      Lab: block lab used for the stencil for the upscaling interpolation

    Arguments:
      grid: AMR MPI grid
      getter: Function that takes an element and returns an arbitrary
          arithmetic type T. For example, it can be used to select only one
          element of a vector element. The upscaling interpolation is performed
          after the transformation.
      components: components of the element that need to be exchanged for ghost
          cells
      out: output matrix
*/
template <typename Lab, typename Grid, typename Getter, typename T>
void exportGridToUniformMatrix(
    Grid *grid, Getter getter, std::vector<int> components, T * __restrict__ out);

}  // namespace cubism
