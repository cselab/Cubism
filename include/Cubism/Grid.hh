#include "Grid.h"

namespace cubism
{

template <typename Block, template <typename> class Allocator>
template <typename T>
T *Grid<Block, Allocator>::copyToUniformNoInterpolation(T *out) const
{
  constexpr int BX = Block::sizeX;
  constexpr int BY = Block::sizeY;
  constexpr int BZ = Block::sizeZ;
  const auto C = getMaxMostRefinedCells();
  constexpr int xStride = 1;
  const int yStride = C[0];
  const int zStride = C[0] * C[1];

  if (out == nullptr)
    out = new T[C[0] * C[1] * C[2]];

#pragma omp parallel for
  for (size_t i = 0; i < m_vInfo.size(); ++i) {
    const BlockInfo &info = m_vInfo[i];
    const Block &block = *(Block *)info.ptrBlock;
    const int level = info.level;
    const int factor = 1 << ((levelMax - 1) - level);

    for (int iz = 0; iz < Block::sizeZ; ++iz)
    for (int iy = 0; iy < Block::sizeY; ++iy)
    for (int ix = 0; ix < Block::sizeX; ++ix)
    {
      // If needed, we can add a func() that transforms the given cell.
      const auto value = block(ix, iy, iz);

      const int zOffset = (info.index[2] * BZ + iz) * factor;
      const int yOffset = (info.index[1] * BY + iy) * factor;
      const int xOffset = (info.index[0] * BX + ix) * factor;

      // It might be beneficial to make this
      for (int jz = 0; jz < (DIMENSION == 3 ? factor : 1); ++jz)
      for (int jy = 0; jy < factor; ++jy)
      for (int jx = 0; jx < factor; ++jx)
      {
        const int kz = zOffset + jz;
        const int ky = yOffset + jy;
        const int kx = xOffset + jx;
        const int idx = kx * xStride + ky * yStride + kz * zStride;
        assert(idx < C[0] * C[1] * C[2]);
        out[idx] = value;
      }
    }
  }

  return out;
}

}  // namespace cubism
