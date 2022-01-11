#include "Grid.h"

namespace cubism
{

template <typename Block, template <typename> class Allocator>
auto Grid<Block, Allocator>::copyToUniformNoInterpolation(
    ElementType *out) const -> ElementType *
{
  constexpr int BX = Block::sizeX;
  constexpr int BY = Block::sizeY;
  constexpr int BZ = Block::sizeZ;
  const auto C = getMaxMostRefinedCells();
  constexpr int xStride = 1;
  const int yStride = C[0];
  const int zStride = C[0] * C[1];

  if (out == nullptr)
    out = new ElementType[C[0] * C[1] * C[2]];

#pragma omp parallel for
  for (size_t i = 0; i < m_vInfo.size(); ++i)
  {
    const BlockInfo &info = m_vInfo[i];
    const Block &block = *(Block *)info.ptrBlock;
    const int factor = 1 << ((levelMax - 1) - info.level);

    for (int iz = 0; iz < Block::sizeZ; ++iz)
    for (int iy = 0; iy < Block::sizeY; ++iy)
    for (int ix = 0; ix < Block::sizeX; ++ix)
    {
      // If needed, we can add a func() that transforms the given cell.
      const auto value = block(ix, iy, iz);

      const int zOffset = (info.index[2] * BZ + iz) * factor;
      const int yOffset = (info.index[1] * BY + iy) * factor;
      const int xOffset = (info.index[0] * BX + ix) * factor;

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


template <typename Block, template <typename> class Allocator>
void Grid<Block, Allocator>::copyFromMatrix(const ElementType *in)
{
  constexpr int BX = Block::sizeX;
  constexpr int BY = Block::sizeY;
  constexpr int BZ = Block::sizeZ;
  const auto C = getMaxMostRefinedCells();
  constexpr int xStride = 1;
  const int yStride = C[0];
  const int zStride = C[0] * C[1];

#pragma omp parallel for
  for (size_t i = 0; i < m_vInfo.size(); ++i)
  {
    const BlockInfo &info = m_vInfo[i];
    Block &block = *(Block *)info.ptrBlock;
    const int factor = 1 << ((levelMax - 1) - info.level);
    const auto avgFactor = (typename ElementType::RealType)1
                         / (factor * factor * (DIMENSION == 3 ? factor : 1));

    for (int iz = 0; iz < BZ; ++iz)
    for (int iy = 0; iy < BY; ++iy)
    for (int ix = 0; ix < BX; ++ix)
    {
      const int zOffset = (info.index[2] * BZ + iz) * factor;
      const int yOffset = (info.index[1] * BY + iy) * factor;
      const int xOffset = (info.index[0] * BX + ix) * factor;

      ElementType sum{};
      for (int jz = 0; jz < (DIMENSION == 3 ? factor : 1); ++jz)
      for (int jy = 0; jy < factor; ++jy)
      for (int jx = 0; jx < factor; ++jx)
      {
        const int kz = zOffset + jz;
        const int ky = yOffset + jy;
        const int kx = xOffset + jx;
        const int idx = kx * xStride + ky * yStride + kz * zStride;
        assert(idx < C[0] * C[1] * C[2]);
        sum += in[idx];
      }

      block(ix, iy, iz) = avgFactor * sum;
    }
  }
}

}  // namespace cubism
