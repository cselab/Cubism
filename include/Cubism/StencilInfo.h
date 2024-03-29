#pragma once

#include <cassert>
#include <iostream>
#include <vector>

namespace cubism {

/** 
 * @brief Describes a stencil of points.
 * 
 * This struct is used by BlockLab to determine what halo cells need to be communicated between 
 * different GridBlocks. For a gridpoint (i,j,k) the gridpoints included in the stencil are points
 * (i+ix,j+iy,k+iz), where ix,iy,iz are determined by the stencil starts and ends.
 */
struct StencilInfo
{
   int sx; ///< start of stencil in the x-direction (sx <= ix)
   int sy; ///< start of stencil in the y-direction (sy <= iy)
   int sz; ///< start of stencil in the z-direction (sz <= iz)
   int ex; ///< end of stencil (+1) in the x-direction (ix < ex)
   int ey; ///< end of stencil (+1) in the y-direction (iy < ey)
   int ez; ///< end of stencil (+1) in the z-direction (iz < ez)
   std::vector<int> selcomponents; ///< Components ('members') of Element that will be used
   bool tensorial; ///< if false, stencil only includes points with |ix|+|iy|+|iz| <= 1

   /// Empty constructor.
   StencilInfo() {}

   /// Constructor
   StencilInfo(int _sx, int _sy, int _sz, int _ex, int _ey, int _ez, bool _tensorial, const std::vector<int>
&components) : sx(_sx), sy(_sy), sz(_sz), ex(_ex), ey(_ey), ez(_ez), selcomponents(components), tensorial(_tensorial)
   {
      assert(selcomponents.size() > 0);

      if (!isvalid())
      {
         std::cout << "Stencilinfo instance not valid. Aborting\n";
         abort();
      }
   }

   /// Copy constructor.
   StencilInfo(const StencilInfo &c) : sx(c.sx), sy(c.sy), sz(c.sz), ex(c.ex), ey(c.ey), ez(c.ez), selcomponents(c.selcomponents), tensorial(c.tensorial) {}

   /// Return a vector with all integers that make up this StencilInfo.
   std::vector<int> _all() const
   {
      int extra[] = {sx, sy, sz, ex, ey, ez, (int)tensorial};
      std::vector<int> all(selcomponents);
      all.insert(all.end(), extra, extra + sizeof(extra) / sizeof(int));

      return all;
   }

   /// Check if one stencil is contained in another.
   bool operator<(StencilInfo s) const
   {
      std::vector<int> me = _all(), you = s._all();

      const int N = std::min(me.size(), you.size());

      for (int i = 0; i < N; ++i)
         if (me[i] < you[i]) return true;
         else if (me[i] > you[i])
            return false;

      return me.size() < you.size();
   }

   /// Check if the ends are smaller than the starts of this stencil.
   bool isvalid() const
   {
      const bool not0 = selcomponents.size() == 0;
      const bool not1 = sx > 0 || ex <= 0 || sx > ex;
      const bool not2 = sy > 0 || ey <= 0 || sy > ey;
      const bool not3 = sz > 0 || ez <= 0 || sz > ez;

      return !(not0 || not1 || not2 || not3);
   }
};

}//namespace cubism
