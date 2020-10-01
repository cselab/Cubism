#pragma once
#include "AMR_MeshAdaptation.h"

namespace cubism
{


template <typename TGrid, typename TLab>
class MeshAdaptationMPI : public MeshAdaptation<TGrid,TLab>
{
  public:
   MeshAdaptationMPI(TGrid &grid, double Rtol, double Ctol): MeshAdaptation<TGrid,TLab>(grid,Rtol,Ctol)
   {
    std::cout << "MeshAdaptationMPI created." << std::endl;
   }

};

} // namespace cubism
