#pragma once


#include "GridMPI.h"
#include "AMR_SynchronizerMPI.h"

CUBISM_NAMESPACE_BEGIN

template<typename MyBlockLab>
class BlockLabMPI : public MyBlockLab
{
public:
    typedef typename MyBlockLab::Real Real;

private:
    typedef typename MyBlockLab::BlockType BlockType; 
    typedef SynchronizerMPI_AMR<Real> SynchronizerMPIType;   
    SynchronizerMPIType * refSynchronizerMPI;


public:
    template< typename TGrid >
    void prepare(GridMPI<TGrid>& grid, SynchronizerMPIType& synchronizer)
    {
        refSynchronizerMPI = &synchronizer;
        StencilInfo stencil  = refSynchronizerMPI->getstencil();
        StencilInfo Cstencil = refSynchronizerMPI->getCstencil();
        assert(stencil.isvalid());
        MyBlockLab::prepare(grid, stencil.sx, stencil.ex, 
                                  stencil.sy, stencil.ey, 
                                  stencil.sz, stencil.ez, 
                                  stencil.tensorial, 
                                  Cstencil.sx, Cstencil.ex, 
                                  Cstencil.sy, Cstencil.ey, 
                                  Cstencil.sz, Cstencil.ez);
    }

    void load(const BlockInfo& info, const Real t=0, const bool applybc=true)
    {
        MyBlockLab::load(info, t, applybc);     

        typedef typename MyBlockLab::ElementType ET;
        const int gptfloats = sizeof(ET)/sizeof(Real);
        const size_t  Length [3] = {MyBlockLab::m_cacheBlock    ->getSize(0),MyBlockLab::m_cacheBlock    ->getSize(1),MyBlockLab::m_cacheBlock    ->getSize(2)};
        const size_t CLength [3] = {MyBlockLab::m_CoarsenedBlock->getSize(0),MyBlockLab::m_CoarsenedBlock->getSize(1),MyBlockLab::m_CoarsenedBlock->getSize(2)};
        const size_t m_nElemsPerSlice [2] ={MyBlockLab::m_cacheBlock->getNumberOfElementsPerSlice(), MyBlockLab::m_CoarsenedBlock->getNumberOfElementsPerSlice()}; 
        Real * dst  = & MyBlockLab ::m_cacheBlock    ->LinAccess(0).alpha1rho1;
        Real * dst1 = & MyBlockLab ::m_CoarsenedBlock->LinAccess(0).alpha1rho1;
        refSynchronizerMPI->fetch(info,gptfloats,Length,CLength,m_nElemsPerSlice,dst,dst1);

        MyBlockLab::post_load(info, t, applybc);
    }

    void release()
    {
        MyBlockLab::release();
    }
};

CUBISM_NAMESPACE_END
