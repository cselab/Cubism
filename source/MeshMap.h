/*
 *  MeshMap.h
 *  Cubism
 *
 *  Created by Fabian Wermelinger on 05/03/17.
 *  Copyright 2017 CSE Lab, ETH Zurich. All rights reserved.
 *
 */
#ifndef MESHMAP_H_UAYWTJDH
#define MESHMAP_H_UAYWTJDH

#include <cassert>
#include <cmath>

struct UniformDensity
{
    void compute_spacing(const double xS, const double xE, const unsigned int ncells, double* const ary) const
    {
        const double h = (xE - xS) / ncells;
        for (int i = 0; i < ncells; ++i)
            ary[i] = h;
    }
};

struct GaussianDensity
{
    const double A;
    const double B;

    GaussianDensity(const double A=1.0, const double B=0.25) : A(A), B(B) {}

    void compute_spacing(const double xS, const double xE, const unsigned int ncells, double* const ary) const
    {
        const double y = 1.0/(B*(ncells+1));
        double ducky = 0.0;
        for (int i = 0; i < ncells; ++i)
        {
            const double x = i - (ncells+1)*0.5;
            ary[i] = 1.0/( A*std::exp(-0.5*x*x*y*y) + 1.0);
            ducky += ary[i];
        }

        const double scale = (xE-xS)/ducky;
        for (int i = 0; i < ncells; ++i)
            ary[i] *= scale;
    }
};


template <typename TBlock, int _ghostS=0, int _ghostE=0>
class MeshMap
{
public:
    MeshMap(const double xS, const double xE, const unsigned int Nblocks, const bool bUniform=true) :
        m_xS(xS), m_xE(xE), m_extent(xE-xS), m_Nblocks(Nblocks),
        m_Ncells(Nblocks*TBlock::sizeX), // assumes uniform cells in all directions!
        m_uniform(bUniform), m_initialized(false)
    {}

    ~MeshMap()
    {
        if (m_initialized)
        {
            delete[] m_grid_spacing;
            delete[] m_block_spacing;
        }
    }

    template <typename TKernel=UniformDensity>
    void init(const TKernel& kernel=UniformDensity())
    {
        _alloc();

        kernel.compute_spacing(m_xS, m_xE, m_Ncells, m_grid_spacing);

        assert(m_Nblocks > 0);
        for (int i = 0; i < m_Nblocks; ++i)
        {
            double delta_block = 0.0;
            for (int j = 0; j < TBlock::sizeX; ++j)
                delta_block += m_grid_spacing[i*TBlock::sizeX + j];
            m_block_spacing[i] = delta_block;
        }

        m_initialized = true;
    }

    inline double extent() const { return m_extent; }
    inline unsigned int nblocks() const { return m_Nblocks; }
    inline unsigned int ncells() const { return m_Ncells; }
    inline bool uniform() const { return m_uniform; }

    inline double cell_width(const int ix) const
    {
        assert(m_initialized && ix >= 0 && ix < m_Ncells);
        return m_grid_spacing[ix];
    }

    inline double block_width(const int bix) const
    {
        assert(m_initialized && bix >= 0 && bix < m_Nblocks);
        return m_block_spacing[bix];
    }

    inline double block_origin(const int bix) const
    {
        assert(m_initialized && bix >= 0 && bix < m_Nblocks);
        double offset = m_xS;
        for (int i = 0; i < bix; ++i)
            offset += m_block_spacing[i];
        return offset;
    }

    inline double* get_grid_spacing(const int bix)
    {
        assert(m_initialized && bix >= 0 && bix < m_Nblocks);
        return &m_grid_spacing[bix*TBlock::sizeX];
    }

private:
    const double m_xS;
    const double m_xE;
    const double m_extent;
    const unsigned int m_Nblocks;
    const unsigned int m_Ncells;
    const bool m_uniform;

    bool m_initialized;
    double* m_grid_spacing;
    double* m_block_spacing;

    inline void _alloc()
    {
        m_grid_spacing = new double[m_Ncells];
        m_block_spacing= new double[m_Nblocks];
    }
};

#endif /* MESHMAP_H_UAYWTJDH */
