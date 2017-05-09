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
    const bool uniform;

    UniformDensity() : uniform(true) {}

    void compute_spacing(const double xS, const double xE, const unsigned int ncells, double* const ary,
            const unsigned int ghostS=0, const unsigned int ghostE=0, double* const ghost_spacing=NULL) const
    {
        const double h = (xE - xS) / ncells;
        for (int i = 0; i < ncells; ++i)
            ary[i] = h;

        // ghost cells are given by ghost start (ghostS) and ghost end
        // (ghostE) and count the number of ghosts on either side (inclusive).
        // For example, for a symmetric 6-point stencil -> ghostS = 3 and
        // ghostE = 3.  ghost_spacing must provide valid memory for it.
        if (ghost_spacing)
            for (int i = 0; i < ghostS+ghostE; ++i)
                ghost_spacing[i] = h;
    }
};

struct GaussianDensity
{
    const double A;
    const double B;
    const bool uniform;

    GaussianDensity(const double A=1.0, const double B=0.25) : A(A), B(B), uniform(false) {}

    void compute_spacing(const double xS, const double xE, const unsigned int ncells, double* const ary,
            const unsigned int ghostS=0, const unsigned int ghostE=0, double* const ghost_spacing=NULL) const
    {
        const unsigned int total_cells = ncells + ghostE + ghostS;
        double* const buf = new double[total_cells];

        const double y = 1.0/(B*(total_cells+1));
        double ducky = 0.0;
        for (int i = 0; i < total_cells; ++i)
        {
            const double x = i - (total_cells+1)*0.5;
            buf[i] = 1.0/(A*std::exp(-0.5*x*x*y*y) + 1.0);

            if (i >= ghostS && i < ncells + ghostS)
                ducky += buf[i];
        }

        const double scale = (xE-xS)/ducky;
        for (int i = 0; i < total_cells; ++i)
            buf[i] *= scale;

        for (int i = 0; i < ncells; ++i)
            ary[i] = buf[i+ghostS];

        if (ghost_spacing)
        {
            for (int i = 0; i < ghostS; ++i)
                ghost_spacing[i] = buf[i];
            for (int i = 0; i < ghostE; ++i)
                ghost_spacing[i+ghostS] = buf[i+ncells+ghostS];
        }
    }
};

struct SmoothHeavisideDensity
{
    const double A;
    const double B;
    const double c;
    const double eps;
    const bool uniform;

    SmoothHeavisideDensity(const double A=2.0, const double B=1.0, const double c=0.5, const double eps=0.2) : A(A), B(B), c(c), eps(eps), uniform(false) {}

    inline double _smooth_heaviside(const double r, const double A, const double B, const double c, const double eps, const bool mirror=false) const
    {
        const double theta = M_PI*std::max(0.0, std::min(1.0, (1.0-2.0*mirror)/eps*(r - (c-0.5*eps)) + 1.0*mirror));
        const double q = 0.5*(std::cos(theta) + 1.0);
        return B + (A-B)*q;
    }

    void compute_spacing(const double xS, const double xE, const unsigned int ncells, double* const ary,
            const unsigned int ghostS=0, const unsigned int ghostE=0, double* const ghost_spacing=NULL) const
    {
        assert(std::min(A,B) > 0.0);
        assert(0.5*eps <= c && c <= 1.0-0.5*eps);

        const unsigned int total_cells = ncells + ghostE + ghostS;
        double* const buf = new double[total_cells];

        double ducky = 0.0;
        for (int i = 0; i < ncells; ++i)
        {
            const double r = static_cast<double>(i)/ncells;
            double rho;
            if (A >= B)
                rho = _smooth_heaviside(r, A, B, c, eps);
            else
                rho = _smooth_heaviside(r, B, A, c, eps, true);
            buf[i+ghostS] = 1.0/rho;
            ducky += buf[i+ghostS];
        }
        for (int i = 0; i < ghostS; ++i)
            buf[i] = buf[ghostS];
        for (int i = 0; i < ghostE; ++i)
            buf[i+ncells+ghostS] = buf[ncells+ghostS-1];

        const double scale = (xE-xS)/ducky;
        for (int i = 0; i < total_cells; ++i)
            buf[i] *= scale;

        for (int i = 0; i < ncells; ++i)
            ary[i] = buf[i+ghostS];

        if (ghost_spacing)
        {
            for (int i = 0; i < ghostS; ++i)
                ghost_spacing[i] = buf[i];
            for (int i = 0; i < ghostE; ++i)
                ghost_spacing[i+ghostS] = buf[i+ncells+ghostS];
        }
    }
};

struct SmoothHatDensity : public SmoothHeavisideDensity
{
    const double c2;
    const double eps2;

    SmoothHatDensity(const double A=2.0, const double B=1.0, const double c1=0.25, const double eps1=0.15, const double c2=0.75, const double eps2=0.15) : SmoothHeavisideDensity(A,B,c1,eps1), c2(c2), eps2(eps2) {}

    void compute_spacing(const double xS, const double xE, const unsigned int ncells, double* const ary,
            const unsigned int ghostS=0, const unsigned int ghostE=0, double* const ghost_spacing=NULL) const
    {
        assert(std::min(A,B) > 0.0);
        assert(0.5*eps <= c && c <= c2-0.5*eps2);
        assert(c2 <= 1.0-0.5*eps2);

        const unsigned int total_cells = ncells + ghostE + ghostS;
        double* const buf = new double[total_cells];

        double ducky = 0.0;
        for (int i = 0; i < ncells; ++i)
        {
            const double r = static_cast<double>(i)/ncells;
            double rho;
            if (r <= c+0.5*eps)
                if (A >= B)
                    rho = _smooth_heaviside(r, A, B, c, eps);
                else
                    rho = _smooth_heaviside(r, B, A, c, eps, true);
            else if (r >= c2-0.5*eps2)
                if (A >= B)
                    rho = _smooth_heaviside(r, A, B, c2, eps2, true);
                else
                    rho = _smooth_heaviside(r, B, A, c2, eps2);
            else
                rho = B;
            buf[i+ghostS] = 1.0/rho;
            ducky += buf[i+ghostS];
        }
        for (int i = 0; i < ghostS; ++i)
            buf[i] = buf[ghostS];
        for (int i = 0; i < ghostE; ++i)
            buf[i+ncells+ghostS] = buf[ncells+ghostS-1];

        const double scale = (xE-xS)/ducky;
        for (int i = 0; i < total_cells; ++i)
            buf[i] *= scale;

        for (int i = 0; i < ncells; ++i)
            ary[i] = buf[i+ghostS];

        if (ghost_spacing)
        {
            for (int i = 0; i < ghostS; ++i)
                ghost_spacing[i] = buf[i];
            for (int i = 0; i < ghostE; ++i)
                ghost_spacing[i+ghostS] = buf[i+ncells+ghostS];
        }
    }
};


template <typename TBlock>
class MeshMap
{
public:
    MeshMap(const double xS, const double xE, const unsigned int Nblocks) :
        m_xS(xS), m_xE(xE), m_extent(xE-xS), m_Nblocks(Nblocks),
        m_Ncells(Nblocks*TBlock::sizeX), // assumes uniform cells in all directions!
        m_uniform(true), m_initialized(false)
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
    void init(const TKernel& kernel=UniformDensity(), const unsigned int ghostS=0, const unsigned int ghostE=0, double* const ghost_spacing=NULL)
    {
        _alloc();

        kernel.compute_spacing(m_xS, m_xE, m_Ncells, m_grid_spacing, ghostS, ghostE, ghost_spacing);

        assert(m_Nblocks > 0);
        for (int i = 0; i < m_Nblocks; ++i)
        {
            double delta_block = 0.0;
            for (int j = 0; j < TBlock::sizeX; ++j)
                delta_block += m_grid_spacing[i*TBlock::sizeX + j];
            m_block_spacing[i] = delta_block;
        }

        m_uniform = kernel.uniform;
        m_initialized = true;
    }

    inline double start() const { return m_xS; }
    inline double end() const { return m_xE; }
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

    inline double* data_grid_spacing() { return m_grid_spacing; }

private:
    const double m_xS;
    const double m_xE;
    const double m_extent;
    const unsigned int m_Nblocks;
    const unsigned int m_Ncells;

    bool m_uniform;
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
