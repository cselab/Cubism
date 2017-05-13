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
#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>
#include "ArgumentParser.h"

class MeshDensity
{
public:
    const bool uniform;
    MeshDensity(const bool _uniform) : uniform(_uniform) {}

    virtual void compute_spacing(const double xS, const double xE, const unsigned int ncells, double* const ary,
            const unsigned int ghostS=0, const unsigned int ghostE=0, double* const ghost_spacing=NULL) const = 0;
};


class UniformDensity : public MeshDensity
{
public:
    UniformDensity() : MeshDensity(true) {}

    virtual void compute_spacing(const double xS, const double xE, const unsigned int ncells, double* const ary,
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

class GaussianDensity : public MeshDensity
{
    const double A;
    const double B;

public:
    struct DefaultParameter
    {
        double A, B;
        DefaultParameter() : A(1.0), B(0.25) {}
    };

    GaussianDensity(const DefaultParameter d=DefaultParameter()) : MeshDensity(false), A(d.A), B(d.B) {}

    virtual void compute_spacing(const double xS, const double xE, const unsigned int ncells, double* const ary,
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

class SmoothHeavisideDensity : public MeshDensity
{
protected:
    const double A;
    const double B;
    const double c;
    const double eps;

    inline double _smooth_heaviside(const double r, const double A, const double B, const double c, const double eps, const bool mirror=false) const
    {
        const double theta = M_PI*std::max(0.0, std::min(1.0, (1.0-2.0*mirror)/eps*(r - (c-0.5*eps)) + 1.0*mirror));
        const double q = 0.5*(std::cos(theta) + 1.0);
        return B + (A-B)*q;
    }

public:
    struct DefaultParameter
    {
        double A, B, c1, eps1;
        DefaultParameter() : A(2.0), B(1.0), c1(0.5), eps1(0.2) {}
    };

    SmoothHeavisideDensity(const double A, const double B, const double c, const double eps) : MeshDensity(false), A(A), B(B), c(c), eps(eps) {}
    SmoothHeavisideDensity(const DefaultParameter d=DefaultParameter()) : MeshDensity(false), A(d.A), B(d.B), c(d.c1), eps(d.eps1) {}

    virtual void compute_spacing(const double xS, const double xE, const unsigned int ncells, double* const ary,
            const unsigned int ghostS=0, const unsigned int ghostE=0, double* const ghost_spacing=NULL) const
    {
        assert(std::min(A,B) > 0.0);
        assert(0.5*eps <= c && c <= 1.0-0.5*eps);

        const unsigned int total_cells = ncells + ghostE + ghostS;
        double* const buf = new double[total_cells];

        double ducky = 0.0;
        for (int i = 0; i < ncells; ++i)
        {
            const double r = (static_cast<double>(i)+0.5)/ncells;
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

class SmoothHatDensity : public SmoothHeavisideDensity
{
protected:
    const double c2;
    const double eps2;

public:
    struct DefaultParameter
    {
        double A, B, c1, eps1, c2, eps2;
        DefaultParameter() : A(2.0), B(1.0), c1(0.25), eps1(0.15), c2(0.75), eps2(0.15) {}
    };

    SmoothHatDensity(const DefaultParameter d=DefaultParameter()) : SmoothHeavisideDensity(d.A,d.B,d.c1,d.eps1), c2(d.c2), eps2(d.eps2) {}

    virtual void compute_spacing(const double xS, const double xE, const unsigned int ncells, double* const ary,
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
            const double r = (static_cast<double>(i)+0.5)/ncells;
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


class MeshDensityFactory
{
public:
    MeshDensityFactory(ArgumentParser& parser) : m_parser(parser) { _make_mesh_kernels(); }
    ~MeshDensityFactory() { _dealloc(); }

    inline MeshDensity* get_mesh_kernel(const int i) { return m_mesh_kernels[i]; }

private:
    ArgumentParser& m_parser;
    std::vector<MeshDensity*> m_mesh_kernels;

    void _dealloc()
    {
        for (int i = 0; i < (int)m_mesh_kernels.size(); ++i)
            delete m_mesh_kernels[i];
    }

    void _make_mesh_kernels()
    {
        std::vector<std::string> suffix;
        suffix.push_back("_x");
        suffix.push_back("_y");
        suffix.push_back("_z");
        for (int i = 0; i < (int)suffix.size(); ++i)
        {
            const std::string mesh_density("mesh_density" + suffix[i]);
            if (m_parser.exist(mesh_density))
            {
                const std::string density_function(m_parser(mesh_density).asString());
                const std::string A("A" + suffix[i]);
                const std::string B("B" + suffix[i]);
                const std::string c1("c1" + suffix[i]);
                const std::string eps1("eps1" + suffix[i]);
                const std::string c2("c2" + suffix[i]);
                const std::string eps2("eps2" + suffix[i]);
                if (density_function == "GaussianDensity")
                {
                    typename GaussianDensity::DefaultParameter p;
                    if (m_parser.exist(A)) p.A = m_parser(A).asDouble();
                    if (m_parser.exist(B)) p.B = m_parser(B).asDouble();
                    m_mesh_kernels.push_back(new GaussianDensity(p));
                }
                else if (density_function == "SmoothHeavisideDensity")
                {
                    typename SmoothHeavisideDensity::DefaultParameter p;
                    if (m_parser.exist(A)) p.A = m_parser(A).asDouble();
                    if (m_parser.exist(B)) p.B = m_parser(B).asDouble();
                    if (m_parser.exist(c1)) p.c1 = m_parser(c1).asDouble();
                    if (m_parser.exist(eps1)) p.eps1 = m_parser(eps1).asDouble();
                    m_mesh_kernels.push_back(new SmoothHeavisideDensity(p));
                }
                else if (density_function == "SmoothHatDensity")
                {
                    typename SmoothHatDensity::DefaultParameter p;
                    if (m_parser.exist(A)) p.A = m_parser(A).asDouble();
                    if (m_parser.exist(B)) p.B = m_parser(B).asDouble();
                    if (m_parser.exist(c1)) p.c1 = m_parser(c1).asDouble();
                    if (m_parser.exist(eps1)) p.eps1 = m_parser(eps1).asDouble();
                    if (m_parser.exist(c2)) p.c2 = m_parser(c2).asDouble();
                    if (m_parser.exist(eps2)) p.eps2 = m_parser(eps2).asDouble();
                    m_mesh_kernels.push_back(new SmoothHatDensity(p));
                }
                else
                {
                    std::cerr << "ERROR: MeshMap.h: Undefined mesh density function." << std::endl;
                    abort();
                }
            }
            else
                m_mesh_kernels.push_back(new UniformDensity);
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

    void init(const MeshDensity* const kernel, const unsigned int ghostS=0, const unsigned int ghostE=0, double* const ghost_spacing=NULL)
    {
        _alloc();

        kernel->compute_spacing(m_xS, m_xE, m_Ncells, m_grid_spacing, ghostS, ghostE, ghost_spacing);

        assert(m_Nblocks > 0);
        for (int i = 0; i < m_Nblocks; ++i)
        {
            double delta_block = 0.0;
            for (int j = 0; j < TBlock::sizeX; ++j)
                delta_block += m_grid_spacing[i*TBlock::sizeX + j];
            m_block_spacing[i] = delta_block;
        }

        m_uniform = kernel->uniform;
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
