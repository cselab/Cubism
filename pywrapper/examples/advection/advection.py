#!/usr/bin/env coupling_compile_and_run

"""
Example of using Cubism through the coupling framework.

The example implements the 2D advection equation
    dphi / dt = -(ax * dphi / dx + ay * dphi / dy),
for some constant velocity (ax, ay).

There are three parts of the simulation:
    a) initialization
    b) calculation of dphi / dx (gradient)
    c) updating phi (phi += -a * gradient * dt).

Steps b) and c) are performed in each timestep.

Thus, the example below implements the simple 1st order Forward-time
backwards-space Euler scheme.
"""

import os
import sys
# Add folder ../.., to be able to import Cubism.
example_dir = os.path.dirname(__file__)
sys.path.insert(1, os.path.normpath(os.path.join(example_dir, '..', '..')))

import Cubism
import Cubism.misc

from coupling import *                  # Imports most common things.
from coupling.contrib.cpp import cmath  # <cmath> methods.
from coupling.contrib.cpp.extra import sqr, strprintf  # Utility methods.
from coupling.contrib.cpp.cstdio import printf


# Definition of the grid.
# We want our grid to contain two values:
#       `phi`  (a scalar, `double` by default)
#       `grad` (a 2D vector, as `std::array<double, 2>`)
# Cubism assumes that all items in the GridPoint have the same (base) type.
# Therefore, to make sure the user doesn't mix for example Doubles and Floats,
# the entry format is such that the user specifies only the names and
# dimensionality of the items.
GP = Cubism.GridPoint('phi', ('grad', 2))

# Instantiate Cubism library given the GridPoint and the block_size.
# These arguments affect the generated code, either as just template arguments
# of internal structures and methods, or directly by generating code that
# depends on the structure of GridPoint or on the block_size.
# Technically, this instantiates a Library instance, used later for links and
# communication (not in this example).
C = Cubism.Cubism(GP, block_size=(32, 32, 1))
BLOCK_NUM = (4, 4, 1)   # Number of blocks per each dimension.

VEL = (0.5, 0.1)        # Simulation parameter. (velocity)
MU = 2                  # Simulation parameter. (for initial condition).


@make_method  # Create a C++-method `void initial_state(void) { ... }`.
def initial_state():
    """Initializes the grid with a Gaussian."""
    def inner(p, xyz, **kwargs):
        """Sets the initial value of `phi` for the given grid point `p`.

        Arguments:
                p - GridPoint reference.
                xyz - Absolute x, y, z coordinates attributed to `p`.
                **kwargs - Other arguments / values we don't need here.
        """
        # Calculate (x - 0.5)^2 + (y - 0.5)^2. (ignores z coordinate!)
        rr = sqr(xyz[0] - .5) + sqr(xyz[1] - .5)
        # We do not support (yet) the typical format
        #   p.phi = exp(-MU * rr)
        # Instad, we manually write expressions using a special method.
        return expression(p.phi, '=', cmath.exp(-MU * rr))

    # Run function `inner` for each point.
    # C.process_pointwise passes to the given argument one positional argument
    # (GridPoint reference), and multiple keyword arguments (e.g. ijk, xyz...),
    # which might or might not be important to the user. As over time
    # additional keyword arguments could be added to the Cubism API, the passed
    # function therefore must always be defined to be able to ignore unused
    # keyword arguments.
    return C.process_pointwise(inner)


@make_method  # Creates `void timestep(const double &dt) { ... }`.
def timestep(dt: Double):
    """Implementation of a single time step."""
    def calc_grad(p, lab, info, **kwargs):
        """Calculates the gradient `.grad` of the field `.phi`.

        Arguments:
                p - GridPoint reference (for updating).
                lab - Wrapper for BlockLab (used for accessing neighboring
                      GridPoints)
                info - (const) BlockInfo reference.
                **kwargs - Other unused arguments.

        When processing a stencil, i.e. when updating a single grid point
        with values of neighboring grid points, Cubism must make sure the
        data is available. It has to take the data from the neighboring blocks,
        such that the data is available for fast access, by storing it in
        a continuous chuck of memory. Also, if distributed, Cubism must
        acquire data from other machines.

        `lab` here is a wrapper for BlockLab, in the sense that it properly
        generates the code for accessing the BlockLab, while keeping note
        how wide stencil has to be used.
        Here, stencil is (x0, x1) = (-1, 0), (y0, y1) = (-1, 0),
        (z0, z1) = (0, 0), inclusive.
        """
        fac = 1. / info.h_gridpoint
        assert VEL[0] >= 0 and VEL[1] >= 0, "Scheme unstable for vel < 0."
        # [fac] --> Code that should be executed before nested for loops
        #            (see generated C++ code).
        # [expression...] --> Code inside the nested for loops.

        # Because in Python it's not possible to capture assignment as in C++,
        # we write manually "expression(a, '=', b)" instead of "a = b".
        # Note: It is possible to capture assignment if LHS is not a variable
        # but an item like "x.a = b", by overloading __setitem__ such that
        # it keeps track of all operations. That format is available using
        # StructuredWorkflow (still in development).
        return [fac], [
            expression(p.grad[0], '=', fac * (lab(0, 0).phi - lab(-1, 0).phi)),
            expression(p.grad[1], '=', fac * (lab(0, 0).phi - lab(0, -1).phi)),
        ]

    def advance(p, **kwargs):
        """Performs a single step of Euler time integration.

        Updates `.phi` from the already calculated values of `.grad`.
        """
        fac0 = dt * VEL[0]
        fac1 = dt * VEL[1]
        # p.phi -= dt * dot(VEL, p.grad)
        return expression(p.phi, '-=', fac0 * p.grad[0] + fac1 * p.grad[1])

    # Here we list all substeps of one time step, in the order of execution.
    return [
        C.process_stencil(calc_grad),
        C.process_pointwise(advance),
    ]


# Immediately create a folder containing output files.
# Note: This is Python code, and it will be executed during compilation.
os.makedirs(os.path.join(os.path.dirname(__file__), 'output'), exist_ok=True)


# Specification of the complete workflow, in the order of execution.
workflow = [
    C.init(BLOCK_NUM),  # Initialize Cubism.
    initial_state(),    # Initial condition.
    # Do 50 time steps. After each, save the result in a file.
    ForRange(50)(lambda step: [
        timestep(0.01),

        # Save `.phi` values from the given library C.
        Cubism.misc.save_to_txt_file(
                C, 'phi', strprintf('"output/save_%03d.txt"', step)),
        printf('"Saved \'output/save%03d.txt\'.\\n"', step),
    ]),
]
