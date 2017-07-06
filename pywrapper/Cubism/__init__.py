"""Cubism API wrapper for the coupling framework.

Provides access to the Cubism API:
    process_pointwise
    process_stencil

Additional Cubism application-oriented API can be found in misc.py.

Usage:
    import Cubism

    # First, define what values the grid has to store:
    GP = Cubism.GridPoint(
        'density',
        'pressure',
        ('velocity, 3),
        value_type=Double,   # Either Float or Double.
    )
    # This creates a struct of the form:
    #    struct GridPoint {
    #      double density;
    #      double pressure;
    #      std::array<double, 3> velocity;
    #    }

    # Second, instantiate Cubism.
    # block_size argument is optional.
    C = Cubism.Cubism(GP, block_size=(32, 32, 1))

    # Third, in the workflow, use API:
    # (for details, see below)
    step1 = C.process_pointwise(functor1)
    step2 = C.process_stencil(functor2)
    ...
"""
# TODO: Test value_type=Float, maybe it requires changing define compile flags.


from coupling import methods
from coupling.contrib import domains
from coupling.compiler.tokens import Typename
from coupling.library import Library
from coupling.links.backends import mpi
from coupling.methods.helpers import fake_variable, rerun_on_reuse, \
        lazy_global_variable, inline_code, noop, value_placeholder
from coupling.methods.operators import assign
from coupling.methods.variables import New
from coupling.types import Auto, Bool, Double, Int, LongLong, AutoConstructor
from coupling.types.utils import get_template_cname, get_template_names
from coupling.types.arrays import Pointer, StdArray, StdVector, dereference
from coupling.types.structs import Struct
from coupling.workflows.decorators import make_lambda, make_method
from coupling.workflows.linear import InlineWorkflow
from coupling.workflows.loops import ForRange

from functools import reduce

# TODO: Proper non-hacky C++ references once they are implemented.
Ref = lambda x: x

# TODO: selcomponents in Lab.

MAX_DIM = 3

# Passed as an argument to functors of process_pointwise and process_stencil.
BlockInfo = Struct(
    ('blockID', LongLong),
    ('ptrBlock', Pointer(Auto)),
    ('special', Bool),
    ('index', StdArray(Int, 3)),      # As a substitute for int[3].
    ('origin', StdArray(Double, 3)),  # As a substitute for double[3].
    ('h', Double),
    ('h_gridpoint', Double),
    ctype='BlockInfo',
    header='Cubism/source/BlockInfo.h',
)


StencilInfo = Struct(  # Internal.
    ('sx', Int),
    ('sy', Int),
    ('sz', Int),
    ('ex', Int),
    ('ey', Int),
    ('ez', Int),
    ('selcomponents', StdVector(Int)),
    ('tensorial', Bool),
    ctype='StencilInfo',
    header='Cubism/source/StencilInfo.h',
)
StencilInfo.constructor = AutoConstructor(output=StencilInfo)



class GridPoint(Struct):
    def __init__(self, *items, value_type=Double, ctype='GridPoint',
                 pyname='GridPoint', **kwargs):
        """Create a Struct representing a grid point.

        Struct item is specified by a string representing its names or a tuple
        representing names and dimensions.  Type of the items (or array items)
        must have the same type, which is a constraint of the library itself.
        The type is specified using the argument `value_type` (default is
        Double).

        Usage example:
            Point = GridPoint(
                'scalar1',
                'scalar2',
                ('vector1', 3),
                # ('matrix1', 3, 3),  # NOT SUPPORTED YET.
                value_type=Double,
            )
            p = Point()
            ...
            a = p.scalar1
            b = p.vector1[i]
            # c = p.matrix1[i][j]  # NOT SUPPORTED YET.
            # c = p.matrix1[i, j]  # NOT SUPPORTED YET.
            ...
        """
        _items = []
        _count = 0
        _offsets = []
        for item in items:
            _offsets.append(_count)
            if isinstance(item, str):
                _items.append((item, value_type))
                _count += 1
            elif isinstance(item, tuple):
                assert len(item) > 1
                assert len(item) == 2  # For now...
                assert isinstance(item[0], str), item[0]
                assert all(isinstance(size, int) for size in item[1:]), item[1:]
                _items.append((item[0], StdArray(value_type, *item[1:])))
                _count += reduce(lambda x, y: x * y, item[1:])
            else:
                raise TypeError("Unexpected '{}'.".format(item))

        super().__init__(*_items, ctype=ctype, pyname=pyname, declare=True,
                         **kwargs)
        self.count = _count
        self.offsets = _offsets

        @self.make_member_method
        def clear(this:self):
            # TODO: Foreach maybe shouldn't be recursive (i.e. this guy should
            # handle vectors and matrices manually.)
            return this.foreach_item(lambda item: assign(item, value_type(0)))
        self.clear = clear




def _getter_factory(self, name, element_type):
    getter = self.Method(name=name, args=[Int] * self.DIM, output=element_type)
    return lambda this, *ijk: rerun_on_reuse(
            getter(this, *(list(ijk) + [0] * (self.DIM - len(ijk)))))



class Block(Struct):
    """Defines cubism::Block<grid_point, sizeX, sizeY, sizeZ>."""
    def __init__(self, element_type, block_size, DIM):
        # TODO: Template and typedef should be automated.
        _ctype, pyname = get_template_names(
                'cubism::Block', 'CubismBlock', element_type, *block_size)
        ctype = Typename('_CubismBlock', self)

        sizeX, sizeY, sizeZ = block_size
        array_type = \
                StdArray(StdArray(StdArray(element_type, sizeX), sizeY), sizeZ)
        super().__init__(('data', array_type), ctype=ctype, pyname=pyname,
                         header='Cubism/source/Block.h')
        self._ctype = _ctype
        self.sizeX, self.sizeY, self.sizeZ = block_size
        self.sizeArray = block_size
        self.element_type = element_type
        self.grid_point = element_type
        self.block_size = block_size
        self.DIM = DIM

        self.clear = self.Method(name='clear')
        self.functor = _getter_factory(self, '', element_type)

    def definition(self):
        # TODO: This should be a part of TypeBase.
        return ['typedef ', self._ctype, ' ', self.ctype.name]



class BlockLab(Struct):
    """Defines BlockLab<block_type>."""
    def __init__(self, block_type, DIM):
        ctype, pyname = get_template_names('BlockLab', 'BlockLab', block_type)
        super().__init__(ctype=ctype, pyname=pyname,
                         header='Cubism/source/BlockLab.h')
        self.block_type = block_type
        self.element_type = block_type.element_type
        self.DIM = DIM

        # Member functions. .functor is member method called on __call__.
        self.functor = _getter_factory(self, '', block_type.element_type)
        self.read = _getter_factory(self, 'read', block_type.element_type)



class LabGetter(object):
    """Helper function for LabWrapper."""
    def __init__(self, lab_wrapper, ijk):
        self.lab_wrapper = lab_wrapper
        self.ijk = ijk

    def __call__(self, *dijk):
        """Register stencil range and return lab access."""
        # TODO: This implementation doesn't ignore unused lab accesses:
        #       def kernel(p, lab, **kwargs):
        #           a = lab(1, 0)  # <-- should be ignored.
        #           return expression(p, '=', lab(0, 1))
        #
        # In other words, stencil range should be determined using tree
        # traversal.

        wrapper = self.lab_wrapper
        assert wrapper.lab_var is not None
        assert len(dijk) <= MAX_DIM

        nonzero = 0
        final_ijk = []
        for dim, (i, di) in enumerate(zip(self.ijk, dijk)):
            if di < wrapper.min[dim]:
                wrapper.min[dim] = di
            if di > wrapper.max[dim]:
                wrapper.max[dim] = di
            if di:
                nonzero += 1
            final_ijk.append(i + di if di >= 0 else i - (-di))
        if nonzero > 1:
            wrapper.tensorial = True

        # Add current position + relative position.
        return wrapper.lab_var.read(*final_ijk)



class LabWrapper(object):
    """Represents BlockLab, while keeping track of the used stencil range."""
    def __init__(self, grid_point):
        self.grid_point = grid_point
        self.min = [0] * MAX_DIM  # Minimum relative position.
        self.max = [0] * MAX_DIM  # Maximum relative position.
        self.tensorial = False    # Diagonal elements?
        # TODO: Components.
        # self.components = [False] * grid_point.count
        self.components = [True] * grid_point.count
        self.lab_var = None

    def get_stencil(self):
        """Generate a StencilInfo instance with respect to the used range and
        grid_point components."""
        selected = [k for k, used in enumerate(self.components) if used]
        return StencilInfo.constructor(
            *self.min,                       # Includes leftmost.
            *[end + 1 for end in self.max],  # Excludes rightmost, add 1.
            self.tensorial,
            len(selected),                   # Used components.
            *selected)



class Grid(Struct):
    """Internal class Grid<block_type>."""
    def __init__(self, block_type):
        ctype, pyname = get_template_names('Grid', 'Grid', block_type)
        super().__init__(ctype=ctype, pyname=pyname,
                         header='Cubism/source/Grid.h')
        self.block_type = block_type
        self.constructor = self.Constructor(Int, Int, Int)



class Cubism(Library):
    def __init__(self, grid_point, block_size=(32, 32, 1), *args,
                 mpi=False, **kwargs):
        if not (1 <= len(block_size) <= 3):
            raise ValueError("1 to 3 dimensions supported, not {}.".format(
                    len(block_size)))
        block_size = tuple(list(block_size) + [1] * (3 - len(block_size)))
        class Meta:
            include_paths = ['../../..']   # This is not good!
            cpp_flags = '-D_BLOCKSIZEX_={} -D_BLOCKSIZEY_={} ' \
                        '-D_BLOCKSIZEZ_={} -D_ALIGNBYTES_={} ' \
                        '-Wno-reorder -Wno-narrowing -Wno-sign-compare ' \
                        '-Wno-unknown-pragmas -Wno-unused-parameter ' \
                        '-Wno-unused-variable -fopenmp'.format(*block_size, 32)
            mpi_enabled = mpi

        super().__init__(*args, Meta=Meta, **kwargs)
        self.grid_point = grid_point
        self.block_size = block_size
        self.DIM = max(k + 1 for k, size in enumerate(block_size) if size > 1)

        self.block_type = Block(grid_point, block_size, self.DIM)
        self.lab_type = BlockLab(self.block_type, self.DIM)
        self.grid_type = Grid(self.block_type)

        _name1 = 'cubism::utils::process_pointwise'
        _name2 = get_template_cname('cubism::utils::process_stencil',
                                    self.lab_type)

        block_ctype = self.block_type.ctype
        self._process_pointwise = methods.Method(
                name=_name1, args=(Auto, Auto),
                header='Cubism/utils/Process.h', library=self)
        self._process_stencil = methods.Method(
                name=_name2, args=(StencilInfo, Auto, Auto, Double),
                header='Cubism/utils/Process.h', library=self)
        self._linear_p2m = methods.Method(
                name='cubism::utils::linear_p2m<{}>'.format(self.DIM),
                args=(self.grid_type, Auto, Auto),
                header='Cubism/utils/P2M.h', library=self)

        self._init_variables()

    def _init_variables(self):
        self.block_num = None  # Not initialized yet.
        self.grid_ptr = value_placeholder(Pointer(self.grid_type))
        self.grid = rerun_on_reuse(dereference(self.grid_ptr))

    def init(self, block_num):
        assert len(block_num) == 3, "Expected 3 values (x, y, z)."
        self.block_num = block_num
        self.grid_ptr.set_value(lazy_global_variable(
                New(self.grid_type, *block_num), 'grid_ptr'))
        return [
            # FIXME: Now, .clear() won't be generated unless it is used. Fix it.
            inline_code("\n/* This is an ugly hack to get the definition "
                        "of .clear() in GridPoint:\n"),
            methods.Method(name='dummy', output=self.grid_point)().clear(),
            inline_code("*/\n"),
        ]

    def _block_foreach(self, func):
        """Create nested for-loops for iterating over each point in a block."""
        def inner(*ijk):
            if len(ijk) == self.DIM:
                return func(*ijk)
            else:
                size = self.block_size[self.DIM - 1 - len(ijk)]
                if size == 1:
                    return inner(0, *ijk)
                name_hint = "i" + "xyz"[self.DIM - 1 - len(ijk)]
                return ForRange(size, counter_type=Int, name_hint=name_hint)(
                        lambda i: inner(i, *ijk))
        return inner()

    def _process__create_loops(self, funcfunc):
        """Helper method for process_pointwise and process_stencil."""
        init = []
        def inner(*ijk):
            result = funcfunc(*ijk)
            if isinstance(result, tuple):
                assert len(result) == 2
                init[:] = result[0]
                return result[1]
            else:
                return result

        foreach = self._block_foreach(inner)
        return InlineWorkflow(init, return_value=foreach,
                              localvars=(init, foreach),
                              name='_process__create_loops')

    def _common_kwargs(self, info, ijk):
        """Prepare and return a set of common kwargs for process_... methods.

        Arguments are:
            info       - BlockInfo const reference.
            ijk        - GridPoint indices in the block
            global_ijk - GridPoint indices in the whole grid.
            xyz        - Absolute coordinates of the GridPoint.
        """
        global_ijk = [self.block_size[dim] * info.index[dim] + ijk[dim] \
                      for dim in range(len(ijk))]
        # TODO: Arithmetic operators between Double and Int.
        xyz = [info.origin[dim] + info.h_gridpoint * Double(ijk[dim]) \
               for dim in range(len(ijk))]
        return {
            'info': info,
            'ijk': ijk,
            'global_ijk': global_ijk,
            'xyz': xyz,
        }

    def process_pointwise(self, func):
        """Call given function for each point in the grid.

        `func` is assumed to be a method (or a callable object), that takes
        one positional argument (GridPoint reference), and an unspecified
        number of keyword arguments. See `._common_kwargs` for the list
        of arguments. Unused keyword arguments should be ignored.

        The output of `func` should be one of the following:
            - FunctionCall, command to put inside the for loops.
            - [...], list of commands to put inside the for loops.
            - [...], [...] - List of commands to put BEFORE the for loops,
                             and a list of commands to put INSIDE the for loops.

        Example:
            def decay(p, **kwargs):
                return expression(p.phi, '*=', cmath.pow(DECAY, dt))
            C.process_pointwise(decay)
        """
        @make_lambda(capture_default='&',
                     _arg_const=(True, False),
                     name_hint=getattr(func, '__name__', None))
        def rhs(info:BlockInfo,
                block:Ref(self.block_type)):
            return self._process__create_loops(
                    lambda *ijk: func(block(*ijk),
                                      **self._common_kwargs(info, ijk)))
        return InlineWorkflow(
                rhs,
                self._process_pointwise(rhs, self.grid),
                localvars=(rhs, ),
                name='process_pointwise',
        )



    def process_stencil(self, func):
        """Call given (stencil) function for each point in the grid.

        `func` is assumed to be a method (or a callable object), that takes
        two positional arguments -- a GridPoint reference and a BlockLab const
        reference -- and an unspecified number of keyword arguments. See
        `._common_kwargs` for the list of arguments. Unused keyword arguments
        should be ignored.

        The output of `func` should be one of the following:
            - FunctionCall, command to put inside the for loops.
            - [...], list of commands to put inside the for loops.
            - [...], [...] - List of commands to put BEFORE the for loops,
                             and a list of commands to put INSIDE the for loops.

        To access a neighboring grid point (ix + dx, iy + dy), where (ix, iy)
        is the location of the current point, and (dx, dy) is the displacement,
        the function should use the lab variable as "lab(dx, dy)".

        `process_stencil` automatically determines the stencil range from these
        function calls (and will determine what GridPoint struct items are
        used, once it's implemented).

        Example:
            def calc_grad(p, lab, info, **kwargs):
                inv2h = .5 / info.h_gridpoint
                # Calc `inv2h` only once per block, and execute these two
                # assignments for each grid point in a block.
                return [inv2h], [
                    expression(p.grad[0], '=', inv2h * (lab(1, 0).phi - lab(-1, 0).phi),
                    expression(p.grad[1], '=', inv2h * (lab(0, 1).phi - lab(0, -1).phi),
                ]

            C.process_stencil(calc_grad)
        """
        lab_wrapper = LabWrapper(self.grid_point)

        @make_lambda(capture_default='&',
                     _arg_const=(True, True, False),
                     name_hint=getattr(func, '__name__', None))
        def rhs(lab:self.lab_type,
                info:BlockInfo,
                block:Ref(self.block_type)):
            lab_wrapper.lab_var = lab
            return self._process__create_loops(
                    lambda *ijk: func(block(*ijk),
                                      LabGetter(lab_wrapper, ijk),
                                      **self._common_kwargs(info, ijk)))

        stencil = lab_wrapper.get_stencil()

        return InlineWorkflow(
                rhs,
                stencil,
                self._process_stencil(stencil, rhs, self.grid, 0.0),
                localvars=(stencil, rhs),
                name='process_stencil',
        )

    def linear_p2m(self, particles, func, **kwargs):
        # TODO: Documentation. Also, this is not core, but contrib (extra).
        assert isinstance(func, methods.FunctionCall), \
               "func should be a FunctionCall, e.g. Lambda." \
               "Make sure it doesn't have any library dependencies, " \
               "link will probably not work."
        return self._linear_p2m(self.grid, particles, func, **kwargs)

    def get_subdomain(self, rank):
        """Cubism's subdomain method.

        Currently, Cubism has a fixed domain of (0, 1) for all three axes.
        """
        # TODO: This should depend on block_num.
        domain_type = domains.types.AABB(Double, 2)
        domain = domain_type(0., 1., 0., 1.)  # TODO: Formalize this.
        return InlineWorkflow(noop(rank),
                              return_value=domain,
                              name='Cubism.get_subdomain')


###############################################################################
# Cubism MPI
###############################################################################

class GridMPI(Struct):
    """Defines GridMPI<Grid>."""
    def __init__(self, grid_type):
        # TODO: Fix headers dependencies (headers of a struct are not captured
        # if struct used only as template arguments.)
        super().__init__(ctype='GridMPI', pyname='GridMPI',
                         template_args=grid_type,
                         headers=['Cubism/source/GridMPI.h',
                                  'Cubism/source/Grid.h',
                                  'Cubism/source/BlockLabMPI.h'])
        # (nodes per x, y, z, blocks within node per x, y, z, maxextent, comm)
        self.constructor = self.Constructor(Int, Int, Int,
                                            Int, Int, Int, Double, mpi.MPI_Comm)



class BlockLabMPI(Struct):
    """Defines BlockLabMPI<Lab>."""
    def __init__(self, lab_type):
        super().__init__(ctype='BlockLabMPI', pyname='BlockLabMPI',
                         template_args=lab_type,
                         header='Cubism/source/BlockLabMPI.h')



class CubismMPI(Cubism):
    def __init__(self, *args, mpi=True, **kwargs):
        super().__init__(*args, mpi=mpi, **kwargs)

        self._node__process_stencil = self._process_stencil
        self._process_stencil = methods.Method(
                name=get_template_cname('cubism::utils::process_stencil_MPI',
                                        self.mpi_lab_type),
                # Stencil, kernel, grid, t = 0..
                args=(StencilInfo, Auto, Auto, Double),
                header='Cubism/utils/ProcessMPI.h',
                library=self)

    def _init_variables(self):
        self.mpi_lab_type = BlockLabMPI(self.lab_type)
        self.mpi_grid_type = GridMPI(self.grid_type)

        self.block_num = None  # Not initialized yet.
        self.grid_ptr = value_placeholder(Pointer(self.mpi_grid_type))
        self.grid = rerun_on_reuse(dereference(self.grid_ptr))

    def init(self, linker, node_dims, bpd):
        self.block_num = bpd
        grid_ptr = New(self.mpi_grid_type, *node_dims, *bpd, 1.0,
                       linker.get_comm(self))
        self.grid_ptr.set_value(lazy_global_variable(grid_ptr, 'grid_ptr'))
        return [
            noop(self.grid_ptr),
            # FIXME: Now, .clear() won't be generated unless it is used. Fix it.
            inline_code("\n/* This is an ugly hack to get the definition "
                        "of .clear() in GridPoint:\n"),
            methods.Method(name='dummy', output=self.grid_point)().clear(),
            inline_code("*/\n"),
        ]
