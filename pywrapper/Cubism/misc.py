from coupling import StdVector
from coupling.methods.operators import assign
from coupling.workflows.decorators import make_method
from coupling.workflows.linear import InlineWorkflow
from coupling.workflows.loops import ForRange
from coupling.contrib.cpp.cstdio import fopen, fprintf, fclose
from coupling.contrib.cpp.string import StdString

def map_to_vector(cubism, func):
    """Run `func` foreach grid point and save results into a single vector."""

    class G(object):  # Container for the vector.
        vector = None

    assert cubism.block_num, ".init() not yet called?"

    nx = cubism.block_size[0] * cubism.block_num[0]
    ny = cubism.block_size[1] * cubism.block_num[1]

    def inner(p, info, ijk, global_ijk, **kwargs):
        result = func(p, info)

        # We found out what the value_type is, declare the array now.
        G.vector = StdVector(result.method.output)(const=False)

        assert cubism.block_size[2] == 1, "Sorry, 3d not implemented."

        index = global_ijk[1] * nx + global_ijk[0]
        return assign(G.vector[index], result)

    foreach = cubism.process_pointwise(inner)
    return InlineWorkflow(
        G.vector.resize(nx * ny),
        foreach,
        return_value=G.vector,
        localvars=G.vector,
        name='Cubism.extra.map_to_vector',
    )


# TODO: Make some decorator for this cache thing. It's popping out all over the
# place.
_func_cache = {}
def save_to_txt_file(cubism, field_name, filename):
    """Function generator for save-to-txt."""
    assert cubism.block_size[2] == 1, "Not implemented for 3D."

    key = cubism, field_name
    if key in _func_cache:
        return _func_cache[key](filename)

    # TODO: ConstPointer(Char)
    @make_method
    def _save_to_txt_file(filename:StdString):
        def inner(p, info):
            return getattr(p, field_name)

        vector = map_to_vector(cubism, inner)

        f = fopen(filename.c_str(), '"w"')
        assert cubism.block_num, ".init() not yet called?"
        nx = cubism.block_size[0] * cubism.block_num[0]
        ny = cubism.block_size[1] * cubism.block_num[1]

        return InlineWorkflow(
            vector,
            f,
            ForRange(nx)(lambda iy:
                ForRange(ny)(lambda ix:
                    fprintf(f, '"%lf "', vector[iy * nx + ix]),
                ),
                fprintf(f, r'"\n"'),
            ),
            fclose(f),
            localvars=(vector, f),
            name='Cubism.extra.save_to_txt_file',
        )

    _func_cache[key] = _save_to_txt_file
    return _save_to_txt_file(filename)
