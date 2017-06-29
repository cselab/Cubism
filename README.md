# Cubism -- Uniform block-structured grid library

Cubism is a library for uniform block-structured grids, for both single-node
and multi-node usage.


## Repository structure

- `source` - The core of the library (C++ code).
- `pywrapper` - Python wrapper for the [coupling framework][1].

# Running Python examples.

To run the Python example (based on the coupling framework), do the following
steps:

- Install the coupling framework, as described [here][1].
- Go to `pywrapper/example/<example_name>`.
- Run `./example.py 1`. The `1` stands for the number of MPI nodes.

[1]: https://gitlab.ethz.ch/mavt-cse/lugano
