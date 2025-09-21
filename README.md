# cppXplorers : Explorations in C++ for applied sciences

`cppXplorers` is a collection of experiments and examples in modern C++, with a focus on applied sciences and engineering. The project is organized as a mono-repository, where each crate is independent and self-contained.

The purpose of this repository is to provide a space for exploring numerical methods, algorithms, and patterns in C++ in a practical and modular way. It is not intended as a tutorial or comprehensive learning resource, but rather as a set of working examples and references.

Contributions are welcome. If you spot an issue, find an area that could be improved, or want to add your own example, feel free to open an issue or submit a pull request.

# Developper guide

## Spack

We use spack to manage dependencies and build the projects. To get started, clone the repository and run:

```bash
git clone https://github.com/bstaber/cppplorers.git
cd cppXplorers
spack env activate .
spack concretize -f
spack install
```

This will set up the environment and install the necessary packages. You can then enter the environment with:

```bash
spack env activate .
```

Note that we set the CUDA dependency externally, so if you want to use CUDA, make sure to have it installed and configured properly.

## Building

Each crate has its own `CMakeLists.txt` file and can be built independently. However, we provide a top-level `CMakeLists.txt` that can be used to build all crates at once. To build all crates, run:

```bash
just
```

This will run the `justfile` which builds all crates and runs all tests.

## Adding a new crate

To add a new crate, create a new directory under `crates/` and add a `CMakeLists.txt` file. You can use one of the existing crates as a template. Make sure to update the top-level `CMakeLists.txt` to include your new crate.

## Documenting a crate
We use `mdbook` to document each crate. You can find the book source files in the `books/` directory.  To build and serve the book, run:

```bash
just serve-book
```

This will build the book and serve it at `http://localhost:3000`. You can then navigate to the book in your web browser.