# cppXplorers : Explorations in C++ for applied sciences

`cppXplorers` is a collection of experiments and examples in modern C++, with a focus on applied sciences and engineering. The project is organized as a mono-repository, where each crate is independent and self-contained. All the sources can be found in the folder `crates` of the [repository](https://github.com/bstaber/cppxplorers/).

The purpose of this repository is to provide a space for exploring numerical methods, algorithms, and patterns in C++ in a practical and modular way. It is not intended as a tutorial or comprehensive learning resource, but rather as a set of working examples and references.

Contributions are welcome. If you spot an issue, find an area that could be improved, or want to add your own example, feel free to open an issue or submit a pull request.

## Organisation and layout

This project has the following layout:

```bash
├── book
│   └── src
└── crates
    ├── kf_linear
    │   ├── include
    │   ├── src
    │   └── tests
    ├── another_example
    │   ├── include
    │   ├── src
    │   └── tests
    ├── another_example
    │   ├── include
    │   ├── src
    │   └── tests
    ├── ...
    │   ├── include
    │   ├── src
    │   └── tests
    └── simple_optimizers
        ├── include
        ├── src
        └── tests
```

* The `crates` folder contains all the examples. I apologize, I've been doing some Rust lately ([rustineers](https://github.com/bstaber/rustineers)).
* Each example has its own headers, sources, and tests.
* The book folder simply contains the sources for generating this book with `mdBook`. 
* Each chapter in the book will explain what's implemented in each example.