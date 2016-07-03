Installation {#install}
======================

`ssmkit` uses `cmake >= 2.8.8` build system.
`ssmkit` is a header-only library. It can be installed without build:

    $ cd ssmkit
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make install

For using `ssmkit` following packages should also be installed:
1. `armadillo >= 7`
2. `sequences` [link](https://github.com/taocpp/sequences)

`ssmkit` uses `C++11/14` features that requires following compiler versions:
* `gcc >= 4.9`
* `clang >= 3.4`


Build Documentation
-------------------
Additional dependencies:
1. `doxygen`
2. `texlive` optional for generating diagrams
   * `pdflatex`
   * `tikz`
   * `standalone`
3. `imagemagick` optional for generating diagrams


    $ cd ssmkit
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make doc

#### Opening Doc
 
    $ xdg-open doc/html/index.html


Build Tests
-----------
Additional dependencies:
1. `libboost-test-dev >= 1.53`


    $ cd ssmkit
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make tests

#### Running tests

    $ test/tests -p

Build Examples
--------------

    $ cd ssmkit
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make [example_name1] [example_name2] ...
or

    $ make examples

#### Running examples

    $ example/<example_name>

Build Benchmarks
----------------
Additional dependencies:
1. `google/benchmark` [link](https://github.com/google/benchmark)
2. `OpenCV` optional for speed comparison


    $ cd ssmkit
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make bm

#### Running benchmarks

    $ example/bm
