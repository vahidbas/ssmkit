# Build #
## Examples ##
Dependencies:
1. `cmake >= 2.8`
2. `gcc >= 4.9` or `clang >= 3.4`
3. `sequences` [link](https://github.com/taocpp/sequences)
4. `armadillo >= 7`

```
$ cd ssmpack
$ mkdir build
$ cd build
$ cmake ..
$ make
```

## Tests ##
Dependencies:
1. `cmake >= 2.8`
2. `gcc >= 4.9` or `clang >= 3.4`
3. `sequences` [link](https://github.com/taocpp/sequences)
4. `armadillo >= 7`
5. `libboost-test-dev >= 1.53`

```
$ cd ssmpack
$ mkdir build
$ cd build
$ cmake ..
$ make tests
```
## Documentation ##

1. `doxygen`
2. `texlive` for generating diagram
   * `pdflatex`
   * `tikz`
   * `standalone`
3. `imagemagick`
