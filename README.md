`ssmpack` is scalable a C++ library of algorithms and tools for analysis and simulation of State Space Models.
##### Features ######
* a
* b

##### Components ######
* a
* b

## Build ##
### Examples ###
##### Dependencies: #####
1. `cmake >= 2.8`
2. `gcc >= 4.9` or `clang >= 3.4`
3. `sequences` [link](https://github.com/taocpp/sequences)
4. `armadillo >= 7`

##### Build: #####
```
$ cd ssmpack
$ mkdir build
$ cd build
$ cmake ..
$ make [example_name1] [example_name2] ...
```
If no `[example_name]` is given all examples will be built by default.
##### Run example: #####
```
$ example/<example_name>
```

### Tests ###
##### Dependencies: #####
1. `cmake >= 2.8`
2. `gcc >= 4.9` or `clang >= 3.4`
3. `sequences` [link](https://github.com/taocpp/sequences)
4. `armadillo >= 7`
5. `libboost-test-dev >= 1.53`

##### Build: #####
```
$ cd ssmpack
$ mkdir build
$ cd build
$ cmake ..
$ make tests
```
##### Run tests: #####
```
$ test/tests -p
```
or
```
$ ctest test/
```
### Documentation ###
##### Dependencies: #####
1. `doxygen`
2. `texlive` optional for generating diagrams
   * `pdflatex`
   * `tikz`
   * `standalone`
3. `imagemagick` optional for generating diagrams

##### Build: #####
```
$ cd ssmpack
$ mkdir build
$ cd build
$ cmake ..
$ make doc
```
##### Veiw: #####
```
$ xdg-open doc/html/index.html
```
