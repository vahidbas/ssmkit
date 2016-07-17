#!/bin/bash
# Installs build dependencies
apt-get update -qq -y
mkdir ../external_deps

# 1: Armadillo
apt-get install -y build-essential wget cmake libopenblas-dev liblapack-dev 
cd ../external_deps
wget http://sourceforge.net/projects/arma/files/armadillo-7.200.2.tar.xz
tar xf armadillo-7.200.2.tar.xz
cd armadillo-7.200.2
configure
make install
cd ..

# 2: sequences
git clone https://github.com/taocpp/sequences.git
cp -r sequences/include/tao /usr/local/include
