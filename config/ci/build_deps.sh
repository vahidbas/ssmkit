# Installs build dependencies
apt-get update -qq -y
mkdir ../external_deps

# 1: Armadillo
apt-get install -y cmake,libopenblas-dev,liblapack-dev,libarpack-dev
cd ../external_deps
wget http://sourceforge.net/projects/arma/files/armadillo-7.200.2.tar.xz
tar xf armadillo-7.200.2.tar.xz
cd armadillo-7.200.2
configure
make install
cd ..

# 2: sequences
git clone https://github.com/taocpp/sequences.git
cp sequences/include/tao /usr/local/include
