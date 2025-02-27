#!/bin/bash                                                                                                                                                                                                        

export PREFIX=$PWD/opt

module purge
module load cpe/24.07
module load craype-x86-trento craype-accel-amd-gfx90a
module load PrgEnv-amd

set -xe

wget https://github.com/ginkgo-project/ginkgo/archive/refs/tags/v1.8.0.tar.gz
tar -xf v1.8.0.tar.gz
rm -rf v1.8.0.tar.gz

cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_C_COMPILER=amdclang \
    -D CMAKE_CXX_COMPILER=amdclang++ \
    -D CMAKE_HIP_ARCHITECTURES=gfx90a \
    -D GINKGO_BUILD_BENCHMARKS=OFF \
    -D GINKGO_BUILD_EXAMPLES=OFF \
    -D GINKGO_BUILD_HIP=ON \
    -D GINKGO_BUILD_OMP=OFF \
    -D GINKGO_BUILD_MPI=OFF \
    -D GINKGO_BUILD_TESTS=ON \
    -S ginkgo-1.8.0 \
    -B build-ginkgo

cmake --build build-ginkgo --parallel 16
cmake --install build-ginkgo --prefix $PREFIX/ginkgo
rm -rf build-ginkgo ginkgo-1.8.0

wget https://github.com/kokkos/kokkos/releases/download/4.5.01/kokkos-4.5.01.tar.gz
tar -xf kokkos-4.5.01.tar.gz
rm -rf kokkos-4.5.01.tar.gz

cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_COMPILER=hipcc \
    -D Kokkos_ENABLE_HIP=ON \
    -D Kokkos_ARCH_AMD_GFX90A=ON \
    -S kokkos-4.5.01 \
    -B build-kokkos

cmake --build build-kokkos --parallel 16
cmake --install build-kokkos --prefix $PREFIX/kokkos
rm -rf build-kokkos kokkos-4.5.01
