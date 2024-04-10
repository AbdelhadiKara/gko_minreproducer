
compil on v100 architecture with kokkos, cuda activated.
cmake -DCMAKE_CXX_COMPILER=Yourpathtonvcc_wrapper /gyselalibxx/vendor/kokkos/bin/nvcc_wrapper -DKokkos_ARCH_VOLTA70=ON -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=OFF -DKokkos_ENABLE_CUDA=ON   -DKokkos_ENABLE_CUDA_CONSTEXPR=ON  -DKokkos_ENABLE_CUDA_LAMBDA=ON -DCMAKE_CXX_STANDARD=17 -DCMAKE_BUILD_TYPE=Debug  ..
