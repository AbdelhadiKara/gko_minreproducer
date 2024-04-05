#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>
#include <iostream>

#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/log/batch_logger.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/ginkgo.hpp>
#include <ginkgo/core/base/matrix_data.hpp>

#include <Kokkos_Core.hpp>
/* 
https://github.com/ginkgo-project/ginkgo/blob/develop/test/solver/batch_bicgstab_kernels.cpp
*/
inline std::shared_ptr<gko::Executor> create_default_host_executor()
{
#ifdef KOKKOS_ENABLE_SERIAL
    if constexpr (std::is_same_v<Kokkos::DefaultHostExecutionSpace, Kokkos::Serial>) {
        return gko::ReferenceExecutor::create();
    }
#endif
#ifdef KOKKOS_ENABLE_OPENMP
    if constexpr (std::is_same_v<Kokkos::DefaultHostExecutionSpace, Kokkos::OpenMP>) {
        return gko::OmpExecutor::create();
    }
#endif
}

int main(int argc, char** argv)
{
  using BatchEllMtx=gko::batch::matrix::Ell<double, int>;
      using DenseMtx = gko::batch::matrix::Dense<double>;
  static_assert(sizeof(int)==4);
Kokkos::ScopeGuard scopegard(argc, argv);

    int const batch_size = 1;
    int const mat_size = 5;
    int const non_zero_per_row = 3;
    int const max_iter=1000;
    double const tol=1e-6;

std::vector<gko::matrix_data<double, int>> data;
   
  //  double solution[] = {2. / 3., 1. / 9., 7. / 9., -1. / 9., 2. / 9.};
   // DSpan2D res_span(res_view.data(), batch_size, mat_size);
    static_assert(std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::Cuda>);
        Kokkos::DefaultExecutionSpace exec_space;
        std::shared_ptr const gko_exec = gko::CudaExecutor::
                create(exec_space.cuda_device(),
                       create_default_host_executor(),
                       std::make_shared<gko::CudaAllocator>(),
                       exec_space.cuda_stream());
//std::vector<gko::matrix_data<double, int>> batch_data( batch_size, data);
    std::cerr<<"before ell construction "<<std::endl;

auto batch_matrix_dense = gko::share(
                DenseMtx::
                        create(gko_exec->get_master(),
                               gko::batch_dim<2>(batch_size, gko::dim<2>(mat_size,mat_size))));
  data.emplace_back(gko::matrix_data<vdouble, int>(
        {2, 2}, {{0, 0, 1.0}, {0, 1, 3.0}, {1, 0, 0.0}, {1, 1, 5.0}}));
  
  auto batch_matrix_ell = gko::share(
                BatchEllMtx::
                        create(gko_exec,
                               gko::batch_dim<2>(batch_size, gko::dim<2>(mat_size, mat_size)),
                               non_zero_per_row));

  batch_matrix_ell->create_view_for_item(0)->read(data[0]);
  
  //auto Matvals=batch_matrix_dense->create_view_for_item(0); 
 // auto vals=Matvals->get_values();
 // auto idx=Matvals->get_col_idxs();
  //Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace> values_view(vals, mat_size,  mat_size);
  //Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace> idx_view(idx, mat_size,  mat_size);

    std::cerr<<"before  at "<<std::endl;
   // batch_matrix_dense->at(0,0,1)=1.;
 // batch_matrix_dense->at(0,0,0)=2.;
  
for(int i=1;i<mat_size-1;i++)
{
  batch_matrix_dense->at(0,i,i-1)=1.;
  batch_matrix_dense->at(0,i,i)=2.;
   batch_matrix_dense->at(0,i,i+1)=1.;

}
  batch_matrix_dense->at(0,mat_size-1,mat_size-1)=2.;
  batch_matrix_dense->at(0,mat_size-1,mat_size-2)=1.;


/*
             Kokkos::parallel_for(
            "idx_out_loop",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1),
            KOKKOS_LAMBDA(const int) {
              idx_view(0, 0) = 0;
              idx_view(0, 1) = 1;
              idx_view(0, 2) = -1;

            values_view(0, 0) = 2;
            values_view(0, 1) = 1;
            values_view(0, 2) =0;

            idx_view(mat_size  - 1, 0) = -1;
            idx_view(mat_size  - 1, 1) = mat_size - 2;
            idx_view(mat_size  - 1, 2) = mat_size - 1;

            values_view(mat_size  - 1, 0) = 0;
            values_view(mat_size  - 1, 1) = 1;
            values_view(mat_size  - 1, 2) = 2;

            });

   */
    std::cerr<<"after ell "<<std::endl;
    gko::batch::stop::tolerance_type  tol_type = gko::batch::stop::tolerance_type::absolute;//relative;
      

        auto solver_factory = gko::batch::solver::Bicgstab<double>::build()
                                                 .with_max_iterations(max_iter)
                                                 .with_tolerance(tol)
                                                 .with_tolerance_type(tol_type)
                                                 .on(gko_exec->get_master());
        auto solver = solver_factory->generate(batch_matrix_dense);
        // Create a logger to obtain the iteration counts and "implicit" residual norms for every system after the solve.
        std::shared_ptr<const gko::batch::log::BatchConvergence<double>> logger
                = gko::batch::log::BatchConvergence<double>::create();
        solver->add_logger(logger);
        gko_exec->synchronize();
          std::cerr<<"after factory "<<std::endl;

     Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace> 
        res_view("res_view", batch_size, mat_size);
    Kokkos::deep_copy(res_view,1.);
  //------------------------------------------------------------------------------------------     
        Kokkos::View<double***, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>
                x_view("x_view", batch_size, mat_size, 1);
        auto x=gko::share(
            gko::batch::MultiVector<double>::
                    create(gko_exec,
                           gko::batch_dim<
                                   2>(x_view.extent(0), gko::dim<2>(x_view.extent(1), x_view.extent(2))),
                           gko::array<double>::view(gko_exec, x_view.span(), x_view.data())));
        auto b=gko::share(
          gko::batch::MultiVector<double>::
                    create(gko_exec,
                           gko::batch_dim<
                                   2>(res_view.extent(0), gko::dim<2>(res_view.extent(1),res_view.extent(2))),
                           gko::array<double>::view(gko_exec, res_view.span(), res_view.data())));
          std::cerr<<"before apply"<<std::endl;
 

  solver->apply(b,x);
      Kokkos::fence();

      return 0;
}
