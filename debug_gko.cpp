#include <algorithm>
#include <cmath>
#include <memory>


#include <ginkgo/ginkgo.hpp>

#include <Kokkos_Core.hpp>

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
Kokkos::ScopeGuard scopegard(argc, argv);

    int const batch_size = 1;
    int const mat_size = 5;
    int const non_zero_per_row = 3;
    int const max_iter=1000;
    double const tol=1e-6;


    Kokkos::LayoutStride values_layout(
            batch_size,
            non_zero_per_row * mat_size,
            mat_size,
            1,
            non_zero_per_row,
            mat_size);
/*
    Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::DefaultHostExecutionSpace>
            values_host("values_host",  mat_size* non_zero_per_row);
    Kokkos::View<int*, Kokkos::LayoutRight, Kokkos::DefaultHostExecutionSpace>
            idx_host("idx_host", mat_size* non_zero_per_row);
    
    Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::DefaultHostExecutionSpace>
            res_host(res, batch_size, mat_size); */

   // Kokkos::deep_copy(res_view, res_host);

   
    std::cout << " after factorize " << std ::endl;
  //  double solution[] = {2. / 3., 1. / 9., 7. / 9., -1. / 9., 2. / 9.};
   // DSpan2D res_span(res_view.data(), batch_size, mat_size);
  // 
   /*
    if(std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::Serial>)
  std::shared_ptr const gko_exec =gko::ReferenceExecutor::create();
    
    */
Kokkos::DefaultExecutionSpace exec_space;
std::shared_ptr const gko_exec = gko::CudaExecutor::
                create(exec_space.cuda_device(),
                       create_default_host_executor(),
                       std::make_shared<gko::CudaAllocator>(),
                       exec_space.cuda_stream());
    
                      
                      

    Kokkos::View<double***, Kokkos::LayoutStride, Kokkos::DefaultExecutionSpace>
            values_view("values_view", values_layout);

    Kokkos::View<int**, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>
           idx_view("idx_view", mat_size,non_zero_per_row);
  

  Kokkos::parallel_for(
            "idx_inner_loop",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(1, mat_size-1),
            KOKKOS_LAMBDA(const int i) {
                idx_view(i, 0) = i-1 ;
                idx_view(i, 1) = i;
                idx_view(i, 2) =  i+1;
                values_view(0,i,0) =1;
                values_view(0,i,1) =2;
                values_view(0,i,2) =1;

            });

             Kokkos::parallel_for(
            "idx_out_loop",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1),
            KOKKOS_LAMBDA(const int) {
              idx_view(0, 0) = 0;
              idx_view(0, 1) = 1;
              idx_view(0, 2) = -1;

            values_view(0,0, 0) = 2;
            values_view(0,0, 1) = 1;
            values_view(0,0, 2) =0;


              idx_view(mat_size  - 1, 0) = -1;
              idx_view(mat_size  - 1, 1) = mat_size - 2;
              idx_view(mat_size  - 1, 2) = mat_size - 1;
            values_view(0,mat_size  - 1, 0) = 0;
            values_view(0,mat_size  - 1, 1) = 1;
            values_view(0,mat_size  - 1, 2) = 2;

            });
    Kokkos::fence();
    std::cout<<"before ell construction "<<std::endl;

auto gko_values=gko::array<double>::view (gko_exec, values_view.span(), values_view.data());
 auto gko_idx=gko::array<int> ::view (gko_exec, idx_view.span(), idx_view.data());
     std::shared_ptr<gko::batch::matrix::Ell<double, int> >  batch_matrix_ell = gko::share(gko::batch::matrix::Ell<double>::create(
                gko_exec,
                gko::batch_dim<2>(batch_size, 
                gko::dim<2>(mat_size, mat_size)),
                non_zero_per_row,
            gko_values,gko_idx
           ));
    std::cout<<"after ell "<<std::endl;

    gko::batch::stop::tolerance_type  tol_type = gko::batch::stop::tolerance_type::absolute;//relative;
      

        auto solver_factory = gko::batch::solver::Bicgstab<double>::build()
                                                 .with_max_iterations(max_iter)
                                                 .with_tolerance(tol)
                                                 .with_tolerance_type(tol_type)
                                                 .on(gko_exec);
        auto solver = solver_factory->generate(batch_matrix_ell);
        // Create a logger to obtain the iteration counts and "implicit" residual norms for every system after the solve.
      //  std::shared_ptr<const gko::batch::log::BatchConvergence<double>> logger
       //         = gko::batch::log::BatchConvergence<double>::create();
      //  solver->add_logger(logger);
        gko_exec->synchronize();

  //------------------------------------------------------------------------------------------     
        Kokkos::View<double***, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>
                x_view("x_view", batch_size, mat_size, 1);
        Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace> 
        res_view("res_view", batch_size, mat_size);
      Kokkos::deep_copy(res_view,1.);
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

     solver->apply(b,x);
      return 0;
}
