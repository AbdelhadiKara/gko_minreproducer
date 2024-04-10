#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/log/batch_logger.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/ginkgo.hpp>

#include <Kokkos_Core.hpp>
/*
https://github.com/ginkgo-project/ginkgo/blob/develop/test/solver/batch_bicgstab_kernels.cpp
*/
inline std::shared_ptr<gko::Executor> create_default_host_executor() {
#ifdef KOKKOS_ENABLE_SERIAL
  if constexpr (std::is_same_v<Kokkos::DefaultHostExecutionSpace,
                               Kokkos::Serial>) {
    return gko::ReferenceExecutor::create();
  }
#endif
#ifdef KOKKOS_ENABLE_OPENMP
  if constexpr (std::is_same_v<Kokkos::DefaultHostExecutionSpace,
                               Kokkos::OpenMP>) {
    return gko::OmpExecutor::create();
  }
#endif
}

template <typename InputType> auto unbatch(const InputType *batch_object) {
  auto unbatched_mats =
      std::vector<std::unique_ptr<typename InputType::unbatch_type>>{};
  for (int b = 0; b < batch_object->get_num_batch_items(); ++b) {
    unbatched_mats.emplace_back(
        batch_object->create_const_view_for_item(b)->clone());
  }
  return unbatched_mats;
}

int main(int argc, char **argv) {
  using BatchEllMtx = gko::batch::matrix::Ell<double, int>;
  using DenseMtx = gko::batch::matrix::Dense<double>;
  static_assert(sizeof(int) == 4);
  Kokkos::ScopeGuard scopegard(argc, argv);

  int const batch_size = 10;
  int const mat_size = 601;
  int const non_zero_per_row = 3;
  int const max_iter = 1000;
  double const tol = 1e-15;

  std::vector<gko::matrix_data<double, int>> rhs_data{};

  static_assert(std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::Cuda>);
  Kokkos::DefaultExecutionSpace exec_space;
  std::shared_ptr const gko_exec = gko::CudaExecutor::create(
      exec_space.cuda_device(), create_default_host_executor(),
      std::make_shared<gko::CudaAllocator>(), exec_space.cuda_stream());
  // auto gko_exec =
  // gko::ReferenceExecutor::create();//gko_executor->get_master();

  auto batch_matrix_ell = gko::share(BatchEllMtx::create(
      gko_exec, gko::batch_dim<2>(batch_size, gko::dim<2>(mat_size, mat_size)),
      non_zero_per_row));

  auto b_multivec = gko::batch::MultiVector<double>::create(
      gko_exec, gko::batch_dim<2>(batch_size, gko::dim<2>(mat_size, 1)));
  // Read matrix data
  for (int i = 0; i < 1; i++) {
    auto ell_item = batch_matrix_ell->create_view_for_item(i);
    auto rhs_item = b_multivec->create_view_for_item(i);
    std::ifstream ell_stream("../collisionsdata/ell_" + std::to_string(i) +
                             ".mtx");
    std::ifstream rhs_stream("../collisionsdata/rhs" + std::to_string(i) +
                             ".mtx");

    auto ell_buffer = gko::share(
        gko::read<gko::matrix::Ell<double, int>>(ell_stream, gko_exec));
    auto rhs_buffer = gko::read_raw<double>(rhs_stream);
    auto m =
        gko::matrix::Dense<double>::create(gko_exec, gko::dim<2>(mat_size, 1));
    m->read(rhs_buffer);
    m->move_to(rhs_item);
    // gko::write(std::cout,rhs_item);
  }

  gko::batch::stop::tolerance_type tol_type =
      gko::batch::stop::tolerance_type::relative;
  auto solver_factory = gko::batch::solver::Bicgstab<double>::build()
                            .with_max_iterations(max_iter)
                            .with_tolerance(tol)
                            .with_tolerance_type(tol_type)
                            .on(gko_exec);
  auto solver = solver_factory->generate(batch_matrix_ell);
  // Create a logger to obtain the iteration counts and "implicit" residual
  // norms for every system after the solve.
  std::shared_ptr<const gko::batch::log::BatchConvergence<double>> logger =
      gko::batch::log::BatchConvergence<double>::create();
  solver->add_logger(logger);
  gko_exec->synchronize();
  std::cout << "after factory " << std::endl;

  //------------------------------------------------------------------------------------------

 // auto x = gko::batch::MultiVector<double>::create(
   //   gko_exec, gko::batch_dim<2>(batch_size, gko::dim<2>(mat_size, 1)));

  std::cout << "after Xview " << std::endl;
  auto x=b_multivec->clone();
  // x->fill(1.);
  solver->apply(b_multivec, x);

  std::cout << "after solve " << std::endl;

  // allocate and compute the residual
  auto res = gko::batch::MultiVector<double>::create(
      gko_exec, gko::batch_dim<2>(batch_size, gko::dim<2>(mat_size, 1)));
  auto rhs = gko::batch::MultiVector<double>::create(
      gko_exec, gko::batch_dim<2>(batch_size, gko::dim<2>(mat_size, 1)));
  res->copy_from(b_multivec);
  // Compute norm of RHS on the device and automatically copy to host
  auto norm_dim = gko::batch_dim<2>(batch_size, gko::dim<2>(1, 1));
  auto host_b_norm =
      gko::batch::MultiVector<double>::create(gko_exec->get_master(), norm_dim);
  host_b_norm->fill(0.0);
  // allocate and compute residual norm
  auto host_res_norm =
      gko::batch::MultiVector<double>::create(gko_exec->get_master(), norm_dim);
  host_res_norm->fill(0.0);
  b_multivec->compute_norm2(host_b_norm);
  // we need constants on the device
  auto one = gko::batch::MultiVector<double>::create(gko_exec, norm_dim);
  one->fill(1.0);
  auto neg_one = gko::batch::MultiVector<double>::create(gko_exec, norm_dim);
  neg_one->fill(-1.0);
  batch_matrix_ell->apply(one, x, neg_one, res);

  res->compute_norm2(host_res_norm);

  auto host_log_resid = gko::make_temporary_clone(gko_exec->get_master(),
                                                  &logger->get_residual_norm());
  auto host_log_iters = gko::make_temporary_clone(
      gko_exec->get_master(), &logger->get_num_iterations());

  // std::cout << "Residual norm sqrt(r^T r):\n";
  // "unbatch" converts a batch object into a vector of objects of the
  // corresponding single type, eg. batch::matrix::Dense -->
  // std::vector<Dense>.
  auto unb_res = unbatch(host_res_norm.get());
  auto unb_bnorm = unbatch(host_b_norm.get());
  for (int i = 0; i < batch_size; ++i) {

    // Logger  output
    std::cout << " System no. " << i
              << ": residual norm = " << unb_res[i]->at(0, 0)
              << ", implicit residual norm = "
              << host_log_resid->get_const_data()[i]
              << ", iterations = " << host_log_iters->get_const_data()[i]
              << std::endl;
    std::cout << " unbatched bnorm " << unb_bnorm[i]->at(0, 0)
              << " unb residual residual norm " << unb_res[i]->at(0, 0)
              << std::endl;
    const double relresnorm = unb_res[i]->at(0, 0) / unb_bnorm[i]->at(0, 0);
    if (!(relresnorm <= tol)) {
      std::cout << "System " << i << " converged only to " << relresnorm
                << " relative residual." << std::endl;
    }
  }

  return 0;
}
