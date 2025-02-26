#include <algorithm>
#include <cmath>
#include <ginkgo/ginkgo.hpp>
#include <memory>

#include <Kokkos_Core.hpp>

inline void write_log(std::fstream &log_file, int const batch_index,
                      int const num_iterations, double const implicit_res_norm,
                      double const true_res_norm, double const b_norm,
                      double const tol) {
  log_file << " System no. " << batch_index << ":" << std::endl;
  log_file << " Number of iterations = " << num_iterations << std::endl;
  log_file << " Implicit residual norm = " << implicit_res_norm << std::endl;
  log_file << " True (Ax-b) residual norm = " << true_res_norm << std::endl;
  log_file << " Right-hand side (b) norm = " << b_norm << std::endl;
  if (!(true_res_norm <= tol)) {
    log_file << " --- System " << batch_index << " did not converge! ---"
             << std::endl;
  }
  log_file << "------------------------------------------------" << std::endl;
}

void save_logger(
    std::fstream &log_file,
    std::shared_ptr<gko::matrix::Csr<double, int>> batch_matrix,
    Kokkos::View<double *, Kokkos::LayoutRight,
                 Kokkos::DefaultExecutionSpace> const x_view,
    Kokkos::View<double *, Kokkos::LayoutRight,
                 Kokkos::DefaultExecutionSpace> const b_view,
    std::shared_ptr<const gko::batch::log::BatchConvergence<double>> logger,
    double const tol) {
  std::shared_ptr const gko_exec = batch_matrix->get_executor();
  int const batch_size = 2;
  int const mat_size = 4;

  auto x = gko::matrix::Dense<double>::create(
      gko_exec, gko::dim<2>(x_view.extent(0), 1),
      gko::array<double>::view(gko_exec, x_view.span(), x_view.data()),
      x_view.stride_0());
  auto b = gko::matrix::Dense<double>::create(
      gko_exec, gko::dim<2>(b_view.extent(0), 1),
      gko::array<double>::view(gko_exec, b_view.span(), b_view.data()),
      b_view.stride_0());

  // allocate the residual
  auto res =gko::matrix::Dense<double>::create(gko_exec, gko::dim<2>(b_view.extent(0), 1));

  res->copy_from(b);

gko::dim<2> norm_dim(gko::dim<2>(1, 1));
    // allocate rhs norm on host.
    auto b_norm_host = gko::matrix::Dense<double>::create(gko_exec->get_master(), norm_dim);
    b_norm_host->fill(0.0);
    // allocate the residual norm on host.
    auto res_norm_host = gko::matrix::Dense<double>::create(gko_exec->get_master(), norm_dim);
    res_norm_host->fill(0.0);
    // compute rhs norm.
    b->compute_norm2(b_norm_host);
    // we need constants on the device
    auto one = gko::matrix::Dense<double>::create(gko_exec, norm_dim);
    one->fill(1.0);
    auto neg_one = gko::matrix::Dense<double>::create(gko_exec, norm_dim);
    neg_one->fill(-1.0);

  
  // to estimate the "true" residual, the apply function below computes Ax-res,
  // and stores the result in res.
  batch_matrix->apply(one, x, neg_one, res);
  // compute residual norm.
  res->compute_norm2(res_norm_host);

  auto log_iters_host = logger->get_num_iterations();
  auto log_resid_host =logger->get_residual_norm();
      

 // for (int i = 0; i < batch_size; ++i) {
    write_log(log_file, 0, log_iters_host.get_data()[0],
              log_resid_host.get_data()[0],
              res_norm_host->at(0, 0),
              b_norm_host->at(0, 0), tol);
  //}

}


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

inline std::shared_ptr<gko::Executor> create_hip_executor() {
  if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::HIP>) {
    return gko::HipExecutor::create(0, gko::ReferenceExecutor::create());
  }
}

int main(int argc, char **argv) {
  Kokkos::ScopeGuard scopegard(argc, argv);

  int const batch_size = 2;
  int const mat_size = 4;
  int const non_zero_per_system = 4;
  double values[] = {2.0, 4.0, 6.0, 8.0, 3.0, 5.0, 7.0, 9.0};
  int col_idxs[] = {0, 1, 2, 3};
  int nnz_per_row[] = {0, 1, 2, 3, 4};
  // rhs and solution
  double res[] = {0, 3.5, 2., 3., 42., 17., 0.5, 1.};
  double solution[] = {1. / 2., 1. / 4., 1. / 6., 1. / 8.,
                       1. / 3., 1. / 5., 1. / 7., 1. / 9.};

  Kokkos::View<double **, Kokkos::LayoutRight,
               Kokkos::DefaultHostExecutionSpace>
      values_view_host(values, batch_size, non_zero_per_system);
  Kokkos::View<int *, Kokkos::LayoutRight, Kokkos::DefaultHostExecutionSpace>
      idx_view_host(col_idxs, non_zero_per_system);
  Kokkos::View<int *, Kokkos::LayoutRight, Kokkos::DefaultHostExecutionSpace>
      nnz_per_row_view_host(nnz_per_row, mat_size + 1);
  Kokkos::View<double **, Kokkos::LayoutRight,
               Kokkos::DefaultHostExecutionSpace>
      res_view_host(res, batch_size, mat_size);
  Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::DefaultHostExecutionSpace>
      solution_view_host(solution, batch_size * mat_size);

  Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>
      values_view("values", batch_size, non_zero_per_system);
  Kokkos::View<int *, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>
      idx_view("col_idxs", non_zero_per_system);
  Kokkos::View<int *, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>
      nnz_per_row_view("nnz_per_row", mat_size + 1);
  Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>
      res_view("res", batch_size, mat_size);
  Kokkos::View<double **, Kokkos::LayoutRight,
               Kokkos::DefaultHostExecutionSpace>
      res_host("res_host", batch_size, mat_size);

  Kokkos::deep_copy(values_view, values_view_host);
  Kokkos::deep_copy(idx_view, idx_view_host);
  Kokkos::deep_copy(nnz_per_row_view, nnz_per_row_view_host);
  Kokkos::deep_copy(res_view, 1.);

  if (std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::Serial>)
    std::shared_ptr const gko_exec = gko::ReferenceExecutor::create();

  Kokkos::DefaultExecutionSpace exec_space;
  std::shared_ptr const gko_exec = gko::HipExecutor::create(
      exec_space.hip_device(), create_hip_executor(),
      std::make_shared<gko::HipAllocator>(), exec_space.hip_stream());

  auto gko_values = gko::array<double>::view(gko_exec, values_view.span(),
                                             values_view.data());
  auto gko_idx =
      gko::array<int>::view(gko_exec, idx_view.span(), idx_view.data());

  auto gko_nnz_per_row = gko::array<int>::view(
      gko_exec, nnz_per_row_view.span(), nnz_per_row_view.data());

  std::shared_ptr<gko::batch::matrix::Csr<double, int>> batch_matrix_csr =
      gko::share(gko::batch::matrix::Csr<double, int>::create(
          gko_exec,
          gko::batch_dim<2>(batch_size, gko::dim<2>(mat_size, mat_size)),
          gko_values, gko_idx, gko_nnz_per_row));
  std::cout << "after csr " << std::endl;

  int const max_iter = 1000;
  double const tol = 1e-12;
  gko::batch::stop::tolerance_type tol_type =
      gko::batch::stop::tolerance_type::absolute; // relative;

  auto solver_factory = gko::batch::solver::Cg<double>::build()
                            .with_max_iterations(max_iter)
                            .with_tolerance(tol)
                            .with_tolerance_type(tol_type)
                            .on(gko_exec);
  auto solver = solver_factory->generate(batch_matrix_csr);
  // Create a logger to obtain the iteration counts and "implicit"
  // residual norms for every system after the solve.
  std::shared_ptr<const gko::batch::log::BatchConvergence<double>> logger =
      gko::batch::log::BatchConvergence<double>::create();
  solver->add_logger(logger);
  gko_exec->synchronize();

  //------------------------------------------------------------------------------------------
  Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>
      x_view("x_view", batch_size, mat_size);

  auto x = gko::share(gko::batch::MultiVector<double>::create(
      gko_exec,
      gko::batch_dim<2>(x_view.extent(0),
                        gko::dim<2>(x_view.extent(1), x_view.extent(2))),
      gko::array<double>::view(gko_exec, x_view.span(), x_view.data())));
  auto b = gko::share(gko::batch::MultiVector<double>::create(
      gko_exec,
      gko::batch_dim<2>(res_view.extent(0),
                        gko::dim<2>(res_view.extent(1), res_view.extent(2))),
      gko::array<double>::view(gko_exec, res_view.span(), res_view.data())));

  solver->apply(b, x);
  
  std::fstream log_file("csr_log.txt", std::ios::out | std::ios::app);
                  save_logger(log_file, batch_matrix_csr->create_view_for_item(0),
  Kokkos::subview(x_view,0,Kokkos::ALL),
  Kokkos::subview(res_view,0,Kokkos::ALL), logger, tol); log_file.close();

  return 0;
}
