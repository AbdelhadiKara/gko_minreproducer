#include <iostream>
#include <memory>
#include <string>

#include <ginkgo/ginkgo.hpp>

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " nbatch\n";
        return EXIT_FAILURE;
    }
    int const batch_size = std::stoi(argv[1]);
    int const mat_size = 5;
    int const max_iter = 1000;
    int nnz = 1;
    double const tol = 1e-14;

    //     std::shared_ptr const gko_exec = gko::OmpExecutor::create();
    std::shared_ptr const gko_exec = gko::CudaExecutor::create(0, gko::OmpExecutor::create());

    std::shared_ptr batch_matrix = gko::batch::matrix::Dense<
            double>::create(gko_exec, gko::batch_dim<2>(batch_size, gko::dim<2>(mat_size)));

    for (int ibatch = 0; ibatch < batch_size; ++ibatch) {
        gko::matrix_data<double> mtx_data(batch_matrix->get_size().get_common_size());
        for (int i = 0; i < mat_size; ++i) {
            mtx_data.nonzeros.emplace_back(i, i, 2.);
        }
        batch_matrix->create_view_for_item(ibatch)->read(mtx_data);
    }

    std::unique_ptr solver_factory
            = gko::batch::solver::Bicgstab<double>::build()
                      .with_max_iterations(max_iter)
                      .with_tolerance_type(gko::batch::stop::tolerance_type::absolute)
                      .with_tolerance(tol)
                      .with_loggers(gko::batch::log::BatchConvergence<double>::create())
                      .on(gko_exec);
    std::unique_ptr solver = solver_factory->generate(batch_matrix);

    std::shared_ptr x = gko::batch::MultiVector<
            double>::create(gko_exec, gko::batch_dim<2>(batch_size, gko::dim<2>(mat_size, 1)));
    x->fill(0);

    std::shared_ptr b = gko::batch::MultiVector<
            double>::create(gko_exec, gko::batch_dim<2>(batch_size, gko::dim<2>(mat_size, 1)));
    b->fill(2);

    solver->apply(b, x);

    gko_exec->synchronize();

    for (int ibatch = 0; ibatch < batch_size; ++ibatch) {
        gko::write(std::cout, x->create_view_for_item(ibatch));
    }

    return 0;
}
