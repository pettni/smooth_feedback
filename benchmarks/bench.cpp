// smooth_feedback: Control theory on Lie groups
// https://github.com/pettni/smooth_feedback
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2021 Petter Nilsson, John B. Mains
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <gflags/gflags.h>
#include <matplot/matplot.h>

#include <iomanip>
#include <iostream>
#include <ranges>
#include <sstream>

#include "osqp_bench.hpp"
#include "smooth_bench.hpp"

DEFINE_uint64(batch, 10, "Size of each batch");
DEFINE_uint64(min_size, 4, "Minimal problem size");
DEFINE_uint64(max_size, 15, "Maximal problem size");
DEFINE_uint64(step_size, 1, "Step size");
DEFINE_uint64(m_multiple, 1, "Number of inequalities as multiple of size");
DEFINE_bool(verbose, false, "Print per problem information");

void compare_results(
  const std::string & a_name,
  const BatchResult & a,
  const std::string & b_name,
  const BatchResult & b)
{
  assert(a.results.size() == b.results.size());

  std::size_t N = a.results.size();

  std::size_t N_optim           = 0;
  double total_optim_duration_a = 0;
  double total_optim_duration_b = 0;

  double total_duration_ratio = 1;
  double min_duration_ratio   = std::numeric_limits<double>::infinity();
  double max_duration_ratio   = 0;

  double total_primal_diff = 0;
  double min_primal_diff   = std::numeric_limits<double>::infinity();
  double max_primal_diff   = 0;

  std::size_t N_pinfeas           = 0;
  double total_pinfeas_duration_a = 0;
  double total_pinfeas_duration_b = 0;

  std::size_t N_dinfeas           = 0;
  double total_dinfeas_duration_a = 0;
  double total_dinfeas_duration_b = 0;

  std::cout << "-----------------------------------------------------------------" << '\n';
  std::cout << a_name << " (A) vs. " << b_name << " (B)" << '\n';
  std::cout << "-----------------------------------------------------------------" << '\n';

  for (auto i = 0u; i != N; ++i) {
    const auto & a_res = a.results[i];
    const auto & b_res = b.results[i];

    double duration_a = std::chrono::duration<double>(a_res.dt).count();
    double duration_b = std::chrono::duration<double>(b_res.dt).count();

    if (FLAGS_verbose) {
      using std::cout, std::setw;
      cout << "Details for problem " << i << '\n';

      cout << setw(30) << a_name + " code " << static_cast<int>(a_res.status) << '\n';
      cout << setw(30) << b_name + " code " << static_cast<int>(b_res.status) << '\n';

      cout << setw(30) << a_name + " duration " << duration_a << '\n';
      cout << setw(30) << b_name + " duration " << duration_b << '\n';

      cout << setw(30) << a_name + " iterations " << a_res.iter << '\n';
      cout << setw(30) << b_name + " iterations " << b_res.iter << '\n';

      cout << setw(30) << a_name + " solution " << a_res.solution.transpose() << '\n';
      cout << setw(30) << b_name + " solution " << b_res.solution.transpose() << '\n';

      cout << setw(30) << a_name + " objective " << a_res.objective << '\n';
      cout << setw(30) << b_name + " objective " << b_res.objective << '\n' << '\n';
    }

    double duration_ratio = duration_a / duration_b;

    if (
      a_res.status == smooth::feedback::QPSolutionStatus::Optimal
      && b_res.status == smooth::feedback::QPSolutionStatus::Optimal) {
      ++N_optim;

      double primal_diff = (b_res.solution - a_res.solution).norm();

      total_optim_duration_a += duration_a;
      total_optim_duration_b += duration_b;

      total_duration_ratio *= duration_ratio;
      min_duration_ratio = std::min(duration_ratio, min_duration_ratio);
      max_duration_ratio = std::max(duration_ratio, max_duration_ratio);

      total_primal_diff += primal_diff;
      min_primal_diff = std::min(primal_diff, min_primal_diff);
      max_primal_diff = std::max(primal_diff, max_primal_diff);
    } else if (
      a_res.status == smooth::feedback::QPSolutionStatus::PrimalInfeasible
      && b_res.status == smooth::feedback::QPSolutionStatus::PrimalInfeasible) {
      ++N_pinfeas;
      total_pinfeas_duration_a += duration_a;
      total_pinfeas_duration_b += duration_b;
    } else if (
      a_res.status == smooth::feedback::QPSolutionStatus::DualInfeasible
      && b_res.status == smooth::feedback::QPSolutionStatus::DualInfeasible) {
      ++N_dinfeas;
      total_dinfeas_duration_a += duration_a;
      total_dinfeas_duration_b += duration_b;
    }
  }

  using std::cout, std::setw;

  cout << setw(30) << "Total number of problems: " << N << '\n';

  cout << setw(30) << a_name + " optimal:" << a.num_optimal << '\n';
  cout << setw(30) << b_name + " optimal:" << b.num_optimal << '\n';
  cout << setw(30) << "both optimal: " << N_optim << '\n';

  cout << setw(30) << a_name + " avg duration: " << total_optim_duration_a / N_optim << '\n';
  cout << setw(30) << b_name + " avg duration: " << total_optim_duration_b / N_optim << '\n';

  cout << setw(30) << "Avg duration ratio " << std::pow(total_duration_ratio, 1. / N_optim) << '\n';
  cout << setw(30) << "Min duration ratio " << min_duration_ratio << '\n';
  cout << setw(30) << "Max duration ratio " << max_duration_ratio << '\n';

  cout << setw(30) << "Avg primal diff " << total_primal_diff / N_optim << '\n';
  cout << setw(30) << "Min primal diff " << min_primal_diff << '\n';
  cout << setw(30) << "Max primal diff " << max_primal_diff << '\n';
}

struct PlotData
{
  double density;
  std::vector<int> size;
  std::vector<double> time_smooth_s, time_smooth_d, time_osqp;
};

int main(int argc, char ** argv)
{
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  constexpr std::array<double, 3> D = {0.05, 0.3, 1.};

  std::default_random_engine rng(5);

  smooth::feedback::QPSolverParams prm{};
  prm.eps_abs  = 1e-6;
  prm.eps_rel  = 1e-6;
  prm.polish   = true;
  prm.max_iter = 10000;
  prm.scaling  = false;
  // prm.verbose  = true;

  std::vector<PlotData> plot_data;

  for (auto density : D) {
    PlotData data;
    data.density = density;
    for (auto n = FLAGS_min_size; n <= FLAGS_max_size; n += FLAGS_step_size) {
      std::size_t m = FLAGS_m_multiple * n;

      std::default_random_engine gen1, gen2;

      // generate sparse problems
      std::vector<smooth::feedback::QuadraticProgram<-1, -1>> qp_list;
      std::vector<smooth::feedback::QuadraticProgramSparse<double>> qp_sparse_list;

      std::generate_n(
        std::back_inserter(qp_list), FLAGS_batch, [&]() { return random_qp(m, n, density, rng); });

      std::transform(
        qp_list.begin(), qp_list.end(), std::back_inserter(qp_sparse_list), [](const auto & qp) {
          return qp_dense_to_sparse(qp);
        });

      auto smooth_d_res =
        solve_batch<SmoothWrapper<smooth::feedback::QuadraticProgram<-1, -1>>>(qp_list, prm);

      auto smooth_s_res =
        solve_batch<SmoothWrapper<smooth::feedback::QuadraticProgramSparse<double>>>(
          qp_sparse_list, prm);

      auto osqp_res = solve_batch<OsqpWrapper<smooth::feedback::QuadraticProgramSparse<double>>>(
        qp_sparse_list, prm);

      std::cout << "#################################################################" << '\n';
      std::cout << "#################################################################" << '\n';
      std::cout << "Variables: " << n << '\n';
      std::cout << "Constraints: " << m << '\n';
      std::cout << "Density: " << density << '\n';
      std::cout << "#################################################################" << '\n';
      std::cout << "#################################################################" << '\n';

      compare_results("smooth-d", smooth_d_res, "osqp", osqp_res);
      compare_results("smooth-s", smooth_s_res, "osqp", osqp_res);

      // calculate average solution times for problems that were solved by all methods
      std::size_t N_tot = 0;

      std::chrono::nanoseconds time_smooth_s(0);
      std::chrono::nanoseconds time_smooth_d(0);
      std::chrono::nanoseconds time_osqp(0);
      for (auto i = 0u; i != FLAGS_batch; ++i) {
        if (
          smooth_d_res.results[i].status == smooth::feedback::QPSolutionStatus::Optimal
          && smooth_s_res.results[i].status == smooth::feedback::QPSolutionStatus::Optimal
          && osqp_res.results[i].status == smooth::feedback::QPSolutionStatus::Optimal) {
          ++N_tot;
          time_smooth_s += smooth_s_res.results[i].dt;
          time_smooth_d += smooth_d_res.results[i].dt;
          time_osqp += osqp_res.results[i].dt;
        }
      }

      data.size.push_back(n);
      data.time_smooth_s.push_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(time_smooth_s).count() / N_tot);
      data.time_smooth_d.push_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(time_smooth_d).count() / N_tot);
      data.time_osqp.push_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(time_osqp).count() / N_tot);
    }

    plot_data.push_back(data);
  };

  auto h = matplot::figure(true);
  matplot::tiledlayout(3, 1);

  for (const PlotData & data : plot_data) {
    matplot::nexttile();
    matplot::hold(matplot::on);
    matplot::plot(data.size, data.time_osqp, "-o")->line_width(3);
    matplot::plot(data.size, data.time_smooth_d, "-o")->line_width(3);
    matplot::plot(data.size, data.time_smooth_s, "-o")->line_width(3);
    matplot::xticks({});
    matplot::xlim({static_cast<double>(FLAGS_min_size), static_cast<double>(FLAGS_max_size)});

    matplot::title("Density " + std::to_string(data.density));
    matplot::ylabel("Avg Duration (s)");
    auto l = matplot::legend({"osqp", "smooth-d", "smooth-s"});
    l->location(matplot::legend::general_alignment::topleft);
  }

  matplot::xticks({static_cast<double>(FLAGS_min_size)});
  matplot::xticks(matplot::automatic);

  h->draw();
  h->size(1920, 1080);
  h->show();
}
