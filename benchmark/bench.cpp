#include <gflags/gflags.h>
#include <matplot/matplot.h>

#include <iomanip>
#include <iostream>
#include <ranges>
#include <sstream>

#include "osqp_bench.hpp"
#include "smooth_bench.hpp"

DEFINE_uint64(batch, 10, "Size of each batch");
DEFINE_bool(verbose, false, "Print per problem information");

struct SuiteResult
{
  std::size_t num_total       = 0;
  std::size_t num_valid       = 0;
  std::size_t num_a           = 0;
  std::size_t num_b           = 0;
  double total_a_duration     = 0.;
  double total_b_duration     = 0.;
  double total_duration_ratio = 1.;
  double min_duration_ratio   = std::numeric_limits<double>::infinity();
  double max_duration_ratio   = 0.;
  double total_primal_diff    = 0.;
  double total_obj_impr       = 0.;
  double min_primal_diff      = std::numeric_limits<double>::infinity();
  double max_primal_diff      = 0.;
  double worst_obj_impr       = std::numeric_limits<double>::infinity();
  double best_obj_impr        = 0.;
};

void compareRuns(SuiteResult & res, const BatchResult & a, const BatchResult & b)
{
  const auto & [a_p, a_v] = a.batch;
  const auto & [b_p, b_v] = b.batch;

  if (a_v.size() != b_v.size()) { throw std::runtime_error("different bach sizes"); }

  for (auto i = 0u; i != a_v.size(); ++i) {
    if ((a_p[i].qp.u - b_p[i].qp.u).norm() > 1e-6) { throw std::runtime_error("Problem mismatch"); }
    if (FLAGS_verbose) {
      std::cout << "-----------------------------------------------------------------" << std::endl;
    }

    if (a_v[i].has_value()) {
      auto a_dur = std::chrono::duration<double>(a_v[i]->dt).count();
      res.total_a_duration += a_dur;
      ++res.num_a;
      if (FLAGS_verbose) {
        std::cout << "A: " << std::endl;
        std::cout << std::setw(20) << "  Time: " << a_dur << std::endl;
        std::cout << std::setw(20) << "  Iter: " << a_v[i]->iter << std::endl;
        std::cout << std::setw(20) << "  Primal: " << a_v[i]->solution.transpose() << std::endl;
        std::cout << std::setw(20) << "  Obj: " << a_v[i]->objective << std::endl;
      }
    }

    if (b_v[i].has_value()) {
      auto b_dur = std::chrono::duration<double>(b_v[i]->dt).count();
      res.total_b_duration += b_dur;
      ++res.num_b;

      if (FLAGS_verbose) {
        std::cout << "B: " << std::endl;
        std::cout << std::setw(20) << "  Time: " << b_dur << std::endl;
        std::cout << std::setw(20) << "  Iter: " << b_v[i]->iter << std::endl;
        std::cout << std::setw(20) << "  Primal: " << b_v[i]->solution.transpose() << std::endl;
        std::cout << std::setw(20) << "  Obj: " << b_v[i]->objective << std::endl;
      }
    }
    if (b_v[i] && a_v[i]) {
      ++res.num_valid;

      double primal_diff = (b_v[i]->solution - a_v[i]->solution).norm();
      double obj_improvement =
        (a_v[i]->objective - b_v[i]->objective) / std::max(abs(a_v[i]->objective), 1e-3);
      double duration_ratio = std::chrono::duration<double>(b_v[i]->dt).count()
                            / std::chrono::duration<double>(a_v[i]->dt).count();

      if (FLAGS_verbose) {
        std::cout << std::setw(20) << "Primal Difference: " << primal_diff << std::endl;
        std::cout << std::setw(20) << "Object Ratio: " << obj_improvement << std::endl;
        std::cout << std::setw(20) << "Time Difference: " << duration_ratio << std::endl;
      }

      res.total_duration_ratio += duration_ratio;
      res.min_duration_ratio = std::min(duration_ratio, res.min_duration_ratio);
      res.max_duration_ratio = std::max(duration_ratio, res.max_duration_ratio);
      res.total_primal_diff += primal_diff;
      res.min_primal_diff = std::min(res.min_primal_diff, primal_diff);
      res.max_primal_diff = std::max(primal_diff, res.max_primal_diff);
      res.total_obj_impr += obj_improvement;
      res.worst_obj_impr = std::min(res.worst_obj_impr, obj_improvement);
      res.best_obj_impr  = std::max(res.best_obj_impr, obj_improvement);
    }
  }

  using std::cout, std::setw, std::endl;

  // clang-format off
  cout << setw(30) << "Num Problems"          << a_v.size() << endl;
  cout << setw(30) << "Num A"                 << res.num_a << endl;
  cout << setw(30) << "Num B"                 << res.num_b << endl;

  cout << setw(30) << "Avg Duration A"        << res.total_a_duration / std::max<int>(res.num_a, 1) << endl;
  cout << setw(30) << "Avg Duration B"        << res.total_b_duration / std::max<int>(res.num_b, 1) << endl;
  cout << setw(30) << "Avg Duration Ratio"    << res.total_duration_ratio / std::max<int>(1, res.num_valid) << endl;
  cout << setw(30) << "Min Duration Ratio"    << res.min_duration_ratio << endl;
  cout << setw(30) << "Max Duration Ratio"    << res.max_duration_ratio << endl;

  cout << setw(30) << "Avg Primal Diff"       << res.total_primal_diff / std::max<int>(1, res.num_valid) << endl;
  cout << setw(30) << "Min Primal Diff"       << res.min_primal_diff << endl;
  cout << setw(30) << "Max Primal Diff"       << res.max_primal_diff << endl;

  cout << setw(30) << "Avg Obj Improvement"   << res.total_obj_impr / std::max<int>(1, res.num_valid) << endl;
  cout << setw(30) << "Worst Obj Improvement" << res.worst_obj_impr << endl;
  cout << setw(30) << "Best Obj Improvement"  << res.best_obj_impr << endl;
  // clang-format onf
}

int main(int argc, char ** argv)
{
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  constexpr auto startN = 4;
  constexpr auto maxN   = 31;
  constexpr auto lenN   = maxN - startN;

  constexpr std::array<double, 3> D = {0.15, 0.5, 1.};

  using ResultMap =
    std::unordered_map<std::string, std::array<std::array<SuiteResult, lenN>, D.size()>>;

  ResultMap allResults;
  static_for<startN, maxN>()([&](auto i) {
    constexpr auto N = i.value;
    constexpr auto M = N;

    for (auto j = 0u; j < D.size(); ++j) {
      std::default_random_engine gen1, gen2;

      std::cout << "#################################################################" << std::endl;
      std::cout << "#################################################################" << std::endl;
      std::cout << "Variables: " << N << std::endl;
      std::cout << "Constraints: " << M << std::endl;
      std::cout << "Density: " << D[j] << std::endl;
      std::cout << "#################################################################" << std::endl;
      std::cout << "#################################################################" << std::endl;

      std::cout << "-----------------------------------------------------------------" << std::endl;
      std::cout << "OSQP" << std::endl;
      std::cout << "-----------------------------------------------------------------" << std::endl;

      std::srand(1);
      auto osqp_res = BenchSuite<OsqpWrapper, M, N>(gen1, FLAGS_batch, D[j]);

      std::cout << "-----------------------------------------------------------------" << std::endl;
      std::cout << "SMOOTH" << std::endl;
      std::cout << "-----------------------------------------------------------------" << std::endl;

      std::srand(1);
      auto smooth_res = BenchSuite<SmoothWrapper, M, N>(gen2, FLAGS_batch, D[j]);

      auto keys = osqp_res.first;
      for (auto & k : keys) {
        if (allResults.find(k) == allResults.end()) {
          allResults.emplace(
            std::make_pair(k, std::array<std::array<SuiteResult, lenN>, D.size()>{}));
          allResults[k][j] = std::array<SuiteResult, lenN>{};
        }
        std::cout << "###############################################################" << std::endl;
        std::cout << k << std::endl;
        std::cout << "###############################################################" << std::endl;
        auto osqp_k   = osqp_res.second[k];
        auto smooth_k = smooth_res.second[k];
        compareRuns(allResults[k][j][i - startN], osqp_k, smooth_k);
      }
    }
  });

  auto idxView = std::views::iota(startN, maxN);
  std::vector<int> idxVec(idxView.begin(), idxView.end());

  for (auto i = 0u; i < D.size(); ++i) {
    auto h  = matplot::figure();
    auto ax = h->current_axes();
    ax->hold(matplot::on);

    std::vector<std::string> legends;

    for (const auto & [name, result] : allResults) {
      std::array<double, lenN> aAvgDur, bAvgDur;
      std::transform(
        result[i].begin(), result[i].end(), aAvgDur.begin(), [](const SuiteResult & s) -> double {
          return s.total_a_duration / std::max<int>(1, s.num_a);
        });
      std::transform(
        result[i].begin(), result[i].end(), bAvgDur.begin(), [](const SuiteResult & s) -> double {
          return s.total_b_duration / std::max<int>(1, s.num_b);
        });

      ax->plot(idxVec, aAvgDur)->line_width(3);
      legends.push_back("OSQP " + name);

      ax->plot(idxVec, bAvgDur)->line_width(3);
      legends.push_back("Smooth " + name);
    }

    ax->xlabel("Number of Variables");
    ax->title("Density " + std::to_string(D[i]));
    ax->ylabel("Avg Duration (s)");
    ax->legend(legends);
  }

  matplot::show();
}
