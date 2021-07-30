#include "osqp_bench.hpp"
#include "smooth_bench.hpp"
#include <gflags/gflags.h>
#include <iomanip>
#include <iostream>
#include <matplot/matplot.h>
#include <sstream>

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

  auto a_p = a.batch.first;
  auto a_v = a.batch.second;
  auto b_p = b.batch.first;
  auto b_v = b.batch.second;

  if (a_v.size() != b_v.size()) { throw std::runtime_error("different bach sizes"); }
  for (auto i = 0u; i != a_v.size(); ++i) {
    if ((a_p[i].qp.u - b_p[i].qp.u).norm() > 1e-6) { throw std::runtime_error("Problem mismatch"); }
    if (FLAGS_verbose) {

      std::cout
        << "--------------------------------------------------------------------------------"
        << std::endl;
    }
    if (a_v[i]) {
      auto a_dur = std::chrono::duration<double>(a_v[i]->dt).count();
      res.total_a_duration += a_dur;
      ++res.num_a;
      if (FLAGS_verbose) {
        std::cout << "A: " << std::endl;
        std::cout << std::setw(20) << "  Time: " << a_dur << std::endl;
        std::cout << std::setw(20) << "  Primal: " << a_v[i]->solution.transpose() << std::endl;
        std::cout << std::setw(20) << "  Obj: " << a_v[i]->objective << std::endl;
      }
    }
    if (b_v[i]) {
      auto b_dur = std::chrono::duration<double>(b_v[i]->dt).count();
      res.total_b_duration += b_dur;
      ++res.num_b;
      if (FLAGS_verbose) {
        std::cout << "B: " << std::endl;
        std::cout << std::setw(20) << "  Time: " << b_dur << std::endl;
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
  // std::cout << std::setw(30) << "N: " << N << std ::endl;
  // std::cout << std::setw(30) << "M: " << M << std ::endl;
  std::cout << std::setw(30) << "Num Problems: " << a_v.size() << std::endl;
  std::cout << std::setw(30) << "Num A: " << res.num_a << std::endl;
  std::cout << std::setw(30) << "Num B: " << res.num_b << std::endl;

  std::cout << std::setw(30)
            << "Avg Duration A : " << res.total_a_duration / std::max((int)res.num_a, 1)
            << std::endl;
  std::cout << std::setw(30)
            << "Avg Duration B : " << res.total_b_duration / std::max((int)res.num_b, 1)
            << std::endl;
  std::cout << std::setw(30)
            << "Avg Duration Ratio : " << res.total_duration_ratio / std::max(1, (int)res.num_valid)
            << std::endl;
  std::cout << std::setw(30) << "Min Duration Ratio: " << res.min_duration_ratio << std::endl;
  std::cout << std::setw(30) << "Max Duration Ratio: " << res.max_duration_ratio << std::endl;

  std::cout << std::setw(30)
            << "Avg Primal Diff: " << res.total_primal_diff / std::max(1, (int)res.num_valid)
            << std::endl;
  std::cout << std::setw(30) << "Min Primal Diff: " << res.min_primal_diff << std::endl;
  std::cout << std::setw(30) << "Max Primal Diff: " << res.max_primal_diff << std::endl;

  std::cout << std::setw(30)
            << "Avg Obj Improvement: " << res.total_obj_impr / std::max(1, (int)res.num_valid)
            << std::endl;
  std::cout << std::setw(30) << "Worst Obj Improvement: " << res.worst_obj_impr << std::endl;
  std::cout << std::setw(30) << "Best Obj Improvement: " << res.best_obj_impr << std::endl;
}

int main(int argc, char ** argv)
{
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  constexpr auto maxN   = 30;
  constexpr auto startN = 2;
  constexpr auto lenN   = maxN - startN;

  constexpr std::array<double, 4> D = {0.15, 0.3, 0.5, 1.};

  using ResultMap =
    std::unordered_map<std::string, std::array<std::array<SuiteResult, lenN>, D.size()>>;
  ResultMap allResults;
  std::array<SuiteResult, lenN> sparseStatic;
  std::array<int, lenN> indexArr;
  static_for<startN, maxN>()([&](auto i) {
    constexpr auto N = i.value;
    constexpr auto M = N;
    for (auto j = 0u; j < D.size(); ++j) {
      double d = D[j];

      std::cout << "#################################################################" << std::endl;
      std::cout << "#################################################################" << std::endl;
      std::cout << "Variables: " << N << std::endl;
      std::cout << "Constraints: " << M << std::endl;
      std::cout << "Density: " << d << std::endl;
      std::cout << "#################################################################" << std::endl;
      std::cout << "#################################################################" << std::endl;
      std::default_random_engine gen1, gen2;
      std::cout << "-----------------------------------------------------------------" << std::endl;
      std::cout << "OSQP" << std::endl;
      std::cout << "-----------------------------------------------------------------" << std::endl;
      srand(1);
      auto osqp_res = BenchSuite<OsqpWrapper, M, N>(gen1, FLAGS_batch, d);
      srand(1);
      std::cout << "-----------------------------------------------------------------" << std::endl;
      std::cout << "SMOOTH" << std::endl;
      std::cout << "-----------------------------------------------------------------" << std::endl;
      auto smooth_res = BenchSuite<SmoothWrapper, M, N>(gen2, FLAGS_batch, d);

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

        indexArr[i - startN] = i;
      }
      std::cout << "#################################################################" << std::endl;
      std::cout << "Sparse vs Static" << std::endl;
      std::cout << "#################################################################" << std::endl;
      compareRuns(
        sparseStatic[i - startN], smooth_res.second["Sparse"], smooth_res.second["Static"]);
    }
  });

  std::array<matplot::axes_handle, D.size()> axVec;
  std::array<std::vector<std::string>, D.size()> names;

  for (auto i = 0u; i < D.size(); ++i) {
    names[i] = std::vector<std::string>{};
    auto h   = matplot::figure();
    axVec[i] = h->current_axes();
    axVec[i]->hold(matplot::on);
    for (auto it = allResults.begin(); it != allResults.end(); ++it) {
      std::array<double, maxN - startN> aAvgDur, bAvgDur;
      std::transform(
        it->second[i].begin(), it->second[i].end(), aAvgDur.begin(), [](SuiteResult s) -> double {
          return s.total_a_duration / std::max(1, (int)s.num_a);
        });
      std::transform(
        it->second[i].begin(), it->second[i].end(), bAvgDur.begin(), [](SuiteResult s) -> double {
          return s.total_b_duration / std::max(1, (int)s.num_b);
        });
      axVec[i]->plot(indexArr, aAvgDur)->line_width(3);
      std::ostringstream ss1, ss2;
      ss1 << "OSQP " << it->first << " " << D[i];
      ss2 << "Smooth " << it->first << " " << D[i];

      names[i].push_back(ss1.str());
      axVec[i]->plot(indexArr, bAvgDur)->line_width(3);
      names[i].push_back(ss2.str());
    }
    axVec[i]->xlabel("Number of Variables");
    axVec[i]->title("");
    axVec[i]->ylabel("Avg Duration (s)");
    axVec[i]->legend(names[i]);
  }
  // matplot::subplot(1, allResults.size(), 1, true);
  matplot::show();
}
