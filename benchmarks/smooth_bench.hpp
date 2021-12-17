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

#ifndef BENCHMARK__SMOOTH_BENCH_HPP_
#define BENCHMARK__SMOOTH_BENCH_HPP_

#include <smooth/feedback/qp.hpp>

#include "bench_types.hpp"

template<typename Problem>
struct SmoothWrapper
{
  SmoothWrapper(const Problem & pbm, const smooth::feedback::QPSolverParams & prm)
      : qp_(pbm), prm_(prm)
  {}

  BenchResult operator()() const
  {
    auto t0  = std::chrono::high_resolution_clock::now();
    auto sol = smooth::feedback::solve_qp(qp_, prm_);
    auto t1  = std::chrono::high_resolution_clock::now();

    return {
      .status    = sol.code,
      .dt        = t1 - t0,
      .iter      = sol.iter,
      .solution  = sol.primal,
      .objective = (qp_.P.template selfadjointView<Eigen::Upper>() * (0.5 * sol.primal) + qp_.q)
                     .dot(sol.primal),
    };
  }

  Problem qp_;
  smooth::feedback::QPSolverParams prm_;
};

#endif  // BENCHMARK__SMOOTH_BENCH_HPP_
