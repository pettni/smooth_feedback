// Copyright (C) 2022 Petter Nilsson. MIT License.

#include <ranges>
#include <vector>

/// @brief Range to std::vector
const auto r2v = [](std::ranges::range auto && r) {
  std::vector<std::ranges::range_value_t<decltype(r)>> ret;
  for (auto x : r) { ret.push_back(x); }
  return ret;
};
