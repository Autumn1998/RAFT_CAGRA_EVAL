#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include <numeric>
#include <histo/histo.hpp>

namespace mtk {
struct stat_t {
  float max, min, avg;
  float q1, q2, q3;
};

template <class T>
stat_t get_stat(
  T* const ptr,
  const std::size_t N
  ) {
  stat_t stat;

  if (N < 3) {
    std::runtime_error("N must be larger or equal to 3");
  }

  std::sort(ptr, ptr + N);

  const auto get_med = [](const T* const ptr, const std::size_t N) -> float {
    if ((N - 1) % 2 != 0) {
      return (ptr[(N - 2) / 2] + ptr[N / 2]) / 2.f;
    } else {
      return ptr[(N - 1) / 2];
    }
  };

  stat.q2 = get_med(ptr, N);
  if (N % 2 == 0) {
    stat.q1 = get_med(ptr              , (N - 1) / 2);
    stat.q3 = get_med(ptr + (N + 1) / 2, (N - 1) / 2);
  } else {
    stat.q1 = get_med(ptr        , N / 2);
    stat.q3 = get_med(ptr + N / 2, N / 2);
  }

  stat.max = ptr[N - 1];
  stat.min = ptr[0];
  stat.avg = std::accumulate(ptr, ptr + N, 0.f, [&](const T a, const float b) {return static_cast<float>(a) + b;}) / N;

  return stat;
}

} // namespace mtk
