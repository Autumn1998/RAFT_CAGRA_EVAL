#include <iostream>
#include <algorithm>
#include <vector>
#include <numeric>
#include <fstream>
#include <anns_dataset.hpp>
#include <histo/histo.hpp>

#include <omp.h>

namespace {
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


template <class index_t>
void eval_two_hop_nodes(
  const index_t* const graph_ptr,
  const std::size_t size,
  const std::size_t k
  ) {
  std::printf("## %s\n", __func__);
  std::printf("# Input : dataset size = %lu, degree = %lu\n", size, k);
  std::fflush(stdout);

  std::vector<std::uint32_t> num_counted_nodes_list(size);
  std::uint32_t* const num_counted_nodes_list_ptr = num_counted_nodes_list.data();

#pragma omp parallel
  {
    std::vector<index_t> count_node_list(k * k + k);
    index_t* const two_hop_node_list_ptr = count_node_list.data() + k;
    index_t* const one_hop_node_list_ptr = count_node_list.data();

    const auto loop_len = (size + omp_get_num_threads() - 1) / omp_get_num_threads();
    for (std::size_t ii = 0; ii < loop_len; ii++) {
      const auto i = ii * omp_get_num_threads() + omp_get_thread_num();
      if (i < size) {
        for (std::size_t a = 0; a < k; a++) {
          const auto src_node = graph_ptr[i * k + a];
          one_hop_node_list_ptr[a] = src_node;

          auto local_list_ptr = two_hop_node_list_ptr + a * k;
          for (std::size_t b = 0; b < k; b++) {
            local_list_ptr[b] = graph_ptr[src_node * k + b];
          }
        }

        std::sort(count_node_list.begin(), count_node_list.end());
        const auto last_iter = std::unique(count_node_list.begin(), count_node_list.end());
        const auto num_elements = std::distance(count_node_list.begin(), last_iter);
        num_counted_nodes_list_ptr[i] = num_elements;
      }
      if (omp_get_thread_num() == 0 && (i / omp_get_num_threads()) % 1000 == 0) {
        std::printf("[tid=0] %lu / %lu (%3.1f%%)\r",
                    i, size, 100. * i / size
                    );
        std::fflush(stdout);
      }
    }
  }
  std::printf("\n");

  const auto stat = get_stat(num_counted_nodes_list_ptr, size);

  std::printf("eval,degree,min,avg,max,q1,q2,q3\n");
  std::printf("%s,%lu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
              __func__,
              k,
              stat.min,
              stat.avg,
              stat.max,
              stat.q1,
              stat.q2,
              stat.q3
              );
  mtk::histo::print_abs_histogram(num_counted_nodes_list, 20, 200);
}

template <class index_t>
void eval_incoming_count(
  const index_t* const graph_ptr,
  const std::size_t size,
  const std::size_t k
  ) {
  std::printf("## %s\n", __func__);
  std::printf("# Input : dataset size = %lu, degree = %lu\n", size, k);
  std::fflush(stdout);

  std::vector<std::uint32_t> num_counted_nodes_list(size, 0);
  std::uint32_t* const num_counted_nodes_list_ptr = num_counted_nodes_list.data();

#pragma omp parallel for
  for (std::size_t i = 0; i < size * k; i++) {
    const auto v = graph_ptr[i];
#pragma omp atomic
    num_counted_nodes_list[v]++;
  }

  const auto stat = get_stat(num_counted_nodes_list_ptr, size);

  std::printf("eval,degree,min,avg,max,q1,q2,q3\n");
  std::printf("%s,%lu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
              __func__,
              k,
              stat.min,
              stat.avg,
              stat.max,
              stat.q1,
              stat.q2,
              stat.q3
              );
  mtk::histo::print_abs_histogram(num_counted_nodes_list, 20, 200);
}
} // unnamed namespace

void eval_graph(
  const std::string index_path
  ) {
  std::printf("File = %s\n", index_path.c_str());

  std::ifstream ifs(index_path);
  if (!ifs) {
    std::printf("No such file: %s\n", index_path.c_str());
    return;
  }

  const auto [graph_size, graph_degree] = mtk::anns_dataset::load_size_info<std::uint32_t>(index_path);

  std::vector<std::uint32_t> graph(graph_size * graph_degree);
  mtk::anns_dataset::load(graph.data(), index_path, true);

  eval_two_hop_nodes(graph.data(), graph_size, graph_degree);
  eval_incoming_count(graph.data(), graph_size, graph_degree);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("Usage: %s [/path/to/graph/file.cagra]\n", argv[0]);
    return 1;
  }

  eval_graph(argv[1]);
}
