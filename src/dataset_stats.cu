#include <iostream>
#include <anns_dataset.hpp>
#include <histo/histo.hpp>

#include <omp.h>

#include "utils.hpp"

namespace {
template <class data_t>
void eval_dataset_distribution(
  const data_t* const dataset_ptr,
  const std::size_t size,
  const std::uint32_t dim
  ) {
  std::printf("## %s\n", __func__);
  std::printf("# Input : dataset size = %lu, dim = %u\n", size, dim);

  std::vector<double> norm(size);

#pragma omp parallel for
  for (std::size_t i = 0; i < size; i++) {
    double s = 0;
    for (std::uint32_t j = 0; j < dim; j++) {
      const double v = dataset_ptr[i * dim + j];
      s += v * v;
    }
    norm[i] = std::sqrt(s);
  }

  const auto stat = mtk::get_stat(norm.data(), size);

  std::printf("eval,dim,min,avg,max,q1,q2,q3\n");
  std::printf("%s,%lu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
              __func__,
              dim,
              stat.min,
              stat.avg,
              stat.max,
              stat.q1,
              stat.q2,
              stat.q3
              );
  mtk::histo::print_abs_histogram(norm, 20, 200);
}

template <class data_t>
void eval_dataset(const std::string dataset_path) {
  std::printf("File = %s\n", dataset_path.c_str());

  std::ifstream ifs(dataset_path);
  if (!ifs) {
    std::printf("No such file: %s\n", dataset_path.c_str());
    return;
  }

  const auto [size, dim] = mtk::anns_dataset::load_size_info<std::uint32_t>(dataset_path);

  std::vector<data_t> dataset(size * dim);
  mtk::anns_dataset::load(dataset.data(), dataset_path, true);

  eval_dataset_distribution<data_t>(dataset.data(), size, dim);
}
} // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::printf("Usage: %s [/path/to/dataset] [dtype: float | uint8 | int8]\n", argv[0]);
    return 1;
  }

  const std::string dataset_path = argv[1];
  const std::string dataset_dtype = argv[2];

  if (dataset_dtype == "float") {
    eval_dataset<float>(argv[1]);
  } else if (dataset_dtype == "uint8") {
    eval_dataset<std::uint8_t>(argv[1]);
  } else if (dataset_dtype == "int8") {
    eval_dataset<std::int8_t>(argv[1]);
  } else {
    std::printf("Unsupported dataset data type: %s\n", dataset_dtype.c_str());
  }
}
