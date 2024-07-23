#include <iostream>
#include <anns_dataset.hpp>
#include <raft/neighbors/cagra_serialize.cuh>
#include <raft/neighbors/cagra.cuh>

template <class data_t, class index_t>
void core(
  const std::string input_path,
  const std::string output_path
  ) {

  raft::device_resources resource_handle;
  auto cagra_index = raft::neighbors::cagra::deserialize<data_t, index_t>(resource_handle, input_path);

  const auto num_graph_elements = cagra_index.graph_degree() * cagra_index.size();
  std::vector<data_t> graph(num_graph_elements);
  cudaMemcpy(graph.data(), cagra_index.graph().data_handle(), sizeof(data_t) * num_graph_elements, cudaMemcpyDefault);

  mtk::anns_dataset::store(output_path, cagra_index.size(), cagra_index.graph_degree(), graph.data(), mtk::anns_dataset::format_t::FORMAT_BIGANN | mtk::anns_dataset::format_t::HEADER_U32, true);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf(
      "Extract a CAGRA graph from a RAFT CAGRA index and save it as an internal CAGRA format\n"
      "Usage : %s [path/to/input/raft/cagra/index] [dataset_dtype: float | int8 | uint8] [path/to/output/graph/file.cagra]\n",
      argv[0]
      );
    return 1;
  }

  const std::string input_raft_index_path = argv[1];
  const std::string dtype = argv[2];
  const std::string output_graph_path = argv[3];

  if (dtype == "float") {
    core<float, std::uint32_t>(input_raft_index_path, output_graph_path);
  } else if (dtype == "int8") {
    core<std::int8_t, std::uint32_t>(input_raft_index_path, output_graph_path);
  } else if (dtype == "uint8") {
    core<std::uint8_t, std::uint32_t>(input_raft_index_path, output_graph_path);
  } else {
    std::printf("Unsupported data type : %s\n", dtype.c_str());
  }
}
