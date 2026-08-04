#include <torch/extension.h>

#define DECLARE_COMPACT_QTIP(R, K)                                                \
  void decompress_matvec_compact_##R##_##K(                                      \
      torch::Tensor& out, const torch::Tensor& sources,                           \
      const torch::Tensor& family_block_count,                                    \
      const torch::Tensor& block_experts, const torch::Tensor& block_valid_m,      \
      const torch::Tensor& block_route_rows, const torch::Tensor& x,               \
      const torch::Tensor& codebook, const torch::Tensor& physical_counters);

DECLARE_COMPACT_QTIP(2, 4096)
DECLARE_COMPACT_QTIP(3, 4096)
DECLARE_COMPACT_QTIP(2, 2048)
DECLARE_COMPACT_QTIP(3, 2048)

#undef DECLARE_COMPACT_QTIP

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decompress_matvec_compact_2_4096",
        &decompress_matvec_compact_2_4096,
        "Compact QTIP2 packed GEMV, K=4096");
  m.def("decompress_matvec_compact_3_4096",
        &decompress_matvec_compact_3_4096,
        "Compact QTIP3 packed GEMV, K=4096");
  m.def("decompress_matvec_compact_2_2048",
        &decompress_matvec_compact_2_2048,
        "Compact QTIP2 packed GEMV, K=2048");
  m.def("decompress_matvec_compact_3_2048",
        &decompress_matvec_compact_3_2048,
        "Compact QTIP3 packed GEMV, K=2048");
}
