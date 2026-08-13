#include <torch/extension.h>

#define DECLARE_QTIP_SPECIALIZATION(NAME) \
  void NAME(torch::Tensor& out, const torch::Tensor& sources, \
      const torch::Tensor& family_block_count, const torch::Tensor& block_experts, \
      const torch::Tensor& block_valid_m, const torch::Tensor& block_route_rows, \
      const torch::Tensor& x, const torch::Tensor& codebook, \
      const torch::Tensor& physical_counters, int64_t specialized_counter_index);

DECLARE_QTIP_SPECIALIZATION(qtip2_k4096_decode_c1)
DECLARE_QTIP_SPECIALIZATION(qtip2_k4096_decode_c2)
DECLARE_QTIP_SPECIALIZATION(qtip2_k4096_decode_c4)
DECLARE_QTIP_SPECIALIZATION(qtip2_k4096_decode_c8)
DECLARE_QTIP_SPECIALIZATION(qtip2_k4096_decode_c16)
DECLARE_QTIP_SPECIALIZATION(qtip2_k4096_prefill_bm16)
DECLARE_QTIP_SPECIALIZATION(qtip2_k4096_prefill_large)
DECLARE_QTIP_SPECIALIZATION(qtip2_k4096_prefill_exact_2k)
DECLARE_QTIP_SPECIALIZATION(qtip2_k4096_prefill_large_8192)
DECLARE_QTIP_SPECIALIZATION(qtip2_k2048_decode_c1)
DECLARE_QTIP_SPECIALIZATION(qtip2_k2048_decode_c2)
DECLARE_QTIP_SPECIALIZATION(qtip2_k2048_decode_c4)
DECLARE_QTIP_SPECIALIZATION(qtip2_k2048_decode_c8)
DECLARE_QTIP_SPECIALIZATION(qtip2_k2048_decode_c16)
DECLARE_QTIP_SPECIALIZATION(qtip2_k2048_prefill_bm16)
DECLARE_QTIP_SPECIALIZATION(qtip2_k2048_prefill_large)
DECLARE_QTIP_SPECIALIZATION(qtip2_k2048_prefill_exact_2k)
DECLARE_QTIP_SPECIALIZATION(qtip2_k2048_prefill_large_8192)
DECLARE_QTIP_SPECIALIZATION(qtip3_k4096_decode_c1)
DECLARE_QTIP_SPECIALIZATION(qtip3_k4096_decode_c2)
DECLARE_QTIP_SPECIALIZATION(qtip3_k4096_decode_c4)
DECLARE_QTIP_SPECIALIZATION(qtip3_k4096_decode_c8)
DECLARE_QTIP_SPECIALIZATION(qtip3_k4096_decode_c16)
DECLARE_QTIP_SPECIALIZATION(qtip3_k4096_prefill_bm16)
DECLARE_QTIP_SPECIALIZATION(qtip3_k4096_prefill_large)
DECLARE_QTIP_SPECIALIZATION(qtip3_k4096_prefill_exact_2k)
DECLARE_QTIP_SPECIALIZATION(qtip3_k4096_prefill_large_8192)
DECLARE_QTIP_SPECIALIZATION(qtip3_k2048_decode_c1)
DECLARE_QTIP_SPECIALIZATION(qtip3_k2048_decode_c2)
DECLARE_QTIP_SPECIALIZATION(qtip3_k2048_decode_c4)
DECLARE_QTIP_SPECIALIZATION(qtip3_k2048_decode_c8)
DECLARE_QTIP_SPECIALIZATION(qtip3_k2048_decode_c16)
DECLARE_QTIP_SPECIALIZATION(qtip3_k2048_prefill_bm16)
DECLARE_QTIP_SPECIALIZATION(qtip3_k2048_prefill_large)
DECLARE_QTIP_SPECIALIZATION(qtip3_k2048_prefill_exact_2k)
DECLARE_QTIP_SPECIALIZATION(qtip3_k2048_prefill_large_8192)

#undef DECLARE_QTIP_SPECIALIZATION

void qtip2_v7_direct(
    torch::Tensor& out, const torch::Tensor& sources,
    const torch::Tensor& family_block_count, const torch::Tensor& block_experts,
    const torch::Tensor& block_valid_m, const torch::Tensor& block_route_rows,
    const torch::Tensor& x, const torch::Tensor& embedded_codebook,
    const torch::Tensor& physical_counters, int64_t variant,
    int64_t specialized_counter_index);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qtip2_v7_direct", &qtip2_v7_direct, "direct packed QTIP2 V7 projection");
  m.def("qtip2_k4096_decode_c1", &qtip2_k4096_decode_c1, "qtip2_k4096_decode_c1");
  m.def("qtip2_k4096_decode_c2", &qtip2_k4096_decode_c2, "qtip2_k4096_decode_c2");
  m.def("qtip2_k4096_decode_c4", &qtip2_k4096_decode_c4, "qtip2_k4096_decode_c4");
  m.def("qtip2_k4096_decode_c8", &qtip2_k4096_decode_c8, "qtip2_k4096_decode_c8");
  m.def("qtip2_k4096_decode_c16", &qtip2_k4096_decode_c16, "qtip2_k4096_decode_c16");
  m.def("qtip2_k4096_prefill_bm16", &qtip2_k4096_prefill_bm16, "qtip2_k4096_prefill_bm16");
  m.def("qtip2_k4096_prefill_large", &qtip2_k4096_prefill_large, "qtip2_k4096_prefill_large");
  m.def("qtip2_k4096_prefill_exact_2k", &qtip2_k4096_prefill_exact_2k, "qtip2_k4096_prefill_exact_2k");
  m.def("qtip2_k4096_prefill_large_8192", &qtip2_k4096_prefill_large_8192, "qtip2_k4096_prefill_large_8192");
  m.def("qtip2_k2048_decode_c1", &qtip2_k2048_decode_c1, "qtip2_k2048_decode_c1");
  m.def("qtip2_k2048_decode_c2", &qtip2_k2048_decode_c2, "qtip2_k2048_decode_c2");
  m.def("qtip2_k2048_decode_c4", &qtip2_k2048_decode_c4, "qtip2_k2048_decode_c4");
  m.def("qtip2_k2048_decode_c8", &qtip2_k2048_decode_c8, "qtip2_k2048_decode_c8");
  m.def("qtip2_k2048_decode_c16", &qtip2_k2048_decode_c16, "qtip2_k2048_decode_c16");
  m.def("qtip2_k2048_prefill_bm16", &qtip2_k2048_prefill_bm16, "qtip2_k2048_prefill_bm16");
  m.def("qtip2_k2048_prefill_large", &qtip2_k2048_prefill_large, "qtip2_k2048_prefill_large");
  m.def("qtip2_k2048_prefill_exact_2k", &qtip2_k2048_prefill_exact_2k, "qtip2_k2048_prefill_exact_2k");
  m.def("qtip2_k2048_prefill_large_8192", &qtip2_k2048_prefill_large_8192, "qtip2_k2048_prefill_large_8192");
  m.def("qtip3_k4096_decode_c1", &qtip3_k4096_decode_c1, "qtip3_k4096_decode_c1");
  m.def("qtip3_k4096_decode_c2", &qtip3_k4096_decode_c2, "qtip3_k4096_decode_c2");
  m.def("qtip3_k4096_decode_c4", &qtip3_k4096_decode_c4, "qtip3_k4096_decode_c4");
  m.def("qtip3_k4096_decode_c8", &qtip3_k4096_decode_c8, "qtip3_k4096_decode_c8");
  m.def("qtip3_k4096_decode_c16", &qtip3_k4096_decode_c16, "qtip3_k4096_decode_c16");
  m.def("qtip3_k4096_prefill_bm16", &qtip3_k4096_prefill_bm16, "qtip3_k4096_prefill_bm16");
  m.def("qtip3_k4096_prefill_large", &qtip3_k4096_prefill_large, "qtip3_k4096_prefill_large");
  m.def("qtip3_k4096_prefill_exact_2k", &qtip3_k4096_prefill_exact_2k, "qtip3_k4096_prefill_exact_2k");
  m.def("qtip3_k4096_prefill_large_8192", &qtip3_k4096_prefill_large_8192, "qtip3_k4096_prefill_large_8192");
  m.def("qtip3_k2048_decode_c1", &qtip3_k2048_decode_c1, "qtip3_k2048_decode_c1");
  m.def("qtip3_k2048_decode_c2", &qtip3_k2048_decode_c2, "qtip3_k2048_decode_c2");
  m.def("qtip3_k2048_decode_c4", &qtip3_k2048_decode_c4, "qtip3_k2048_decode_c4");
  m.def("qtip3_k2048_decode_c8", &qtip3_k2048_decode_c8, "qtip3_k2048_decode_c8");
  m.def("qtip3_k2048_decode_c16", &qtip3_k2048_decode_c16, "qtip3_k2048_decode_c16");
  m.def("qtip3_k2048_prefill_bm16", &qtip3_k2048_prefill_bm16, "qtip3_k2048_prefill_bm16");
  m.def("qtip3_k2048_prefill_large", &qtip3_k2048_prefill_large, "qtip3_k2048_prefill_large");
  m.def("qtip3_k2048_prefill_exact_2k", &qtip3_k2048_prefill_exact_2k, "qtip3_k2048_prefill_exact_2k");
  m.def("qtip3_k2048_prefill_large_8192", &qtip3_k2048_prefill_large_8192, "qtip3_k2048_prefill_large_8192");
}
