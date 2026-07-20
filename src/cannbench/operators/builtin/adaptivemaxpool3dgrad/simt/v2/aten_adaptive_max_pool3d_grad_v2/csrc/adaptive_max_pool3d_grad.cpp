#include <ATen/Functions.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cstdint>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

extern "C" void launch_adaptive_max_pool3d_grad_v2_float(
    const float* grad_output,
    const int32_t* indices,
    float* grad_input,
    int64_t n_dim,
    int64_t c_dim,
    int64_t in_d,
    int64_t in_h,
    int64_t in_w,
    int64_t out_d,
    int64_t out_h,
    int64_t out_w,
    bool is_overlap,
    bool deterministic,
    int64_t block_num,
    aclrtStream stream);

extern "C" void launch_adaptive_max_pool3d_grad_v2_half(
    const at::Half* grad_output,
    const int32_t* indices,
    at::Half* grad_input,
    int64_t n_dim,
    int64_t c_dim,
    int64_t in_d,
    int64_t in_h,
    int64_t in_w,
    int64_t out_d,
    int64_t out_h,
    int64_t out_w,
    bool is_overlap,
    bool deterministic,
    int64_t block_num,
    aclrtStream stream);

namespace aten_adaptive_max_pool3d_grad_v2 {
namespace {

int64_t ceil_div(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

bool has_overlapping_windows(const at::Tensor& self, const at::Tensor& grad_output) {
  return self.size(2) % grad_output.size(2) != 0 ||
      self.size(3) % grad_output.size(3) != 0 ||
      self.size(4) % grad_output.size(4) != 0;
}

int64_t block_num_for_shape(
    const at::Tensor& self,
    const at::Tensor& grad_output,
    bool deterministic) {
  constexpr int64_t thread_dim = 512;
  constexpr int64_t max_blocks = 64;
  int64_t blocks = 1;
  if (deterministic) {
    blocks = ceil_div(self.size(0) * self.size(1), thread_dim);
  } else {
    blocks = grad_output.numel();
  }
  if (blocks > max_blocks) {
    blocks = max_blocks;
  }
  if (blocks < 1) {
    blocks = 1;
  }
  return blocks;
}

bool deterministic_requested(bool is_overlap) {
  if (!is_overlap) {
    return false;
  }
  return at::globalContext().deterministicAlgorithms();
}

void run_adaptive_max_pool3d_grad_v2_float(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& indices,
    at::Tensor& grad_input,
    bool is_overlap,
    bool deterministic,
    int64_t block_num) {
  const auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);
  auto grad_input_tensor = grad_input;
  auto acl_call = [=]() -> int {
    launch_adaptive_max_pool3d_grad_v2_float(
        grad_output.const_data_ptr<float>(),
        indices.const_data_ptr<int32_t>(),
        grad_input_tensor.mutable_data_ptr<float>(),
        self.size(0),
        self.size(1),
        self.size(2),
        self.size(3),
        self.size(4),
        grad_output.size(2),
        grad_output.size(3),
        grad_output.size(4),
        is_overlap,
        deterministic,
        block_num,
        acl_stream);
    return 0;
  };
  at_npu::native::OpCommand::RunOpApiV2(
      "aten_adaptive_max_pool3d_grad_v2::adaptive_max_pool3d_backward",
      acl_call);
}

void run_adaptive_max_pool3d_grad_v2_half(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& indices,
    at::Tensor& grad_input,
    bool is_overlap,
    bool deterministic,
    int64_t block_num) {
  const auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);
  auto grad_input_tensor = grad_input;
  auto acl_call = [=]() -> int {
    launch_adaptive_max_pool3d_grad_v2_half(
        grad_output.const_data_ptr<at::Half>(),
        indices.const_data_ptr<int32_t>(),
        grad_input_tensor.mutable_data_ptr<at::Half>(),
        self.size(0),
        self.size(1),
        self.size(2),
        self.size(3),
        self.size(4),
        grad_output.size(2),
        grad_output.size(3),
        grad_output.size(4),
        is_overlap,
        deterministic,
        block_num,
        acl_stream);
    return 0;
  };
  at_npu::native::OpCommand::RunOpApiV2(
      "aten_adaptive_max_pool3d_grad_v2::adaptive_max_pool3d_backward",
      acl_call);
}

at::Tensor adaptive_max_pool3d_backward_privateuse1(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& indices) {
  TORCH_CHECK(
      grad_output.device().type() == at::DeviceType::PrivateUse1,
      "expected PrivateUse1/NPU grad_output tensor");
  TORCH_CHECK(
      self.device().type() == at::DeviceType::PrivateUse1,
      "expected PrivateUse1/NPU self tensor");
  TORCH_CHECK(
      indices.device().type() == at::DeviceType::PrivateUse1,
      "expected PrivateUse1/NPU indices tensor");
  TORCH_CHECK(self.dim() == 5, "self must be 5D NCDHW");
  TORCH_CHECK(grad_output.dim() == 5, "grad_output must be 5D NCDHW");
  TORCH_CHECK(indices.dim() == 5, "indices must be 5D NCDHW");
  TORCH_CHECK(
      self.scalar_type() == grad_output.scalar_type(),
      "self and grad_output dtype must match");
  TORCH_CHECK(
      indices.scalar_type() == at::ScalarType::Int,
      "SIMT v2 requires int32 indices");
  TORCH_CHECK(
      self.size(0) == grad_output.size(0) && self.size(1) == grad_output.size(1),
      "self and grad_output must have the same N and C");
  TORCH_CHECK(
      indices.sizes() == grad_output.sizes(),
      "indices shape must match grad_output shape");

  auto grad_output_contiguous = grad_output.contiguous();
  auto indices_contiguous = indices.contiguous();
  auto grad_input = at::zeros_like(self, self.options(), at::MemoryFormat::Contiguous);
  if (grad_output_contiguous.numel() == 0) {
    return grad_input;
  }

  const bool is_overlap = has_overlapping_windows(self, grad_output);
  const bool deterministic = deterministic_requested(is_overlap);
  const int64_t block_num = block_num_for_shape(self, grad_output, deterministic);

  if (self.scalar_type() == at::ScalarType::Float) {
    run_adaptive_max_pool3d_grad_v2_float(
        grad_output_contiguous,
        self,
        indices_contiguous,
        grad_input,
        is_overlap,
        deterministic,
        block_num);
    return grad_input;
  }

  if (self.scalar_type() == at::ScalarType::Half) {
    run_adaptive_max_pool3d_grad_v2_half(
        grad_output_contiguous,
        self,
        indices_contiguous,
        grad_input,
        is_overlap,
        deterministic,
        block_num);
    return grad_input;
  }

  auto grad_output_float = grad_output_contiguous.to(at::kFloat);
  auto grad_input_float = at::zeros(self.sizes(), self.options().dtype(at::kFloat));
  run_adaptive_max_pool3d_grad_v2_float(
      grad_output_float,
      self,
      indices_contiguous,
      grad_input_float,
      is_overlap,
      deterministic,
      block_num);
  return grad_input_float.to(self.scalar_type());
}

} // namespace

TORCH_LIBRARY_FRAGMENT(aten_adaptive_max_pool3d_grad_v2, m) {
  m.def(
      "adaptive_max_pool3d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor");
}

TORCH_LIBRARY_IMPL(aten_adaptive_max_pool3d_grad_v2, PrivateUse1, m) {
  m.impl(
      "adaptive_max_pool3d_backward",
      &adaptive_max_pool3d_backward_privateuse1);
}

} // namespace aten_adaptive_max_pool3d_grad_v2

