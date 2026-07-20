#include <ATen/Functions.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cstdint>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

extern "C" void launch_adaptive_max_pool3d_grad_float(
    const float* grad_output,
    const int32_t* indices,
    float* grad_input,
    int64_t output_numel,
    int64_t output_spatial,
    int64_t input_spatial,
    aclrtStream stream);

extern "C" void launch_adaptive_max_pool3d_grad_half(
    const at::Half* grad_output,
    const int32_t* indices,
    at::Half* grad_input,
    int64_t output_numel,
    int64_t output_spatial,
    int64_t input_spatial,
    aclrtStream stream);

namespace aten_adaptive_max_pool3d_grad {
namespace {

void run_adaptive_max_pool3d_grad_float(
    const at::Tensor& grad_output,
    const at::Tensor& indices,
    at::Tensor& grad_input,
    int64_t output_spatial,
    int64_t input_spatial) {
  const auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);
  const int64_t output_numel = grad_output.numel();
  auto acl_call = [=, &grad_input]() -> int {
    launch_adaptive_max_pool3d_grad_float(
        grad_output.const_data_ptr<float>(),
        indices.const_data_ptr<int32_t>(),
        grad_input.mutable_data_ptr<float>(),
        output_numel,
        output_spatial,
        input_spatial,
        acl_stream);
    return 0;
  };
  at_npu::native::OpCommand::RunOpApiV2(
      "aten_adaptive_max_pool3d_grad::adaptive_max_pool3d_backward",
      acl_call);
}

void run_adaptive_max_pool3d_grad_half(
    const at::Tensor& grad_output,
    const at::Tensor& indices,
    at::Tensor& grad_input,
    int64_t output_spatial,
    int64_t input_spatial) {
  const auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);
  const int64_t output_numel = grad_output.numel();
  auto acl_call = [=, &grad_input]() -> int {
    launch_adaptive_max_pool3d_grad_half(
        grad_output.const_data_ptr<at::Half>(),
        indices.const_data_ptr<int32_t>(),
        grad_input.mutable_data_ptr<at::Half>(),
        output_numel,
        output_spatial,
        input_spatial,
        acl_stream);
    return 0;
  };
  at_npu::native::OpCommand::RunOpApiV2(
      "aten_adaptive_max_pool3d_grad::adaptive_max_pool3d_backward",
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
      "SIMT v1 requires int32 indices");
  TORCH_CHECK(
      self.size(0) == grad_output.size(0) && self.size(1) == grad_output.size(1),
      "self and grad_output must have the same N and C");
  TORCH_CHECK(
      indices.sizes() == grad_output.sizes(),
      "indices shape must match grad_output shape");

  auto grad_output_contiguous = grad_output.contiguous();
  auto indices_contiguous = indices.contiguous();
  auto grad_input = at::zeros_like(self, self.options(), at::MemoryFormat::Contiguous);

  const int64_t input_spatial = self.size(2) * self.size(3) * self.size(4);
  const int64_t output_spatial =
      grad_output.size(2) * grad_output.size(3) * grad_output.size(4);
  if (grad_output_contiguous.numel() == 0) {
    return grad_input;
  }

  if (self.scalar_type() == at::ScalarType::Float) {
    run_adaptive_max_pool3d_grad_float(
        grad_output_contiguous,
        indices_contiguous,
        grad_input,
        output_spatial,
        input_spatial);
    return grad_input;
  }

  if (self.scalar_type() == at::ScalarType::Half) {
    run_adaptive_max_pool3d_grad_half(
        grad_output_contiguous,
        indices_contiguous,
        grad_input,
        output_spatial,
        input_spatial);
    return grad_input;
  }

  auto grad_output_float = grad_output_contiguous.to(at::kFloat);
  auto grad_input_float = at::zeros(self.sizes(), self.options().dtype(at::kFloat));
  run_adaptive_max_pool3d_grad_float(
      grad_output_float,
      indices_contiguous,
      grad_input_float,
      output_spatial,
      input_spatial);
  return grad_input_float.to(self.scalar_type());
}

} // namespace

TORCH_LIBRARY_FRAGMENT(aten_adaptive_max_pool3d_grad, m) {
  m.def(
      "adaptive_max_pool3d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor");
}

TORCH_LIBRARY_IMPL(aten_adaptive_max_pool3d_grad, PrivateUse1, m) {
  m.impl(
      "adaptive_max_pool3d_backward",
      &adaptive_max_pool3d_backward_privateuse1);
}

} // namespace aten_adaptive_max_pool3d_grad

