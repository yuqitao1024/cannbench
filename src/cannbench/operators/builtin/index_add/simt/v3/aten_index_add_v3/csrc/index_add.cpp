#include <ATen/Functions.h>
#include <ATen/WrapDimUtils.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cstdint>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

extern "C" void launch_index_add_generic_float(
    const int32_t* index,
    const float* source,
    float* output,
    float alpha,
    int64_t index_size,
    int64_t inner_stride,
    int64_t self_dim_size,
    int64_t total_length,
    aclrtStream stream);

extern "C" void launch_index_add_generic_half(
    const int32_t* index,
    const at::Half* source,
    at::Half* output,
    uint16_t alpha,
    int64_t index_size,
    int64_t inner_stride,
    int64_t self_dim_size,
    int64_t total_length,
    aclrtStream stream);

extern "C" void launch_index_add_1d_dim0_float(
    const int32_t* index,
    const float* source,
    float* output,
    float alpha,
    int64_t total_length,
    aclrtStream stream);

extern "C" void launch_index_add_1d_dim0_half(
    const int32_t* index,
    const at::Half* source,
    at::Half* output,
    uint16_t alpha,
    int64_t total_length,
    aclrtStream stream);

extern "C" void launch_index_add_2d_dim0_float(
    const int32_t* index,
    const float* source,
    float* output,
    float alpha,
    int64_t cols,
    int64_t total_length,
    aclrtStream stream);

extern "C" void launch_index_add_2d_dim0_half(
    const int32_t* index,
    const at::Half* source,
    at::Half* output,
    uint16_t alpha,
    int64_t cols,
    int64_t total_length,
    aclrtStream stream);

extern "C" void launch_index_add_3d_dim0_float(
    const int32_t* index,
    const float* source,
    float* output,
    float alpha,
    int64_t slice_size,
    int64_t total_length,
    aclrtStream stream);

extern "C" void launch_index_add_3d_dim0_half(
    const int32_t* index,
    const at::Half* source,
    at::Half* output,
    uint16_t alpha,
    int64_t slice_size,
    int64_t total_length,
    aclrtStream stream);

extern "C" void launch_index_add_4d_dim3_float(
    const int32_t* index,
    const float* source,
    float* output,
    float alpha,
    int64_t self_dim_size,
    int64_t index_size,
    int64_t total_length,
    aclrtStream stream);

extern "C" void launch_index_add_4d_dim3_half(
    const int32_t* index,
    const at::Half* source,
    at::Half* output,
    uint16_t alpha,
    int64_t self_dim_size,
    int64_t index_size,
    int64_t total_length,
    aclrtStream stream);

namespace aten_index_add_v3 {
namespace {

struct IndexAddShape {
  int64_t rank;
  int64_t wrapped_dim;
  int64_t self_dim_size;
  int64_t index_size;
  int64_t inner_stride;
  int64_t outer_size;
  int64_t total_length;
};

IndexAddShape build_shape(
    const at::Tensor& self,
    int64_t wrapped_dim,
    int64_t index_size) {
  int64_t inner_stride = 1;
  for (int64_t d = wrapped_dim + 1; d < self.dim(); ++d) {
    inner_stride *= self.size(d);
  }

  int64_t outer_size = 1;
  for (int64_t d = 0; d < wrapped_dim; ++d) {
    outer_size *= self.size(d);
  }

  return IndexAddShape{
      self.dim(),
      wrapped_dim,
      self.size(wrapped_dim),
      index_size,
      inner_stride,
      outer_size,
      outer_size * index_size * inner_stride,
  };
}

void run_index_add_float(
    const at::Tensor& index,
    const at::Tensor& source,
    at::Tensor& output,
    float alpha,
    const IndexAddShape& shape) {
  const auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);
  auto output_tensor = output;
  auto acl_call = [=]() mutable -> int {
    const auto* index_ptr = index.const_data_ptr<int32_t>();
    const auto* source_ptr = source.const_data_ptr<float>();
    auto* output_ptr = output_tensor.mutable_data_ptr<float>();
    const int64_t rank = shape.rank;
    const int64_t wrapped_dim = shape.wrapped_dim;

    if (rank == 1 && wrapped_dim == 0) {
      launch_index_add_1d_dim0_float(
          index_ptr, source_ptr, output_ptr, alpha, shape.total_length, acl_stream);
      return 0;
    }
    if (rank == 2 && wrapped_dim == 0 && shape.inner_stride <= 256) {
      launch_index_add_2d_dim0_float(
          index_ptr,
          source_ptr,
          output_ptr,
          alpha,
          shape.inner_stride,
          shape.total_length,
          acl_stream);
      return 0;
    }
    if (rank == 3 && wrapped_dim == 0) {
      launch_index_add_3d_dim0_float(
          index_ptr,
          source_ptr,
          output_ptr,
          alpha,
          shape.inner_stride,
          shape.total_length,
          acl_stream);
      return 0;
    }
    if (rank == 4 && wrapped_dim == 3) {
      launch_index_add_4d_dim3_float(
          index_ptr,
          source_ptr,
          output_ptr,
          alpha,
          shape.self_dim_size,
          shape.index_size,
          shape.total_length,
          acl_stream);
      return 0;
    }
    launch_index_add_generic_float(
        index_ptr,
        source_ptr,
        output_ptr,
        alpha,
        shape.index_size,
        shape.inner_stride,
        shape.self_dim_size,
        shape.total_length,
        acl_stream);
    return 0;
  };
  at_npu::native::OpCommand::RunOpApiV2(
      "aten_index_add_v3::index_add_forward",
      acl_call);
}

void run_index_add_half(
    const at::Tensor& index,
    const at::Tensor& source,
    at::Tensor& output,
    at::Half alpha,
    const IndexAddShape& shape) {
  const auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);
  auto output_tensor = output;
  auto acl_call = [=]() mutable -> int {
    const auto* index_ptr = index.const_data_ptr<int32_t>();
    const auto* source_ptr = source.const_data_ptr<at::Half>();
    auto* output_ptr = output_tensor.mutable_data_ptr<at::Half>();
    const int64_t rank = shape.rank;
    const int64_t wrapped_dim = shape.wrapped_dim;

    if (rank == 1 && wrapped_dim == 0) {
      launch_index_add_1d_dim0_half(
          index_ptr, source_ptr, output_ptr, alpha.x, shape.total_length, acl_stream);
      return 0;
    }
    if (rank == 2 && wrapped_dim == 0 && shape.inner_stride <= 256) {
      launch_index_add_2d_dim0_half(
          index_ptr,
          source_ptr,
          output_ptr,
          alpha.x,
          shape.inner_stride,
          shape.total_length,
          acl_stream);
      return 0;
    }
    if (rank == 3 && wrapped_dim == 0) {
      launch_index_add_3d_dim0_half(
          index_ptr,
          source_ptr,
          output_ptr,
          alpha.x,
          shape.inner_stride,
          shape.total_length,
          acl_stream);
      return 0;
    }
    if (rank == 4 && wrapped_dim == 3) {
      launch_index_add_4d_dim3_half(
          index_ptr,
          source_ptr,
          output_ptr,
          alpha.x,
          shape.self_dim_size,
          shape.index_size,
          shape.total_length,
          acl_stream);
      return 0;
    }
    launch_index_add_generic_half(
        index_ptr,
        source_ptr,
        output_ptr,
        alpha.x,
        shape.index_size,
        shape.inner_stride,
        shape.self_dim_size,
        shape.total_length,
        acl_stream);
    return 0;
  };
  at_npu::native::OpCommand::RunOpApiV2(
      "aten_index_add_v3::index_add_forward",
      acl_call);
}

at::Tensor index_add_forward_privateuse1(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source,
    const c10::Scalar& alpha) {
  TORCH_CHECK(
      self.device().type() == at::DeviceType::PrivateUse1,
      "expected PrivateUse1/NPU self tensor");
  TORCH_CHECK(
      index.device().type() == at::DeviceType::PrivateUse1,
      "expected PrivateUse1/NPU index tensor");
  TORCH_CHECK(
      source.device().type() == at::DeviceType::PrivateUse1,
      "expected PrivateUse1/NPU source tensor");
  TORCH_CHECK(self.dim() > 0, "index_add self must have at least one dimension");
  TORCH_CHECK(index.dim() == 1, "index_add index must be one-dimensional");
  TORCH_CHECK(
      source.dim() == self.dim(),
      "index_add source rank must match self rank");
  TORCH_CHECK(
      source.scalar_type() == self.scalar_type(),
      "SIMT index_add source dtype must match self dtype");

  const auto wrapped_dim = at::maybe_wrap_dim(dim, self.dim());
  auto source_contiguous = source.contiguous();
  auto index_int = index.to(at::kInt).contiguous();

  const int64_t index_size = index_int.numel();
  TORCH_CHECK(
      source_contiguous.size(wrapped_dim) == index_size,
      "index_add source dim size must match index size");
  for (int64_t d = 0; d < self.dim(); ++d) {
    if (d == wrapped_dim) {
      continue;
    }
    TORCH_CHECK(
        source_contiguous.size(d) == self.size(d),
        "index_add source shape must match self outside dim");
  }

  const IndexAddShape shape = build_shape(self, wrapped_dim, index_size);
  if (shape.total_length == 0) {
    return self.contiguous().clone();
  }

  if (self.scalar_type() == at::ScalarType::Float) {
    auto compute_self = self.contiguous();
    auto output = compute_self.clone();
    run_index_add_float(
        index_int,
        source_contiguous,
        output,
        alpha.to<float>(),
        shape);
    return output;
  }

  if (self.scalar_type() == at::ScalarType::Half) {
    auto compute_self = self.contiguous();
    auto output = compute_self.clone();
    run_index_add_half(
        index_int,
        source_contiguous,
        output,
        alpha.to<at::Half>(),
        shape);
    return output;
  }

  auto compute_self = self.contiguous().to(at::kFloat);
  auto compute_source = source_contiguous.to(at::kFloat);
  auto output = compute_self.clone();
  run_index_add_float(
      index_int,
      compute_source,
      output,
      alpha.to<float>(),
      shape);
  return output.to(self.scalar_type());
}

} // namespace

TORCH_LIBRARY_FRAGMENT(aten_index_add_v3, m) {
  m.def(
      "index_add_forward(Tensor self, int dim, Tensor index, Tensor source, Scalar alpha=1) -> Tensor");
}

TORCH_LIBRARY_IMPL(aten_index_add_v3, PrivateUse1, m) {
  m.impl("index_add_forward", &index_add_forward_privateuse1);
}

} // namespace aten_index_add_v3
