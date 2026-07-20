#include <cassert>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Shape5D {
  int n;
  int c;
  int d;
  int h;
  int w;
};

// 计算某shape的元素总数
static int64_t numel(const Shape5D &shape) {
  return static_cast<int64_t>(shape.n) * shape.c * shape.d * shape.h * shape.w;
}

// 计算某坐标在一维展开时的全局偏移
static int64_t offset5d(const Shape5D &shape, int n, int c, int d, int h, int w) {
  return (((static_cast<int64_t>(n) * shape.c + c) * shape.d + d) * shape.h + h) *
             shape.w +
         w;
}

// 校验参数是非负的
static void check_shape(const Shape5D &shape, const std::string &name) {
  if (shape.n <= 0 || shape.c <= 0 || shape.d <= 0 || shape.h <= 0 || shape.w <= 0) {
    throw std::invalid_argument(name + " must contain only positive dimensions");
  }
}

std::vector<float> adaptive_max_pool3d_grad_cpu(
    const std::vector<float> &grad_output,
    const std::vector<int64_t> &indices,
    const Shape5D &grad_input_shape,
    const Shape5D &grad_output_shape) {
  check_shape(grad_input_shape, "grad_input_shape");
  check_shape(grad_output_shape, "grad_output_shape");

  // input 除去n,c之外的元素总数
  const int64_t input_spatial_size =
      static_cast<int64_t>(grad_input_shape.d) * grad_input_shape.h * grad_input_shape.w;
  const int64_t output_numel = numel(grad_output_shape);
  // 避免实际用的 grad_output 和 给的 shape 匹配不上
  if (static_cast<int64_t>(grad_output.size()) != output_numel) {
    throw std::invalid_argument("grad_output size does not match grad_output_shape");
  }
  // 避免实际用的 indices 和 给的 shape 匹配不上，indices 的大小和 output 的 大小应该是一致的
  if (static_cast<int64_t>(indices.size()) != output_numel) {
    throw std::invalid_argument("indices size does not match grad_output_shape");
  }
  // input 和 output 的 n,c 应该一样大小
  if (grad_input_shape.n != grad_output_shape.n || grad_input_shape.c != grad_output_shape.c) {
    throw std::invalid_argument("grad_input and grad_output must have the same N and C");
  }

  // 初始化 grad_input 全部为 0 
  std::vector<float> grad_input(numel(grad_input_shape), 0.0f);

  for (int n = 0; n < grad_output_shape.n; ++n) {
    for (int c = 0; c < grad_output_shape.c; ++c) {
      for (int od = 0; od < grad_output_shape.d; ++od) {
        for (int oh = 0; oh < grad_output_shape.h; ++oh) {
          for (int ow = 0; ow < grad_output_shape.w; ++ow) {
            const int64_t out_offset = offset5d(grad_output_shape, n, c, od, oh, ow);
            // 要修改的 input 的位置，当然是除去n,c之外的
            const int64_t linear_index = indices[out_offset];

            if (linear_index < 0 || linear_index >= input_spatial_size) {
              throw std::out_of_range("indices contains a value outside the input D/H/W slice");
            }
            
            // 根据 linear_index 反推 id, ih, iw, 已知：linear_index = id * h * w + ih * w + iw;
            const int id = static_cast<int>(linear_index / (grad_input_shape.h * grad_input_shape.w));
            const int rem = static_cast<int>(linear_index % (grad_input_shape.h * grad_input_shape.w));
            const int ih = rem / grad_input_shape.w;
            const int iw = rem % grad_input_shape.w;
            const int64_t in_offset = offset5d(grad_input_shape, n, c, id, ih, iw);
            
            // indices 里面的取值可能重复, 所以反推出来的 id, ih, iw 也可能重复，所以是 += 而不是 ==
            grad_input[in_offset] += grad_output[out_offset];
          }
        }
      }
    }
  }

  return grad_input;
}

static void print_dhw_slice(const std::vector<float> &values, const Shape5D &shape) {
  assert(shape.n == 1 && shape.c == 1);
  for (int d = 0; d < shape.d; ++d) {
    std::cout << "D = " << d << ":\n";
    for (int h = 0; h < shape.h; ++h) {
      std::cout << "  [";
      for (int w = 0; w < shape.w; ++w) {
        if (w != 0) {
          std::cout << ", ";
        }
        std::cout << std::setw(4) << values[offset5d(shape, 0, 0, d, h, w)];
      }
      std::cout << "]\n";
    }
  }
}

static void assert_equal(const std::vector<float> &actual, const std::vector<float> &expected) {
  if (actual != expected) {
    std::cerr << "Actual:\n";
    for (float value : actual) {
      std::cerr << value << " ";
    }
    std::cerr << "\nExpected:\n";
    for (float value : expected) {
      std::cerr << value << " ";
    }
    std::cerr << "\n";
    throw std::runtime_error("unexpected grad_input");
  }
}

int main() {
  {
    std::cout << "Example 1: one output gradient routes to one recorded max position\n";
    const Shape5D grad_input_shape{1, 1, 2, 2, 2};
    const Shape5D grad_output_shape{1, 1, 1, 2, 2};

    const std::vector<float> grad_output{
        10.0f, 20.0f,
        30.0f, 40.0f,
    };
    const std::vector<int64_t> indices{
        0, 5,
        2, 7,
    };

    const std::vector<float> grad_input =
        adaptive_max_pool3d_grad_cpu(grad_output, indices, grad_input_shape, grad_output_shape);

    const std::vector<float> expected{
        10.0f, 0.0f,
        30.0f, 0.0f,
        0.0f, 20.0f,
        0.0f, 40.0f,
    };
    assert_equal(grad_input, expected);

    std::cout << "grad_output = [10, 20, 30, 40]\n";
    std::cout << "indices     = [0, 5, 2, 7]\n";
    std::cout << "grad_input:\n";
    print_dhw_slice(grad_input, grad_input_shape);
    std::cout << "\n";
  }

  {
    std::cout << "Example 2: repeated indices accumulate gradients with +=\n";
    const Shape5D grad_input_shape{1, 1, 2, 2, 2};
    const Shape5D grad_output_shape{1, 1, 1, 1, 3};

    const std::vector<float> grad_output{1.0f, 2.0f, 3.0f};
    const std::vector<int64_t> indices{6, 6, 1};

    const std::vector<float> grad_input =
        adaptive_max_pool3d_grad_cpu(grad_output, indices, grad_input_shape, grad_output_shape);

    const std::vector<float> expected{
        0.0f, 3.0f,
        0.0f, 0.0f,
        0.0f, 0.0f,
        3.0f, 0.0f,
    };
    assert_equal(grad_input, expected);

    std::cout << "grad_output = [1, 2, 3]\n";
    std::cout << "indices     = [6, 6, 1]\n";
    std::cout << "grad_input[6] = 1 + 2, grad_input[1] = 3\n";
    std::cout << "grad_input:\n";
    print_dhw_slice(grad_input, grad_input_shape);
  }

  return 0;
}

