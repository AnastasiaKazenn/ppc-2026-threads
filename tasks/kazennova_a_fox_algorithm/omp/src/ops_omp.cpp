#include "kazennova_a_fox_algorithm/omp/include/ops_omp.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <omp.h>

#include "kazennova_a_fox_algorithm/common/include/common.hpp"

namespace kazennova_a_fox_algorithm {

namespace {

int ChooseBlockSize(int n) {
  int best = 1;
  int sqrt_n = static_cast<int>(std::sqrt(static_cast<double>(n)));

  for (int bs = sqrt_n; bs >= 1; --bs) {
    if (n % bs == 0) {
      best = bs;
      break;
    }
  }
  return best;
}

}  // namespace

KazennovaATestTaskOMP::KazennovaATestTaskOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KazennovaATestTaskOMP::ValidationImpl() {
  const auto &in = GetInput();

  if (in.A.data.empty() || in.B.data.empty()) {
    return false;
  }

  if (in.A.rows <= 0 || in.A.cols <= 0 || in.B.rows <= 0 || in.B.cols <= 0) {
    return false;
  }

  if (in.A.cols != in.B.rows) {
    return false;
  }

  if (in.A.rows != in.A.cols || in.B.rows != in.B.cols || in.A.rows != in.B.rows) {
    return false;
  }

  return true;
}

bool KazennovaATestTaskOMP::PreProcessingImpl() {
  const auto &in = GetInput();

  matrix_size = in.A.rows;
  GetOutput().rows = matrix_size;
  GetOutput().cols = matrix_size;
  GetOutput().data.assign(static_cast<size_t>(matrix_size) * matrix_size, 0.0);

  block_size = ChooseBlockSize(matrix_size);
  block_count = matrix_size / block_size;

  size_t total_blocks = static_cast<size_t>(block_count) * block_count;
  size_t block_elements = static_cast<size_t>(block_size) * block_size;
  a_blocks.assign(total_blocks * block_elements, 0.0);
  b_blocks.assign(total_blocks * block_elements, 0.0);
  c_blocks.assign(total_blocks * block_elements, 0.0);

  #pragma omp parallel for collapse(2)
  for (int bi = 0; bi < block_count; ++bi) {
    for (int bj = 0; bj < block_count; ++bj) {
      int block_offset = ((bi * block_count) + bj) * block_elements;

      for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < block_size; ++j) {
          int src_idx = (((bi * block_size) + i) * matrix_size) + ((bj * block_size) + j);
          int dst_idx = block_offset + (i * block_size) + j;
          a_blocks[dst_idx] = in.A.data[src_idx];
        }
      }
    }
  }

  #pragma omp parallel for collapse(2)
  for (int bi = 0; bi < block_count; ++bi) {
    for (int bj = 0; bj < block_count; ++bj) {
      int block_offset = ((bi * block_count) + bj) * block_elements;

      for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < block_size; ++j) {
          int src_idx = (((bi * block_size) + i) * matrix_size) + ((bj * block_size) + j);
          int dst_idx = block_offset + (i * block_size) + j;
          b_blocks[dst_idx] = in.B.data[src_idx];
        }
      }
    }
  }

  return true;
}

bool KazennovaATestTaskOMP::RunImpl() {
  size_t block_elements = static_cast<size_t>(block_size) * block_size;

  std::ranges::fill(c_blocks, 0.0);

  for (int step = 0; step < block_count; ++step) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < block_count; ++i) {
      for (int j = 0; j < block_count; ++j) {
        int k = (i + step) % block_count;

        int a_idx = ((i * block_count) + k) * block_elements;
        int b_idx = ((k * block_count) + j) * block_elements;
        int c_idx = ((i * block_count) + j) * block_elements;

        for (int ii = 0; ii < block_size; ++ii) {
          for (int kk = 0; kk < block_size; ++kk) {
            double a_val = a_blocks[a_idx + (ii * block_size) + kk];
            for (int jj = 0; jj < block_size; ++jj) {
              c_blocks[c_idx + (ii * block_size) + jj] +=
                  a_val * b_blocks[b_idx + (kk * block_size) + jj];
            }
          }
        }
      }
    }
  }

  return true;
}

bool KazennovaATestTaskOMP::PostProcessingImpl() {
  size_t block_elements = static_cast<size_t>(block_size) * block_size;
  auto &out = GetOutput().data;

  #pragma omp parallel for collapse(2)
  for (int bi = 0; bi < block_count; ++bi) {
    for (int bj = 0; bj < block_count; ++bj) {
      int block_offset = ((bi * block_count) + bj) * block_elements;

      for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < block_size; ++j) {
          int dst_idx = (((bi * block_size) + i) * matrix_size) + ((bj * block_size) + j);
          int src_idx = block_offset + (i * block_size) + j;
          out[dst_idx] = c_blocks[src_idx];
        }
      }
    }
  }

  return true;
}

}  // namespace kazennova_a_fox_algorithm