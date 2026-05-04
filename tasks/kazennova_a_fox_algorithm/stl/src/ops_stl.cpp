#include "kazennova_a_fox_algorithm/stl/include/ops_stl.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

#include "util/include/util.hpp"

namespace kazennova_a_fox_algorithm {

KazennovaATestTaskSTL::KazennovaATestTaskSTL(const InType& in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KazennovaATestTaskSTL::ValidationImpl() {
  const auto& in = GetInput();
  if (in.A.data.empty() || in.B.data.empty()) return false;
  if (in.A.rows <= 0 || in.A.cols <= 0 || in.B.rows <= 0 || in.B.cols <= 0) return false;
  if (in.A.cols != in.B.rows) return false;
  return true;
}

bool KazennovaATestTaskSTL::PreProcessingImpl() {
  const auto& in = GetInput();
  auto& out = GetOutput();
  out.rows = in.A.rows;
  out.cols = in.B.cols;
  out.data.assign(static_cast<size_t>(out.rows) * out.cols, 0.0);
  return true;
}

bool KazennovaATestTaskSTL::RunImpl() {
  const auto& in = GetInput();
  auto& out = GetOutput();

  const int M = in.A.rows;
  const int K = in.A.cols;
  const int N = in.B.cols;
  const auto& a = in.A.data;
  const auto& b = in.B.data;
  auto& c = out.data;

  const int BS = BLOCK_SIZE;

  const int blocks_i = (M + BS - 1) / BS;
  const int blocks_j = (N + BS - 1) / BS;
  const int blocks_k = (K + BS - 1) / BS;

  auto get_block = [BS](const std::vector<double>& mat, int rows, int cols, int block_row, int block_col,
                        double* block_buf) {
    int start_row = block_row * BS;
    int start_col = block_col * BS;
    int end_row = std::min(start_row + BS, rows);
    int end_col = std::min(start_col + BS, cols);

    for (int i = 0; i < BS; ++i) {
      for (int j = 0; j < BS; ++j) {
        block_buf[i * BS + j] = 0.0;
      }
    }
    for (int i = start_row; i < end_row; ++i) {
      for (int j = start_col; j < end_col; ++j) {
        block_buf[(i - start_row) * BS + (j - start_col)] = mat[i * cols + j];
      }
    }
  };

  int num_threads = ppc::util::GetNumThreads();
  if (num_threads <= 0) num_threads = std::thread::hardware_concurrency();
  if (num_threads <= 0) num_threads = 2;

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  std::atomic<size_t> next_block_idx(0);
  size_t total_blocks = static_cast<size_t>(blocks_i) * blocks_j;

  auto worker = [&]() {
    std::vector<double> block_a(BS * BS);
    std::vector<double> block_b(BS * BS);

    while (true) {
      size_t idx = next_block_idx.fetch_add(1);
      if (idx >= total_blocks) break;

      int bi = static_cast<int>(idx / blocks_j);
      int bj = static_cast<int>(idx % blocks_j);

      for (int bk = 0; bk < blocks_k; ++bk) {
        get_block(a, M, K, bi, bk, block_a.data());
        get_block(b, K, N, bk, bj, block_b.data());

        int block_rows_i = std::min(BS, M - bi * BS);
        int block_cols_j = std::min(BS, N - bj * BS);
        int block_inner_k = std::min(BS, K - bk * BS);

        for (int i = 0; i < block_rows_i; ++i) {
          for (int j = 0; j < block_cols_j; ++j) {
            double sum = 0.0;
            for (int k = 0; k < block_inner_k; ++k) {
              sum += block_a[i * BS + k] * block_b[k * BS + j];
            }
            c[(bi * BS + i) * N + (bj * BS + j)] += sum;
          }
        }
      }
    }
  };

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back(worker);
  }
  for (auto& thr : threads) {
    thr.join();
  }

  return true;
}

bool KazennovaATestTaskSTL::PostProcessingImpl() {
  return !GetOutput().data.empty();
}

}  // namespace kazennova_a_fox_algorithm