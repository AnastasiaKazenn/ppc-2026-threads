#include "kazennova_a_fox_algorithm/all/include/ops_all.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "kazennova_a_fox_algorithm/common/include/common.hpp"

namespace kazennova_a_fox_algorithm {

namespace {

void GetBlock(const std::vector<double> &mat, int rows, int cols, int block_row, int block_col, int block_size,
              double *block_buf) {
  const int start_row = block_row * block_size;
  const int start_col = block_col * block_size;
  const int end_row = std::min(start_row + block_size, rows);
  const int end_col = std::min(start_col + block_size, cols);

  for (int i = 0; i < block_size; ++i) {
    for (int j = 0; j < block_size; ++j) {
      block_buf[(i * block_size) + j] = 0.0;
    }
  }

  for (int i = start_row; i < end_row; ++i) {
    for (int j = start_col; j < end_col; ++j) {
      block_buf[((i - start_row) * block_size) + (j - start_col)] = mat[(i * cols) + j];
    }
  }
}

}  // namespace

KazennovaATestTaskALL::KazennovaATestTaskALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KazennovaATestTaskALL::ValidationImpl() {
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
  return true;
}

bool KazennovaATestTaskALL::PreProcessingImpl() {
  const auto &in = GetInput();
  auto &out = GetOutput();
  out.rows = in.A.rows;
  out.cols = in.B.cols;
  out.data.assign(static_cast<size_t>(out.rows) * out.cols, 0.0);
  return true;
}

bool KazennovaATestTaskALL::RunImpl() {
  int rank = -1, world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto &in = GetInput();
  auto &out = GetOutput();

  const int m = in.A.rows;
  const int k = in.A.cols;
  const int n = in.B.cols;
  const auto &a = in.A.data;
  const auto &b = in.B.data;
  auto &c = out.data;

  const int bs = kBlockSize;

  const int rows_per_proc = m / world_size;
  const int remainder = m % world_size;
  const int start_row = rank * rows_per_proc + std::min(rank, remainder);
  const int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);

  if (local_rows == 0) {
    if (rank == 0) {
      return true;
    }
    return true;
  }

  std::vector<double> local_c(static_cast<size_t>(local_rows) * n, 0.0);

  const int blocks_i_local = (local_rows + bs - 1) / bs;
  const int blocks_j = (n + bs - 1) / bs;
  const int blocks_k = (k + bs - 1) / bs;

#pragma omp parallel for collapse(2) default(none) \
    shared(a, b, local_c, blocks_i_local, blocks_j, blocks_k, local_rows, n, bs, start_row, k, m)
  for (int bi = 0; bi < blocks_i_local; ++bi) {
    for (int bj = 0; bj < blocks_j; ++bj) {
      std::vector<double> block_a(static_cast<size_t>(bs) * bs);
      std::vector<double> block_b(static_cast<size_t>(bs) * bs);

      for (int bk = 0; bk < blocks_k; ++bk) {
        const int bi_global = (start_row / bs) + bi;
        GetBlock(a, m, k, bi_global, bk, bs, block_a.data());
        GetBlock(b, k, n, bk, bj, bs, block_b.data());

        const int offset = start_row % bs;
        const int max_i = std::min(bs, local_rows - (bi * bs) - offset);
        const int max_j = std::min(bs, n - (bj * bs));
        const int max_k = std::min(bs, k - (bk * bs));

        for (int i = 0; i < max_i; ++i) {
          const int local_row = (bi * bs) + i;
          for (int j = 0; j < max_j; ++j) {
            double sum = 0.0;
            for (int kk = 0; kk < max_k; ++kk) {
              sum += block_a[(i + offset) * bs + kk] * block_b[kk * bs + j];
            }
            local_c[local_row * n + (bj * bs + j)] += sum;
          }
        }
      }
    }
  }

  std::vector<int> recv_counts(world_size, 0);
  std::vector<int> displs(world_size, 0);
  int total_elements = 0;
  for (int r = 0; r < world_size; ++r) {
    const int r_local_rows = rows_per_proc + (r < remainder ? 1 : 0);
    recv_counts[r] = r_local_rows * n;
    displs[r] = total_elements;
    total_elements += recv_counts[r];
  }

  if (rank == 0) {
    std::vector<double> gathered(static_cast<size_t>(total_elements));
    MPI_Gatherv(local_c.data(), static_cast<int>(local_c.size()), MPI_DOUBLE, gathered.data(), recv_counts.data(),
                displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int r = 0; r < world_size; ++r) {
      const int r_start_row = r * rows_per_proc + std::min(r, remainder);
      const int r_local_rows = rows_per_proc + (r < remainder ? 1 : 0);
      for (int i = 0; i < r_local_rows; ++i) {
        for (int j = 0; j < n; ++j) {
          c[(r_start_row + i) * n + j] = gathered[displs[r] + i * n + j];
        }
      }
    }
  } else {
    MPI_Gatherv(local_c.data(), static_cast<int>(local_c.size()), MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool KazennovaATestTaskALL::PostProcessingImpl() {
  return !GetOutput().data.empty();
}

}  // namespace kazennova_a_fox_algorithm
