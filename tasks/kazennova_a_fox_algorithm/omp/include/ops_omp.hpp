#pragma once

#include "kazennova_a_fox_algorithm/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kazennova_a_fox_algorithm {

class KazennovaATestTaskOMP : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kOMP;
  }
  explicit KazennovaATestTaskOMP(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int matrix_size;
  int block_size;
  int block_count;
  std::vector<double> a_blocks;
  std::vector<double> b_blocks;
  std::vector<double> c_blocks;
};

}  // namespace kazennova_a_fox_algorithm