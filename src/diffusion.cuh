#pragma once

#include "diffusion.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace diffusion {

template <typename T>
class DiffusionCalculator {
 public:
  DiffusionCalculator(const size_t nx, const size_t ny, const T kappa);
  ~DiffusionCalculator() = default;
  
  DiffusionCalculator(const DiffusionCalculator&) = delete;
  DiffusionCalculator& operator=(const DiffusionCalculator&) = delete;
  DiffusionCalculator(DiffusionCalculator&&) = delete;
  DiffusionCalculator& operator=(DiffusionCalculator&) = delete;

  bool Initialize();
  bool Compute();
  bool Dump();
  bool Finalize();
 private:
  size_t time_;
  size_t nx_;
  size_t ny_;
  size_t n_;
  T c0_;
  T c1_;
  T c2_;
  std::vector<T> f_;
  std::vector<uint8_t> buf_;
  T* D_f_;
  T* D_tmp_;
  bool initialized_;
};

} // namespace culib
