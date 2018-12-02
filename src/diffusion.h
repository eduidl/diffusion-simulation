#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace diffusion {

template <typename T>
class DiffusionCalculatorCPU {
 public:
  DiffusionCalculatorCPU(const size_t nx, const size_t ny, const T kappa);
  ~DiffusionCalculatorCPU() = default;
  
  DiffusionCalculatorCPU(const DiffusionCalculatorCPU&) = delete;
  DiffusionCalculatorCPU& operator=(const DiffusionCalculatorCPU&) = delete;
  DiffusionCalculatorCPU(DiffusionCalculatorCPU&&) = delete;
  DiffusionCalculatorCPU& operator=(DiffusionCalculatorCPU&) = delete;

  void Compute();
  bool Dump();
 private:
  size_t time_;
  size_t nx_;
  size_t ny_;
  size_t n_;
  T c0_;
  T c1_;
  T c2_;
  std::vector<T> f_;
  std::vector<T> tmp_;
  std::vector<uint8_t> buf_;
};

} // namespace culib
