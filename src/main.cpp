#include "diffusion.cuh"

#include <cassert>

int main() {
  diffusion::DiffusionCalculator<double> calculator(1024, 1024, 0.1);
  assert(calculator.Initialize());
  assert(calculator.Dump());
  for (auto i = 0; i < 100000; ++i) {
    assert(calculator.Compute());
  }
  assert(calculator.Finalize());
}
