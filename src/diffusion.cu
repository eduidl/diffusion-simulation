#include "cuda_runtime.h"
#include "diffusion.cuh"
#include "bmp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>

#define blockDim_x 256
#define blockDim_y 8 

#define CUDA_SAFE_CALL(call) \
{ \
  const cudaError_t error = call; \
  if (error != cudaSuccess) { \
    fprintf(stderr, "[Error]: %s:%d, ", __FILE__, __LINE__); \
    fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
    exit(1); \
  } \
}

namespace diffusion {

template <typename T>
__global__ void diffusion2d_0(T *f, T *f_tmp,
                            const size_t nx, const size_t ny,
                            const T c0, const T c1, const T c2) {
  const auto jx = blockDim.x * blockIdx.x + threadIdx.x;
  const auto jy = blockDim.y * blockIdx.y + threadIdx.y;

  const auto j = nx * jy + jx;
  const auto fcc = f[j];

  const auto fcw = (jx == 0) ? fcc : f[j - 1];
  const auto fce = (jx == nx - 1) ? fcc : f[j + 1];

  const auto fcn = (jy == 0) ? fcc : f[j - nx];
  const auto fcs = (jy == ny - 1) ? fcc : f[j + nx];

  f_tmp[j] = c0 * (fce + fcw) + c1 * (fcn + fcs) + c2 * fcc;
}

template <typename T>
__global__ void diffusion2d(T *f, T *f_tmp,
                            const size_t nx, const size_t ny,
                            const T c0, const T c1, const T c2) {
  __shared__ T fs[blockDim_x + 2];

  const auto jy = blockDim_y * blockIdx.y;
  auto j = nx * jy + blockDim.x * blockIdx.x + threadIdx.x;

  auto f1 = f[j];
  auto f0 = (blockIdx.y == 0) ? f1 : f[j - nx];
  j += nx;
  T f2;

#pragma unroll
  for (auto jy = 0; jy < blockDim_y; jy++) {
    f2 = (blockIdx.y == gridDim.y - 1) ? f1 : f[j];
    fs[threadIdx.x + 1] = f1;
     
    if (threadIdx.x == 0) {
      fs[0] = (blockIdx.x == 0) ? f1 : f[j - nx - 1];
    }

    if (threadIdx.x == blockDim.x - 1) {
      fs[threadIdx.x + 2] = (blockIdx.x == gridDim.x - 1) ? f1 : f[j - nx + 1];
    }

    __syncthreads();

    f_tmp[j - nx] = c0 * (fs[threadIdx.x] + fs[threadIdx.x + 2]) +
                     c1 * (f0 + f2) + c2 * f1;

    j += nx;

    f0 = f1;
    f1 = f2;
  }
}

template <typename T>
DiffusionCalculator<T>::DiffusionCalculator(
    const size_t nx, const size_t ny, const T kappa)
  : time_(0), nx_(nx), ny_(ny), n_(nx * ny),
    f_(nx * ny, 0), buf_(nx * ny, 0), initialized_(false) {

  const T dt = 0.20 / (kappa * std::pow(std::max(nx, ny), 2));
  c0_ = kappa * dt * (std::pow(nx, 2));
  c1_ = kappa * dt * (std::pow(ny, 2));
  c2_ = 1.0 - 2.0 * (c0_ + c1_);

  for (auto jy = (decltype(ny))0; jy < ny_; ++jy) {
    for (auto jx = (decltype(nx))0; jx < nx_; ++jx) {
      const auto j = nx_ * jy + jx;
      const auto x = (jx + 0.5) / nx - 0.5;
      const auto y = (jy + 0.5) / ny - 0.5;

      f_[j] = std::exp(-100.0 * (std::pow(x, 2) + std::pow(y, 2)));
    }
  }
}

template <typename T>
bool DiffusionCalculator<T>::Initialize() {
  if (initialized_) return false;

  const auto f_size = n_ * sizeof(T);
  CUDA_SAFE_CALL(cudaMalloc((void**)&D_f_, f_size));
  CUDA_SAFE_CALL(cudaMalloc((void**)&D_tmp_, f_size));
  CUDA_SAFE_CALL(cudaMemcpy((void*)D_f_, (void*)f_.data(), f_size, cudaMemcpyHostToDevice));

  initialized_ = true;
  return true;
}

template <typename T>
bool DiffusionCalculator<T>::Compute() {
  if (!initialized_) return false;

  dim3 grid(nx_ / blockDim_x, ny_ / blockDim_y, 1);
  dim3 threads(blockDim_x, 1, 1);
  diffusion2d<<<grid, threads>>>(D_f_, D_tmp_, nx_, ny_, c0_, c1_, c2_);
  std::swap(D_f_, D_tmp_);
  
  time_++;
  return true; 
}

template <typename T>
bool DiffusionCalculator<T>::Dump() {
  if (!initialized_) return false;

  const auto size = nx_ * ny_;
  CUDA_SAFE_CALL(cudaMemcpy((void*)f_.data(), (void*)D_f_, size * sizeof(T), cudaMemcpyDeviceToHost));

  for (auto i = (decltype(size))0; i < size; ++i) {
    buf_[i] = static_cast<uint8_t>(std::round(255 * f_[i]));
  }

  std::ostringstream sout;
  sout << "results/" << std::setfill('0') << std::setw(6) << time_ << ".bmp";
  return bmp::WriteBmp(sout.str(), nx_, ny_, buf_.data());
}

template <typename T>
bool DiffusionCalculator<T>::Finalize() {
  if (!initialized_) return false;

  CUDA_SAFE_CALL(cudaFree((void*)D_f_));
  CUDA_SAFE_CALL(cudaFree((void*)D_tmp_));
  initialized_ = false;
  return true;
}

template class DiffusionCalculator<float>;
template class DiffusionCalculator<double>;

} // namespace diffusion 
