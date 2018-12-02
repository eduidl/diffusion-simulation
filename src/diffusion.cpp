#include "diffusion.h"
#include "bmp.h"

#include <cmath>
#include <iomanip>
#include <sstream>

namespace diffusion {

template <typename T>
DiffusionCalculatorCPU<T>::DiffusionCalculatorCPU(
    const size_t nx, const size_t ny, const T kappa)
  : time_(0), nx_(nx), ny_(ny), n_(nx * ny),
    f_(nx * ny, 0), tmp_(nx * ny, 0), buf_(nx * ny, 0) {

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
void DiffusionCalculatorCPU<T>::Compute() {
  for (auto jy = (decltype(ny_))0; jy < ny_; ++jy) {
    for (auto jx = (decltype(nx_))0; jx < nx_; ++jx) {
      const auto j = nx_ * jy + jx;
      const auto fcc = f_[j];

      const auto fcw = (jx == 0) ? fcc : f_[j - 1];
      const auto fce = (jx == nx_ - 1) ? fcc : f_[j + 1];

      const auto fcn = (jy == 0) ? fcc : f_[j - nx_];
      const auto fcs = (jy == ny_ - 1) ? fcc : f_[j + nx_];

      tmp_[j] = c0_ * (fce + fcw) + c1_ * (fcn + fcs) + c2_ * fcc;
    }
  }
  f_.swap(tmp_);
  
  time_++;
}

template <typename T>
bool DiffusionCalculatorCPU<T>::Dump() {
  for (auto i = (decltype(n_))0; i < n_; ++i) {
    buf_[i] = static_cast<uint8_t>(std::round(255 * f_[i]));
  }

  std::ostringstream sout;
  sout << "results/" << std::setfill('0') << std::setw(6) << time_ << ".bmp";
  return bmp::WriteBmp(sout.str(), nx_, ny_, buf_.data());
}

template class DiffusionCalculatorCPU<float>;
template class DiffusionCalculatorCPU<double>;

} // namespace diffusion 

