#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace bmp {

bool WriteBmp(const std::string &filename, const size_t width,
              const size_t height, const uint8_t* const img);

} // namespace bmp
