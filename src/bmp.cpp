#include "bmp.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#define FILEHEADERSIZE 14 //ファイルヘッダのサイズ
#define INFOHEADERSIZE 40 //情報ヘッダのサイズ
#define HEADERSIZE (FILEHEADERSIZE+INFOHEADERSIZE)

namespace bmp {

bool WriteBmp(const std::string &filename, const size_t width,
              const size_t height, const uint8_t* const img) {
  std::ofstream ofs(filename, std::ios::binary); 
  if (ofs.fail()) {
    std::cout << "Fail to open file: " << filename << std::endl;
    return false;
  }

  std::vector<uint8_t> header_buf(HEADERSIZE, 0); //ヘッダを格納する
  const auto real_width = width * 3 + width % 4;
  std::vector<uint8_t> bmp_line(real_width); //画像1行分のRGB情報を格納する

  //ここからヘッダ作成
  const uint32_t file_size = height * real_width + HEADERSIZE;
  const uint32_t offset_to_data = HEADERSIZE;
  uint32_t info_header_size = INFOHEADERSIZE;
  uint16_t planes = 1;
  uint16_t color = 24;
  uint32_t compress = 0;
  uint32_t data_size = height * real_width;
  int32_t xppm = 1;
  int32_t yppm = 1;
  
  header_buf[0] = 'B';
  header_buf[1] = 'M';
  memcpy(&header_buf[2], &file_size, sizeof(file_size));
  // 6-9は0
  memcpy(&header_buf[10], &offset_to_data, sizeof(file_size));
  memcpy(&header_buf[14], &info_header_size, sizeof(info_header_size));
  memcpy(&header_buf[18], &width, sizeof(width));
  memcpy(&header_buf[22], &height, sizeof(height));
  memcpy(&header_buf[26], &planes, sizeof(planes));
  memcpy(&header_buf[28], &color, sizeof(color));
  memcpy(&header_buf[30], &compress, sizeof(compress));
  memcpy(&header_buf[34], &data_size, sizeof(data_size));
  memcpy(&header_buf[38], &xppm, sizeof(xppm));
  memcpy(&header_buf[42], &yppm, sizeof(yppm));
  // 46~は0

  //ヘッダの書き込み
  ofs.write(reinterpret_cast<const char*>(header_buf.data()),
            sizeof(uint8_t) * HEADERSIZE);
  
  //RGB情報の書き込み
  for (auto i = (decltype(height))0; i < height; i++){
    for (auto j = (decltype(width))0; j < width; j++){
      auto datum = img[(height - i - 1) * width + j];
      bmp_line[j * 3]     = datum;
      bmp_line[j * 3 + 1] = datum;
      bmp_line[j * 3 + 2] = datum;
    }
    //RGB情報を4バイトの倍数に合わせている
    for (auto j = (decltype(real_width))(width * 3); j < real_width; j++){
      bmp_line[j] = 0;
    }
    ofs.write(reinterpret_cast<const char*>(bmp_line.data()),
              sizeof(uint8_t) * real_width);
  }

  return true;
}

} // namespace bmp
