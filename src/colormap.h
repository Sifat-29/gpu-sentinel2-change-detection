// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// Colour rule of the change map, shared by the CPU and GPU colorize paths so
// that both produce byte-identical images.
#ifndef SRC_COLORMAP_H_
#define SRC_COLORMAP_H_

#include <cstdint>

#include "host_device.h"
#include "types.h"

namespace cd {

// Writes the RGBA colour of one pixel into rgba[0..3].
//  - Break pixels get one of 8 ramp stops chosen by the break date inside the
//    monitoring period: with span = max(1, last_day - first_monitor_day) and
//    offset = clamp(break_day - first_monitor_day, 0, span),
//    stop = min(7, 8 * offset / span). Only integer arithmetic is used, so
//    host and device agree exactly. Greenness losses (magnitude < 0) are
//    opaque (alpha 255), gains are translucent (alpha 160).
//  - Stable pixels are dark grey (40, 40, 40, 255).
//  - Pixels with insufficient data are fully transparent (0, 0, 0, 0).
// Days are days since 1970-01-01; first_monitor_day is the day of the first
// monitoring date and last_day the day of the last date.
//
// The ramp is viridis sampled at 0, 1/7, ..., 1 (early break = dark purple,
// late break = yellow):
//   0 (68, 1, 84)    1 (70, 50, 126)   2 (54, 92, 141)  3 (39, 127, 142)
//   4 (31, 161, 135) 5 (74, 193, 109)  6 (159, 218, 58) 7 (253, 231, 37)
CD_HOST_DEVICE inline void ChangeColor(int16_t break_index, float magnitude,
                                       uint8_t status, int first_monitor_day,
                                       int last_day, int break_day,
                                       uint8_t rgba[4]) {
  if (status == static_cast<uint8_t>(PixelStatus::kBreak) && break_index >= 0) {
    int span = last_day - first_monitor_day;
    if (span < 1) {
      span = 1;
    }
    int offset = break_day - first_monitor_day;
    if (offset < 0) {
      offset = 0;
    }
    if (offset > span) {
      offset = span;
    }
    int stop = (8 * offset) / span;
    if (stop > 7) {
      stop = 7;
    }
    // Packed 0xRRGGBB value of the selected ramp stop.
    uint32_t rgb = 0;
    switch (stop) {
      case 0:
        rgb = 0x440154;
        break;
      case 1:
        rgb = 0x46327E;
        break;
      case 2:
        rgb = 0x365C8D;
        break;
      case 3:
        rgb = 0x277F8E;
        break;
      case 4:
        rgb = 0x1FA187;
        break;
      case 5:
        rgb = 0x4AC16D;
        break;
      case 6:
        rgb = 0x9FDA3A;
        break;
      default:
        rgb = 0xFDE725;
        break;
    }
    rgba[0] = static_cast<uint8_t>(rgb >> 16);
    rgba[1] = static_cast<uint8_t>((rgb >> 8) & 0xFF);
    rgba[2] = static_cast<uint8_t>(rgb & 0xFF);
    rgba[3] = magnitude < 0.0f ? 255 : 160;
  } else if (status == static_cast<uint8_t>(PixelStatus::kStable)) {
    rgba[0] = 40;
    rgba[1] = 40;
    rgba[2] = 40;
    rgba[3] = 255;
  } else {
    rgba[0] = 0;
    rgba[1] = 0;
    rgba[2] = 0;
    rgba[3] = 0;
  }
}

}  // namespace cd

#endif  // SRC_COLORMAP_H_
