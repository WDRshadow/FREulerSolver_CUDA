#ifndef SHAPE_F_H
#define SHAPE_F_H

#include <vector>
#include <array>

#include "type_def.cuh"

std::array<int, 2> mapping_q9(const int i);
Point gll_2d(int i);

std::array<double, 2> dshape_q4(int i, double xi, double eta);
Point interpolate_q4(const std::array<Point, 4> &pts, double xi, double eta);

#endif // SHAPE_F_H
