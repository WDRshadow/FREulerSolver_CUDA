#ifndef SHAPE_F_H
#define SHAPE_F_H

#include <vector>
#include <array>

#include "type_def.cuh"

using Q3 = std::array<Vec4, 3>;
using Q9 = std::array<Vec4, 9>;

std::vector<Q9> operator+(const std::vector<Q9> &a, const std::vector<Q9> &b);
std::vector<Q9> operator*(const std::vector<Q9> &a, double b);
void operator+=(std::vector<Q9> &a, const std::vector<Q9> &b);

std::array<int, 2> mapping_q9(int i);
std::array<int, 3> face_mapping(int faceType);
Q3 face_mapping(const Q9 &q9, int faceType);

double gll_1d(int i);
double gll_weight_1d(int i);
Vec4 gll_integrate_1d(const Q3 &q3);
Point gll_2d(int i);
double gll_weight_2d(int i);
Vec4 gll_integrate_2d(const Q9 &q9);

double lagrange(int i, double s);
double dlagrange(int i, double s);
Vec4 interpolate(const Q3 &Qs, double s);
double lagrange(int i, double xi, double eta);
std::array<double, 2> dlagrange(int i, double xi, double eta);
Vec4 interpolate(const Q9 &Qs, double xi, double eta);

double shape(int i, double xi, double eta);
std::array<double, 2> dshape(int i, double xi, double eta);
Point interpolate(const std::array<Point, 4> &pts, double xi, double eta);

#endif // SHAPE_F_H
