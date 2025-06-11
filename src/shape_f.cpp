#include <stdexcept>

#include "shape_f.h"

std::array<int, 2> mapping_q9(const int i)
{
    switch (i)
    {
    case 0:
        return {0, 0};
    case 1:
        return {1, 0};
    case 2:
        return {2, 0};
    case 3:
        return {2, 1};
    case 4:
        return {2, 2};
    case 5:
        return {1, 2};
    case 6:
        return {0, 2};
    case 7:
        return {0, 1};
    case 8:
        return {1, 1};
    default:
        throw std::invalid_argument("Invalid index for 9 node quadrilateral");
    }
}

double gll_1d(const int i)
{
    switch (i)
    {
    case 0:
        return -1.0;
    case 1:
        return 0.0;
    case 2:
        return 1.0;
    default:
        throw std::invalid_argument("Invalid index for GLL node");
    }
}

Point gll_2d(const int i)
{
    auto [x, y] = mapping_q9(i);
    return {gll_1d(x), gll_1d(y)};
}

double shape(const int i, const double xi, const double eta)
{
    switch (i)
    {
    case 0:
        return 1.0 / 4.0 * (1 - xi) * (1 - eta);
    case 1:
        return 1.0 / 4.0 * (1 + xi) * (1 - eta);
    case 2:
        return 1.0 / 4.0 * (1 + xi) * (1 + eta);
    case 3:
        return 1.0 / 4.0 * (1 - xi) * (1 + eta);
    default:
        throw std::invalid_argument("Invalid index for shape function");
    }
}

std::array<double, 2> dshape_q4(const int i, const double xi, const double eta)
{
    switch (i)
    {
    case 0:
        return {-1.0 / 4.0 * (1 - eta), -1.0 / 4.0 * (1 - xi)};
    case 1:
        return {1.0 / 4.0 * (1 - eta), -1.0 / 4.0 * (1 + xi)};
    case 2:
        return {1.0 / 4.0 * (1 + eta), 1.0 / 4.0 * (1 + xi)};
    case 3:
        return {-1.0 / 4.0 * (1 + eta), 1.0 / 4.0 * (1 - xi)};
    default:
        throw std::invalid_argument("Invalid index for shape function");
    }
}

Point interpolate_q4(const std::array<Point, 4> &pts, double xi, double eta)
{
    Point result{};
    for (int i = 0; i < 4; ++i)
    {
        const auto shape_i = shape(i, xi, eta);
        result.x += pts[i].x * shape_i;
        result.y += pts[i].y * shape_i;
    }
    return result;
}
