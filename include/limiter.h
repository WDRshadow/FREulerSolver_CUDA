#ifndef LIMITER_H
#define LIMITER_H

#include "type_def.cuh"

void tvd_limiter(
    const Cell *d_cells,
    const Face *d_faces,
    const Vec4 *d_nodes,
    Vec4 *d_new_nodes,
    int num_elements,
    const Vec4 &bc_Q,
    double gamma,
    const double hx,
    const double hy,
    double Mh2);

#endif // LIMITER_H