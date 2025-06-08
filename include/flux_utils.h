#ifndef FLUX_UTILS_H
#define FLUX_UTILS_H

#include "type_def.cuh"
#include "jacobian.cuh"

void calculate_physical_flux(
    const Vec4 *d_nodes,
    Flux *d_fluxs,
    int num_elements,
    double gamma);

void calculate_face_num_flux(
    const Face *d_faces,
    const Vec4 *d_nodes,
    const Flux *d_fluxs,
    Vec4 *d_face_fluxs,
    const Vec4 &bc_Q,
    const int nx,
    const int ny,
    const double gamma);

void calculate_node_div_flux(
    const Flux *d_flux,
    Flux *d_dflux,
    const int num_elements,
    const double gamma);

void calculate_rhs(const Cell *d_elements,
                   const Face *d_faces,
                   const Flux *d_fluxs,
                   const Flux *d_dfluxs,
                   const Vec4 *d_face_fluxs,
                   const JMatrix2d *d_jacobian_invT,
                   const double *d_jacobian_det,
                   const double *d_jacobian_face,
                   Vec4 *d_rhs,
                   const int num_elements,
                   const double gamma);

void calculate_time_forward(const Vec4 *d_src_nodes, Vec4 *d_dst_nodes, const Vec4 *d_rhs, double dt, int numElements);

#endif // FLUX_UTILS_H
