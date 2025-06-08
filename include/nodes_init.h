#ifndef NODES_INIT_H
#define NODES_INIT_H

#include "type_def.cuh"

void init_fws_nodes(const Mesh &mesh, Vec4 *nodes, const Vec4 &init_P, double gamma);
void init_inf_nodes(const Mesh &mesh, Vec4 *nodes, double u_inf, double v_inf, double beta, double gamma);

#endif // NODES_INIT_H