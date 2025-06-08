#ifndef MESH_INIT_H
#define MESH_INIT_H

#include "type_def.cuh"

void init_fws_mesh(Mesh &mesh, int stepNX, int stepNY);
void init_inf_mesh(Mesh &mesh);

#endif // MESH_INIT_H