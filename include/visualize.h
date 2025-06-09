#ifndef VISUALIZE_H
#define VISUALIZE_H

#include "type_def.cuh"

void writeVTU(const char *filename,
              const Vec4 *h_nodes,
              const Mesh &mesh,
              double gamma = GAMMA);

#endif // VISUALIZE_H