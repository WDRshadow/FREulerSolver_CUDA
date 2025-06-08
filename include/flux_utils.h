#ifndef FLUX_UTILS_H
#define FLUX_UTILS_H

#include "type_def.cuh"

class NumFluxCalculator
{
    const int nx, ny;
    const double width, height;
    const int numVertices, numFaces, numElements;
    const Vec4 bc;
    const double gamma;
    Point *d_vertices = nullptr;
    Face *d_faces = nullptr;
    Cell *d_elements = nullptr;

public:
    NumFluxCalculator(const Mesh &mesh, const Vec4 &bc_P, double gamma);
    void calculate_face_num_flux(const CellNode *cell_nodes, FaceNumFlux *face_flux);
};

#endif // FLUX_UTILS_H
