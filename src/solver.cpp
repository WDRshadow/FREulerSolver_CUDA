#include "solver.h"
#include "mesh_init.h"
#include "flux_utils.h"
#include "jacobian.cuh"
#include "euler_eq.cuh"

FREulerCache::FREulerCache(const Mesh &mesh)
    : numElements(mesh.numElements), numFaces(mesh.numFaces)
{
    cudaMalloc(&d_nodes, sizeof(Vec4) * numElements * 9);
    cudaMalloc(&d_fluxs, sizeof(Flux) * numElements * 9);
    cudaMalloc(&d_dfluxs, sizeof(Flux) * numElements * 9);
    cudaMalloc(&d_face_fluxs, sizeof(Vec4) * numFaces * 3);
    cudaMalloc(&d_rhs, sizeof(Vec4) * numElements * 9);
}

FREulerCache::FREulerCache(const Mesh &mesh, const Vec4 *h_nodes)
    : numElements(mesh.numElements), numFaces(mesh.numFaces)
{
    cudaMalloc(&d_nodes, sizeof(Vec4) * numElements * 9);
    cudaMalloc(&d_fluxs, sizeof(Flux) * numElements * 9);
    cudaMalloc(&d_dfluxs, sizeof(Flux) * numElements * 9);
    cudaMalloc(&d_face_fluxs, sizeof(Vec4) * numFaces * 3);
    cudaMalloc(&d_rhs, sizeof(Vec4) * numElements * 9);
    cudaMemcpy(d_nodes, h_nodes, sizeof(Vec4) * numElements * 9, cudaMemcpyHostToDevice);
}

FREulerCache::~FREulerCache()
{
    cudaFree(d_nodes);
    cudaFree(d_fluxs);
    cudaFree(d_dfluxs);
    cudaFree(d_face_fluxs);
    cudaFree(d_rhs);
}

FREulerSolver::FREulerSolver(const Mesh &mesh, const Vec4 *h_nodes)
    : nx(mesh.nx), ny(mesh.ny), k1(FREulerCache(mesh, h_nodes)), k2(FREulerCache(mesh)),
      k3(FREulerCache(mesh)), k4(FREulerCache(mesh))
{
    JMatrix2d *h_jacobian_invT = new JMatrix2d[mesh.numElements * 9];
    double *h_jacobian_det = new double[mesh.numElements * 9];
    double *h_jacobian_face = new double[mesh.numFaces * 3];
    calculate_jacobian_cell(mesh, h_jacobian_invT, h_jacobian_det);
    calculate_jacobian_face(mesh, h_jacobian_face);
    cudaMalloc(&d_elements, sizeof(Cell) * mesh.numElements * 9);
    cudaMalloc(&d_faces, sizeof(Face) * mesh.numElements * 9);
    cudaMalloc(&d_jacobian_invT, sizeof(JMatrix2d) * mesh.numElements * 9);
    cudaMalloc(&d_jacobian_det, sizeof(double) * mesh.numElements * 9);
    cudaMalloc(&d_jacobian_face, sizeof(double) * mesh.numFaces * 3);
    cudaMemcpy(d_elements, mesh.elements, sizeof(Cell) * mesh.numElements, cudaMemcpyHostToDevice);
    cudaMemcpy(d_faces, mesh.faces, sizeof(Face) * mesh.numFaces, cudaMemcpyHostToDevice);
    cudaMemcpy(d_jacobian_invT, h_jacobian_invT, sizeof(JMatrix2d) * mesh.numElements * 9, cudaMemcpyHostToDevice);
    cudaMemcpy(d_jacobian_det, h_jacobian_det, sizeof(double) * mesh.numElements * 9, cudaMemcpyHostToDevice);
    cudaMemcpy(d_jacobian_face, h_jacobian_face, sizeof(double) * mesh.numFaces * 3, cudaMemcpyHostToDevice);
    delete[] h_jacobian_invT;
    delete[] h_jacobian_det;
    delete[] h_jacobian_face;
}

FREulerSolver::~FREulerSolver()
{
    cudaFree(d_elements);
    cudaFree(d_faces);
    cudaFree(d_jacobian_invT);
    cudaFree(d_jacobian_det);
    cudaFree(d_jacobian_face);
}

void FREulerSolver::setGamma(const double g)
{
    gamma = g;
}

void FREulerSolver::set_fws_bc(const Vec4 &bc)
{
    bc_P = bc;
}

double FREulerSolver::getCurrentTime() const
{
    return currentTime;
}

void FREulerSolver::getNodes(Vec4 *h_nodes)
{
    cudaMemcpy(h_nodes, k1.d_nodes, sizeof(Vec4) * k1.numElements * 9, cudaMemcpyDeviceToHost);
}

void FREulerSolver::computeRHS(FREulerCache &cache)
{
    Vec4 bc_Q = toConservative(bc_P, gamma);
    calculate_physical_flux(cache.d_nodes, cache.d_fluxs, cache.numElements, gamma);
    calculate_node_div_flux(cache.d_fluxs, cache.d_dfluxs, cache.numElements, gamma);
    calculate_face_num_flux(
        d_faces,
        cache.d_nodes,
        cache.d_fluxs,
        cache.d_face_fluxs,
        bc_Q,
        nx,
        ny,
        gamma);
    calculate_rhs(
        d_elements,
        d_faces,
        cache.d_fluxs,
        cache.d_dfluxs,
        cache.d_face_fluxs,
        d_jacobian_invT,
        d_jacobian_det,
        d_jacobian_face,
        cache.d_rhs,
        cache.numElements,
        gamma);
}

void FREulerSolver::advance(const double dt)
{
    // RK4
    computeRHS(k1);
    calculate_time_forward(k1.d_nodes, k2.d_nodes, k1.d_rhs, dt / 2, k1.numElements);
    computeRHS(k2);
    calculate_time_forward(k1.d_nodes, k3.d_nodes, k2.d_rhs, dt / 2, k1.numElements);
    computeRHS(k3);
    calculate_time_forward(k1.d_nodes, k4.d_nodes, k3.d_rhs, dt, k1.numElements);
    computeRHS(k4);
    calculate_time_forward(k1.d_nodes, k1.d_nodes, k1.d_rhs, dt / 6, k1.numElements);
    calculate_time_forward(k1.d_nodes, k1.d_nodes, k2.d_rhs, dt / 3, k1.numElements);
    calculate_time_forward(k1.d_nodes, k1.d_nodes, k3.d_rhs, dt / 3, k1.numElements);
    calculate_time_forward(k1.d_nodes, k1.d_nodes, k4.d_rhs, dt / 6, k1.numElements);

    // Euler
    // computeRHS(k1);
    // calculate_time_forward(k1.d_nodes, k1.d_nodes, k1.d_rhs, dt, k1.numElements);

    currentTime += dt;
}
