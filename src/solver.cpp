#include <stdexcept>

#include "solver.h"
#include "mesh_init.h"
#include "flux_utils.h"
#include "jacobian.cuh"
#include "euler_eq.cuh"
#include "limiter.h"

FREulerCache::FREulerCache(const Mesh &mesh)
    : numElements(mesh.numElements), numFaces(mesh.numFaces)
{
    cudaMalloc(&d_nodes, sizeof(Vec4) * numElements * 9);
    cudaMalloc(&d_fluxs, sizeof(Flux) * numElements * 9);
    cudaMalloc(&d_dfluxs, sizeof(Flux) * numElements * 9);
    cudaMalloc(&d_face_fluxs, sizeof(Vec4) * numFaces * 3);
    cudaMalloc(&d_rhs, sizeof(Vec4) * numElements * 9);
}

void FREulerCache::setNodes(const Vec4 *h_nodes)
{
    cudaMemcpy(d_nodes, h_nodes, sizeof(Vec4) * numElements * 9, cudaMemcpyHostToDevice);
}

FREulerCache::~FREulerCache()
{
    if (d_nodes)
        cudaFree(d_nodes);
    if (d_fluxs)
        cudaFree(d_fluxs);
    if (d_dfluxs)
        cudaFree(d_dfluxs);
    if (d_face_fluxs)
        cudaFree(d_face_fluxs);
    if (d_rhs)
        cudaFree(d_rhs);
}

FREulerSolver::FREulerSolver(const Mesh &mesh, const Vec4 *h_nodes)
    : nx(mesh.nx), ny(mesh.ny), k1(FREulerCache(mesh)), k2(FREulerCache(mesh)),
      k3(FREulerCache(mesh)), k4(FREulerCache(mesh))
{
    k1.setNodes(h_nodes);
    auto *h_jacobian_invT = new Matrix2d[mesh.numElements * 9];
    auto *h_jacobian_det = new double[mesh.numElements * 9];
    auto *h_jacobian_face = new double[mesh.numFaces * 3];
    calculate_jacobian_cell(mesh, h_jacobian_invT, h_jacobian_det);
    calculate_jacobian_face(mesh, h_jacobian_face);
    cudaMalloc(&d_elements, sizeof(Cell) * mesh.numElements * 9);
    cudaMalloc(&d_faces, sizeof(Face) * mesh.numElements * 9);
    cudaMalloc(&d_jacobian_invT, sizeof(Matrix2d) * mesh.numElements * 9);
    cudaMalloc(&d_jacobian_det, sizeof(double) * mesh.numElements * 9);
    cudaMalloc(&d_jacobian_face, sizeof(double) * mesh.numFaces * 3);
    cudaMemcpy(d_elements, mesh.elements, sizeof(Cell) * mesh.numElements, cudaMemcpyHostToDevice);
    cudaMemcpy(d_faces, mesh.faces, sizeof(Face) * mesh.numFaces, cudaMemcpyHostToDevice);
    cudaMemcpy(d_jacobian_invT, h_jacobian_invT, sizeof(Matrix2d) * mesh.numElements * 9, cudaMemcpyHostToDevice);
    cudaMemcpy(d_jacobian_det, h_jacobian_det, sizeof(double) * mesh.numElements * 9, cudaMemcpyHostToDevice);
    cudaMemcpy(d_jacobian_face, h_jacobian_face, sizeof(double) * mesh.numFaces * 3, cudaMemcpyHostToDevice);
    delete[] h_jacobian_invT;
    delete[] h_jacobian_det;
    delete[] h_jacobian_face;
}

FREulerSolver::~FREulerSolver()
{
    if (d_elements)
        cudaFree(d_elements);
    if (d_faces)
        cudaFree(d_faces);
    if (d_jacobian_invT)
        cudaFree(d_jacobian_invT);
    if (d_jacobian_det)
        cudaFree(d_jacobian_det);
    if (d_jacobian_face)
        cudaFree(d_jacobian_face);
    if (d_tmp_nodes)
        cudaFree(d_tmp_nodes);
}

void FREulerSolver::setGamma(const double g)
{
    gamma = g;
}

void FREulerSolver::set_fws_bc(const Vec4 &bc)
{
    bc_P = bc;
}

void FREulerSolver::set_tvb_limiter(double M)
{
    isLimiter = true;
    this->M = M;
    cudaMalloc(&d_tmp_nodes, sizeof(Vec4) * k1.numElements * 9);
}

double FREulerSolver::getCurrentTime() const
{
    return currentTime;
}

void FREulerSolver::getNodes(Vec4 *h_nodes) const
{
    cudaMemcpy(h_nodes, k1.d_nodes, sizeof(Vec4) * k1.numElements * 9, cudaMemcpyDeviceToHost);
}

void FREulerSolver::computeRHS(const FREulerCache &cache) const
{
    Vec4 bc_Q = toConservative(bc_P, gamma);
    calculate_physical_flux(cache.d_nodes, cache.d_fluxs, cache.numElements, gamma);
    calculate_physical_div_flux(cache.d_fluxs, cache.d_dfluxs, cache.numElements);
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
        cache.numElements);
}

void FREulerSolver::limit(const FREulerCache &cache)
{
    if (isLimiter)
    {
        tvd_limiter(d_elements, d_faces, cache.d_nodes, d_tmp_nodes, cache.numElements, bc_P, gamma, M);
        cudaMemcpy(cache.d_nodes, d_tmp_nodes, sizeof(Vec4) * cache.numElements * 9, cudaMemcpyDeviceToDevice);
    }
}

void FREulerSolver::advance(const double dt, const int method)
{
    if (currentTime < 1e-10)
    {
        limit(k1);
    }
    switch (method)
    {
    case EULER:
    {
        computeRHS(k1);
        calculate_time_forward(k1.d_nodes, k1.d_nodes, k1.d_rhs, dt, k1.numElements);
        break;
    }
    case RK4:
    {
        computeRHS(k1);
        calculate_time_forward(k1.d_nodes, k2.d_nodes, k1.d_rhs, dt / 2, k1.numElements);
        limit(k2);
        computeRHS(k2);
        calculate_time_forward(k1.d_nodes, k3.d_nodes, k2.d_rhs, dt / 2, k1.numElements);
        limit(k3);
        computeRHS(k3);
        calculate_time_forward(k1.d_nodes, k4.d_nodes, k3.d_rhs, dt, k1.numElements);
        limit(k4);
        computeRHS(k4);
        calculate_time_forward(k1.d_nodes, k1.d_nodes, k1.d_rhs, dt / 6, k1.numElements);
        calculate_time_forward(k1.d_nodes, k1.d_nodes, k2.d_rhs, dt / 3, k1.numElements);
        calculate_time_forward(k1.d_nodes, k1.d_nodes, k3.d_rhs, dt / 3, k1.numElements);
        calculate_time_forward(k1.d_nodes, k1.d_nodes, k4.d_rhs, dt / 6, k1.numElements);
        break;
    }
    default:
        throw std::invalid_argument("Unsupported method for time advancement.");
    }
    limit(k1);
    currentTime += dt;
}
