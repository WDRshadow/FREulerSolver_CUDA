#ifndef SOLVER_H
#define SOLVER_H

#include "flux_utils.h"

#define EULER 1
#define RK4 4

struct FREulerCache
{
    const int numElements;
    const int numFaces;
    Vec4 *d_nodes = nullptr;
    Flux *d_fluxs = nullptr;
    Flux *d_dfluxs = nullptr;
    Vec4 *d_face_fluxs = nullptr;
    Vec4 *d_rhs = nullptr;

    FREulerCache() = delete;
    FREulerCache(const FREulerCache &) = delete;
    FREulerCache &operator=(const FREulerCache &) = delete;
    FREulerCache(FREulerCache &&) = delete;
    FREulerCache &operator=(FREulerCache &&) = delete;

    explicit FREulerCache(const Mesh &mesh);
    void setNodes(const Vec4 *h_nodes);
    ~FREulerCache();
};

class FREulerSolver
{
public:
    FREulerSolver() = delete;
    FREulerSolver(const FREulerSolver &) = delete;
    FREulerSolver &operator=(const FREulerSolver &) = delete;
    FREulerSolver(FREulerSolver &&) = delete;
    FREulerSolver &operator=(FREulerSolver &&) = delete;

    FREulerSolver(const Mesh &mesh, const Vec4 *h_nodes);
    ~FREulerSolver();
    void setGamma(double g);
    void set_fws_bc(const Vec4 &bc);
    void advance(double dt, int method = RK4);
    double getCurrentTime() const;
    void getNodes(Vec4 *h_nodes) const;

private:
    const int nx, ny;
    double currentTime = 0.0;
    double gamma = GAMMA;
    Vec4 bc_P = {1.4, 3.0, 0.0, 1.0};

    Cell *d_elements = nullptr;
    Face *d_faces = nullptr;
    JMatrix2d *d_jacobian_invT = nullptr;
    double *d_jacobian_det = nullptr;
    double *d_jacobian_face = nullptr;

    FREulerCache k1;
    FREulerCache k2;
    FREulerCache k3;
    FREulerCache k4;

    void computeRHS(const FREulerCache &cache) const;
};

#endif // SOLVER_H