#ifndef TYPE_DEF_CUH
#define TYPE_DEF_CUH

#include <cuda_runtime.h>

#define BLOCK_SIZE 256

#define GAMMA 1.4

#define X_WALL (-1)
#define Y_WALL (-2)
#define INLET (-3)
#define OUTLET (-4)

#define BOTTOM 0
#define RIGHT 1
#define TOP 2
#define LEFT 3

struct Vec4
{
    double x, y, z, w;

    __host__ __device__ Vec4() : x(0), y(0), z(0), w(0) {}
    __host__ __device__ Vec4(const double x, const double y, const double z, const double w)
        : x(x), y(y), z(z), w(w) {}

    __host__ __device__ double &operator[](const int i)
    {
        return *(&x + i);
    }

    __host__ __device__ const double &operator[](const int i) const
    {
        return *(&x + i);
    }

    __host__ __device__ Vec4 operator+(const Vec4 &rhs) const
    {
        return {x + rhs.x, y + rhs.y, z + rhs.z, w + rhs.w};
    }

    __host__ __device__ Vec4 operator-(const Vec4 &rhs) const
    {
        return {x - rhs.x, y - rhs.y, z - rhs.z, w - rhs.w};
    }

    __host__ __device__ Vec4 operator-() const
    {
        return {-x, -y, -z, -w};
    }

    __host__ __device__ Vec4 operator*(const double s) const
    {
        return {x * s, y * s, z * s, w * s};
    }

    __host__ __device__ Vec4 operator/(const double s) const
    {
        return {x / s, y / s, z / s, w / s};
    }

    __host__ __device__ Vec4 &operator+=(const Vec4 &rhs)
    {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        w += rhs.w;
        return *this;
    }

    __host__ __device__ Vec4 &operator*=(const double s)
    {
        x *= s;
        y *= s;
        z *= s;
        w *= s;
        return *this;
    }
};

struct Point
{
    double x, y;

    __host__ __device__ Point operator-() const
    {
        return {-x, -y};
    }
};

struct Flux
{
    Vec4 f, g;
};

struct Face
{
    int leftCell;
    int rightCell;
    Point normal;
};

struct Cell
{
    int vertexIds[4];
    int faceIds[4];
    bool isValid;
};

struct Mesh
{
    int nx, ny;
    double width, height;
    int numVertices, numFaces, numElements;
    Point *vertices = nullptr;
    Face *faces = nullptr;
    Cell *elements = nullptr;

    Mesh() = delete;
    Mesh(const Mesh &) = delete;
    Mesh &operator=(const Mesh &) = delete;
    Mesh(Mesh &&) = delete;
    Mesh &operator=(Mesh &&) = delete;

    Mesh(const int nx, const int ny, const double width, const double height)
        : nx(nx), ny(ny), width(width), height(height),
          numVertices((nx + 1) * (ny + 1)), numFaces(nx * (ny + 1) + ny * (nx + 1)),
          numElements(nx * ny)
    {
        vertices = new Point[numVertices];
        faces = new Face[numFaces];
        elements = new Cell[numElements];
    }

    ~Mesh()
    {
        if (vertices)
            delete[] vertices;
        if (faces)
            delete[] faces;
        if (elements)
            delete[] elements;
    }
};

#endif // TYPE_DEF_CUH
