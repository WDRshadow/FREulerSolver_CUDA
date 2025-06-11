#include <array>
#include <vector>
#include <cmath>
#include <fstream>

#include "visualize.h"
#include "shape_f.h"
#include "euler_eq.cuh"

void writeVTU(const char *filename,
              const Vec4 *h_nodes,
              const Mesh &mesh,
              double gamma)
{
    std::vector<std::array<double, 2>> points;
    std::vector<double> rho, u, v, p;
    std::vector<std::array<int, 4>> cells;

    int pt_id = 0;
    std::vector global_id(mesh.nx * 3, std::vector(mesh.ny * 3, -1));

    // Nodes data
    for (int j = 0; j < mesh.ny; ++j)
    {
        for (int i = 0; i < mesh.nx; ++i)
        {
            int cellId = j * mesh.nx + i;
            const Vec4 *q9 = h_nodes + cellId * 9;
            for (int idx = 0; idx < 9; ++idx)
            {
                auto [ii, jj] = mapping_q9(idx);
                auto [xi, eta] = gll_2d(idx);

                std::array<Point, 4> pts = {
                    mesh.vertices[mesh.elements[cellId].vertexIds[0]],
                    mesh.vertices[mesh.elements[cellId].vertexIds[1]],
                    mesh.vertices[mesh.elements[cellId].vertexIds[2]],
                    mesh.vertices[mesh.elements[cellId].vertexIds[3]]};
                auto [x, y] = interpolate_q4(pts, xi, eta);

                int global_x = i * 3 + ii;
                int global_y = j * 3 + jj;

                if (global_id[global_x][global_y] == -1)
                {
                    points.push_back({x, y});
                    if (!mesh.elements[cellId].isValid)
                    {
                        rho.push_back(NAN);
                        u.push_back(NAN);
                        v.push_back(NAN);
                        p.push_back(NAN);
                    }
                    else
                    {
                        Vec4 prim = toPrimitive(q9[idx], gamma);
                        rho.push_back(prim[0]);
                        u.push_back(prim[1]);
                        v.push_back(prim[2]);
                        p.push_back(prim[3]);
                    }
                    global_id[global_x][global_y] = pt_id++;
                }
            }
        }
    }

    // Generate Cells
    for (int j = 0; j < mesh.ny * 3 - 1; ++j)
    {
        for (int i = 0; i < mesh.nx * 3 - 1; ++i)
        {
            int p0 = global_id[i][j];
            int p1 = global_id[i + 1][j];
            int p2 = global_id[i + 1][j + 1];
            int p3 = global_id[i][j + 1];
            cells.push_back({p0, p1, p2, p3});
        }
    }

    // Write VTU file
    std::ofstream out(filename);
    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "<UnstructuredGrid>\n";
    out << "<Piece NumberOfPoints=\"" << points.size()
        << "\" NumberOfCells=\"" << cells.size() << "\">\n";

    // Points
    out << "<Points>\n<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (const auto &pt : points)
        out << pt[0] << " " << pt[1] << " 0\n";
    out << "</DataArray>\n</Points>\n";

    // Cells
    out << "<Cells>\n";

    out << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (const auto &c : cells)
        out << c[0] << " " << c[1] << " " << c[2] << " " << c[3] << "\n";
    out << "</DataArray>\n";

    out << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    int offset = 0;
    for (size_t i = 0; i < cells.size(); ++i)
        out << (offset += 4) << "\n";
    out << "</DataArray>\n";

    out << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (size_t i = 0; i < cells.size(); ++i)
        out << "9\n"; // VTK_QUAD = 9
    out << "</DataArray>\n";

    out << "</Cells>\n";

    // Point Data
    out << "<PointData Scalars=\"Density\">\n";

    auto write_scalar = [&](const std::string &name, const std::vector<double> &data)
    {
        out << "<DataArray type=\"Float32\" Name=\"" << name << "\" format=\"ascii\">\n";
        for (double val : data)
            out << val << "\n";
        out << "</DataArray>\n";
    };

    write_scalar("Density", rho);
    write_scalar("VelocityX", u);
    write_scalar("VelocityY", v);
    write_scalar("Pressure", p);

    out << "</PointData>\n";

    out << "</Piece>\n</UnstructuredGrid>\n</VTKFile>\n";

    out.close();
}