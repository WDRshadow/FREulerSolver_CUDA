#ifndef PARSER_H
#define PARSER_H

#include "type_def.cuh"

struct CaseConfig
{
    std::string type;
};

struct MeshConfig
{
    double width = 0, height = 0;
    int nx = 0, ny = 0, stepNX = 0, stepNY = 0;
};

struct ICConfig
{
    Vec4 fws{};
    double inf_u = 0.0;
    double inf_v = 0.0;
    double inf_beta = 0.0;
};

struct SolverConfig
{
    double step_size = 0;
    int steps = 0;
    std::string method;
};

struct LimiterConfig
{
    bool enable = false;
    double M = 0;
};

struct VisualizeConfig
{
    bool enable = false;
    int gap = 0;
};

void parseConfig(const std::string &filename,
                 CaseConfig &caseCfg,
                 MeshConfig &meshCfg,
                 ICConfig &icCfg,
                 Vec4 &bcCfg,
                 SolverConfig &solverCfg,
                 LimiterConfig &limiterCfg,
                 VisualizeConfig &visCfg);

void ensureOutputFolder(const std::string &dir);
void ensureConfigFile(const std::string &filename);


#endif // PARSER_H