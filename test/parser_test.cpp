#include <gtest/gtest.h>

#include "parser.h"

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(PARSER, generate)
{
    ensureOutputFolder("output");
    ensureConfigFile("config.ini");
}

TEST(PARSER, test)
{
    CaseConfig caseCfg;
    MeshConfig meshCfg;
    ICConfig icCfg;
    Vec4 bcCfg;
    SolverConfig solverCfg;
    LimiterConfig limiterCfg;
    VisualizeConfig visCfg;

    parseConfig("config.ini", caseCfg, meshCfg, icCfg, bcCfg, solverCfg, limiterCfg, visCfg);

    std::cout << "Case type: " << caseCfg.type << "\n";
    std::cout << "Mesh nx: " << meshCfg.nx << ", ny: " << meshCfg.ny << "\n";
    std::cout << "IC rho: " << icCfg.fws[0] << ", u: " << icCfg.fws[1] << "\n";
    std::cout << "Solver method: " << solverCfg.method << "\n";
    std::cout << "Limiter enabled: " << limiterCfg.enable << "\n";
}
