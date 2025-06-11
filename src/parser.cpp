#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <filesystem>

namespace fs = std::filesystem;

#include "parser.h"

bool stringToBool(const std::string &str)
{
    return str == "true" || str == "1";
}

void parseConfig(const std::string &filename,
                 CaseConfig &caseCfg,
                 MeshConfig &meshCfg,
                 ICConfig &icCfg,
                 Vec4 &bcCfg,
                 SolverConfig &solverCfg,
                 LimiterConfig &limiterCfg,
                 VisualizeConfig &visCfg)
{
    std::ifstream file(filename);
    std::string line, section;

    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == '#')
            continue;

        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);

        if (line.front() == '[' && line.back() == ']')
        {
            section = line.substr(1, line.size() - 2);
        }
        else
        {
            auto pos = line.find('=');
            if (pos == std::string::npos)
                continue;

            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);

            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);

            std::istringstream valStream(value);

            if (section == "Case")
            {
                if (key == "type")
                    caseCfg.type = value;
            }
            else if (section == "Mesh")
            {
                if (key == "width")
                    valStream >> meshCfg.width;
                else if (key == "height")
                    valStream >> meshCfg.height;
                else if (key == "nx")
                    valStream >> meshCfg.nx;
                else if (key == "ny")
                    valStream >> meshCfg.ny;
                else if (key == "fws_stepNX")
                    valStream >> meshCfg.stepNX;
                else if (key == "fws_stepNY")
                    valStream >> meshCfg.stepNY;
            }
            else if (section == "IC")
            {
                if (key == "fws_rho")
                    valStream >> icCfg.fws[0];
                else if (key == "fws_u")
                    valStream >> icCfg.fws[1];
                else if (key == "fws_v")
                    valStream >> icCfg.fws[2];
                else if (key == "fws_p")
                    valStream >> icCfg.fws[3];
                else if (key == "inf_u")
                    valStream >> icCfg.inf_u;
                else if (key == "inf_v")
                    valStream >> icCfg.inf_v;
                else if (key == "inf_beta")
                    valStream >> icCfg.inf_beta;
            }
            else if (section == "BC")
            {
                if (key == "fws_rho")
                    valStream >> bcCfg[0];
                else if (key == "fws_u")
                    valStream >> bcCfg[1];
                else if (key == "fws_v")
                    valStream >> bcCfg[2];
                else if (key == "fws_p")
                    valStream >> bcCfg[3];
            }
            else if (section == "Solver")
            {
                if (key == "step_size")
                    valStream >> solverCfg.step_size;
                else if (key == "steps")
                    valStream >> solverCfg.steps;
                else if (key == "method")
                    solverCfg.method = value;
            }
            else if (section == "Limiter")
            {
                if (key == "enable")
                    limiterCfg.enable = stringToBool(value);
                else if (key == "M")
                    valStream >> limiterCfg.M;
            }
            else if (section == "Visiualize")
            {
                if (key == "enable")
                    visCfg.enable = stringToBool(value);
                else if (key == "gap")
                    valStream >> visCfg.gap;
            }
        }
    }
}

void ensureOutputFolder(const std::string &dir)
{
    fs::path configDir(dir);
    if (!fs::exists(configDir))
    {
        fs::create_directories(configDir);
    }
}

void ensureConfigFile(const std::string &filename)
{
    fs::path configFile = filename;

    if (!fs::exists(configFile))
    {
        std::ofstream out(configFile);
        if (out)
        {
            out << "[Case]\n";
            out << "type = fws\n\n";

            out << "[Mesh]\n";
            out << "width = 3.0\n";
            out << "height = 1.0\n";
            out << "nx = 30\n";
            out << "ny = 10\n";
            out << "fws_stepNX = 24\n";
            out << "fws_stepNY = 2\n\n";

            out << "[IC]\n";
            out << "fws_rho = 1.4\n";
            out << "fws_u = 3.0\n";
            out << "fws_v = 0.0\n";
            out << "fws_p = 1.0\n";
            out << "inf_u = 1.0\n";
            out << "inf_v = 1.0\n";
            out << "inf_beta = 5.0\n\n";

            out << "[BC]\n";
            out << "fws_rho = 1.4\n";
            out << "fws_u = 3.0\n";
            out << "fws_v = 0.0\n";
            out << "fws_p = 1.0\n\n";

            out << "[Solver]\n";
            out << "step_size = 0.00001\n";
            out << "steps = 50000\n";
            out << "method = RK4\n\n";

            out << "[Limiter]\n";
            out << "enable = true\n";
            out << "M = 0.0\n\n";

            out << "[Visiualize]\n";
            out << "enable = true\n";
            out << "gap = 1000\n";

            out.close();
        }
    }
}
