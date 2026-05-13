#ifndef SUS2_SH_MODEL_INIT_H
#define SUS2_SH_MODEL_INIT_H

#include <map>
#include <string>
#include <vector>

void WriteSphericalHarmonicModel(const std::string& filename,
                                 const std::map<std::string, std::string>& opts);

#endif
