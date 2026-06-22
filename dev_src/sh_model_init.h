#ifndef SUS2_SH_MODEL_INIT_H
#define SUS2_SH_MODEL_INIT_H

#include <map>
#include <string>
#include <vector>

void WriteSphericalHarmonicModel(const std::string& filename,
                                 const std::map<std::string, std::string>& opts);

double SphericalHarmonicRealCGCoeff(int l1, int rm1,
                                    int l2, int rm2,
                                    int L, int rM);
double SphericalHarmonicRealGauntCoeff(int l1, int rm1,
                                       int l2, int rm2,
                                       int L, int rM);

#endif
