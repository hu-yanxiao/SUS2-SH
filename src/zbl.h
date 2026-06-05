/*   ZBL short-range repulsion support for SUS2 potentials. */

#ifndef MLIP_ZBL_H
#define MLIP_ZBL_H

#include <string>
#include <vector>

#include "configuration.h"
#include "neighborhoods.h"

struct ZBLPairValue {
	double energy;
	double dEdr;
};

struct ZBLPairConstants {
	double screening_inv;
	double prefactor;
};

struct ZBLEFS {
	double energy;
	std::vector<Vector3> forces;
	Matrix3 stresses;

	ZBLEFS();
	explicit ZBLEFS(int atom_count);
	void Resize(int atom_count);
};

double DefaultZBLInnerCutoff();
double DefaultZBLOuterCutoff();
double DefaultZBLTypewiseCutoffFactor();
int ZBLAtomicNumberFromSymbol(const std::string& symbol);
std::vector<int> ParseZBLAtomicNumbers(const std::string& value);
double ZBLCovalentRadius(int atomic_number);
double ZBLTypewiseOuterCutoff(int atomic_number_i,
                              int atomic_number_j,
                              double global_outer_cutoff,
                              double typewise_cutoff_factor);
void FillZBLPairCutoffTables(const std::vector<int>& atomic_numbers,
                             double global_inner_cutoff,
                             double global_outer_cutoff,
                             bool typewise_cutoff_enabled,
                             double typewise_cutoff_factor,
                             std::vector<double>& pair_inner_cutoffs,
                             std::vector<double>& pair_outer_cutoffs,
                             std::vector<double>& pair_outer_sq);
ZBLPairValue ComputeZBLPair(int atomic_number_i,
                            int atomic_number_j,
                            double distance,
                            double inner_cutoff,
                            double outer_cutoff);
ZBLPairConstants MakeZBLPairConstants(int atomic_number_i, int atomic_number_j);
ZBLPairValue ComputeZBLPairCached(const ZBLPairConstants& constants,
                                  double distance,
                                  double inner_cutoff,
                                  double outer_cutoff);
ZBLPairValue ComputeZBLPair(int atomic_number_i,
                            int atomic_number_j,
                            double distance,
                            double inner_cutoff,
                            double outer_cutoff,
                            double typewise_cutoff_factor);

class ZBLPotential {
private:
	bool enabled_;
	double inner_cutoff_;
	double outer_cutoff_;
	bool typewise_cutoff_enabled_;
	double typewise_cutoff_factor_;
	std::vector<int> atomic_numbers_;
	std::vector<double> pair_inner_cutoffs_;
	std::vector<double> pair_outer_cutoffs_;
	std::vector<double> pair_outer_sq_;
	std::vector<ZBLPairConstants> pair_constants_;

public:
	ZBLPotential();

	bool Enabled() const;
	double InnerCutoff() const;
	double OuterCutoff() const;
	bool TypewiseCutoffEnabled() const;
	double TypewiseCutoffFactor() const;
	const std::vector<int>& AtomicNumbers() const;

	void Clear();
	void Configure(const std::vector<int>& atomic_numbers,
	               double inner_cutoff = 0.7,
	               double outer_cutoff = 1.4,
	               bool typewise_cutoff_enabled = false,
	               double typewise_cutoff_factor = 0.7);

	ZBLEFS Compute(const Configuration& cfg) const;
	ZBLEFS Compute(const Configuration& cfg, const Neighborhoods& neighborhoods) const;
	void AddTo(Configuration& cfg) const;
	void AddTo(Configuration& cfg, const Neighborhoods& neighborhoods) const;
};

#endif // MLIP_ZBL_H
