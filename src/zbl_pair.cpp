#include "zbl.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <string>

#include "common/utils.h"

namespace {

const double kZBLCovalentRadius[94] = {
	0.426667, 0.613333, 1.6,     1.25333, 1.02667, 1.0,     0.946667, 0.84,    0.853333,
	0.893333, 1.86667,  1.66667, 1.50667, 1.38667, 1.46667, 1.36,     1.32,    1.28,
	2.34667,  2.05333,  1.77333, 1.62667, 1.61333, 1.46667, 1.42667,  1.38667, 1.33333,
	1.32,     1.34667,  1.45333, 1.49333, 1.45333, 1.53333, 1.46667,  1.52,    1.56,
	2.52,     2.22667,  1.96,    1.85333, 1.76,    1.65333, 1.53333,  1.50667, 1.50667,
	1.44,     1.53333,  1.64,    1.70667, 1.68,    1.68,    1.64,     1.76,    1.74667,
	2.78667,  2.34667,  2.16,    1.96,    2.10667, 2.09333, 2.08,     2.06667, 2.01333,
	2.02667,  2.01333,  2.0,     1.98667, 1.98667, 1.97333, 2.04,     1.94667, 1.82667,
	1.74667,  1.64,     1.57333, 1.54667, 1.48,    1.49333, 1.50667,  1.76,    1.73333,
	1.73333,  1.81333,  1.74667, 1.84,    1.89333, 2.68,    2.41333,  2.22667, 2.10667,
		2.02667,  2.04,     2.05333, 2.06667};

const char* const kZBLElementSymbols[94] = {
	"H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
	"Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca",
	"Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
	"Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr",
	"Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
	"Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
	"Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
	"Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
	"Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
	"Pa", "U",  "Np", "Pu"};

bool UseTypewiseZBLCutoff(double typewise_cutoff_factor)
{
	return typewise_cutoff_factor > 0.0;
}

std::string Trim(const std::string& value)
{
	std::size_t begin = 0;
	while (begin < value.size() &&
	       std::isspace(static_cast<unsigned char>(value[begin])))
		++begin;
	std::size_t end = value.size();
	while (end > begin &&
	       std::isspace(static_cast<unsigned char>(value[end - 1])))
		--end;
	return value.substr(begin, end - begin);
}

} // namespace

double DefaultZBLInnerCutoff()
{
	return 0.7;
}

double DefaultZBLOuterCutoff()
{
	return 1.4;
}

double DefaultZBLTypewiseCutoffFactor()
{
	return 0.7;
}

int ZBLAtomicNumberFromSymbol(const std::string& symbol)
{
	const std::string token = Trim(symbol);
	if (token.empty())
		ERROR("ZBL element list contains an empty token.");
	bool digits = true;
	for (char ch : token)
		digits = digits && std::isdigit(static_cast<unsigned char>(ch));
	if (digits) {
		const int atomic_number = std::stoi(token);
		if (atomic_number < 1 || atomic_number > 94)
			ERROR("ZBL atomic numbers should be in [1, 94].");
		return atomic_number;
	}
	for (int i = 0; i < 94; ++i)
		if (token == kZBLElementSymbols[i])
			return i + 1;
	ERROR("Unknown ZBL element symbol: " + token);
	return 0;
}

std::vector<int> ParseZBLAtomicNumbers(const std::string& value)
{
	std::vector<int> atomic_numbers;
	std::stringstream ss(value);
	std::string token;
	while (std::getline(ss, token, ','))
		atomic_numbers.push_back(ZBLAtomicNumberFromSymbol(token));
	if (atomic_numbers.empty())
		ERROR("ZBL requires at least one element.");
	return atomic_numbers;
}

double ZBLCovalentRadius(int atomic_number)
{
	if (atomic_number <= 0 || atomic_number > 94)
		ERROR("ZBL typewise cutoff requires atomic numbers in [1, 94].");
	return kZBLCovalentRadius[atomic_number - 1];
}

double ZBLTypewiseOuterCutoff(int atomic_number_i,
                              int atomic_number_j,
                              double global_outer_cutoff,
                              double typewise_cutoff_factor)
{
	if (global_outer_cutoff <= 0.0)
		ERROR("ZBL outer cutoff should be positive.");
	if (typewise_cutoff_factor < 0.5)
		ERROR("ZBL typewise cutoff factor should be at least 0.5.");
	const double pair_outer = typewise_cutoff_factor *
		(ZBLCovalentRadius(atomic_number_i) + ZBLCovalentRadius(atomic_number_j));
	return std::min(global_outer_cutoff, pair_outer);
}

void FillZBLPairCutoffTables(const std::vector<int>& atomic_numbers,
                             double global_inner_cutoff,
                             double global_outer_cutoff,
                             bool typewise_cutoff_enabled,
                             double typewise_cutoff_factor,
                             std::vector<double>& pair_inner_cutoffs,
                             std::vector<double>& pair_outer_cutoffs,
                             std::vector<double>& pair_outer_sq)
{
	const int species_count = static_cast<int>(atomic_numbers.size());
	if (species_count <= 0)
		ERROR("ZBL requires one atomic number per model species.");
	if (global_inner_cutoff < 0.0)
		ERROR("ZBL inner cutoff should be non-negative.");
	if (global_outer_cutoff <= 0.0)
		ERROR("ZBL outer cutoff should be positive.");
	if (typewise_cutoff_enabled) {
		if (typewise_cutoff_factor < 0.5)
			ERROR("ZBL typewise cutoff factor should be at least 0.5.");
	} else if (global_outer_cutoff <= global_inner_cutoff) {
		ERROR("ZBL cutoffs should satisfy 0 <= inner < outer.");
	}

	pair_inner_cutoffs.resize(static_cast<size_t>(species_count) * species_count);
	pair_outer_cutoffs.resize(static_cast<size_t>(species_count) * species_count);
	pair_outer_sq.resize(static_cast<size_t>(species_count) * species_count);
	for (int i = 0; i < species_count; ++i) {
		const int Zi = atomic_numbers[i];
		if (Zi <= 0)
			ERROR("ZBL atomic numbers should be positive.");
		if (typewise_cutoff_enabled)
			ZBLCovalentRadius(Zi);
		for (int j = 0; j < species_count; ++j) {
			const int Zj = atomic_numbers[j];
			if (Zj <= 0)
				ERROR("ZBL atomic numbers should be positive.");
			const double pair_inner =
				typewise_cutoff_enabled ? 0.0 : global_inner_cutoff;
			const double pair_outer = typewise_cutoff_enabled ?
				ZBLTypewiseOuterCutoff(Zi, Zj, global_outer_cutoff,
				                       typewise_cutoff_factor) :
				global_outer_cutoff;
			const size_t pair_index = static_cast<size_t>(i) * species_count + j;
			pair_inner_cutoffs[pair_index] = pair_inner;
			pair_outer_cutoffs[pair_index] = pair_outer;
			pair_outer_sq[pair_index] = pair_outer * pair_outer;
		}
	}
}

ZBLPairValue ComputeZBLPair(int atomic_number_i,
                            int atomic_number_j,
                            double distance,
                            double inner_cutoff,
                            double outer_cutoff)
{
	return ComputeZBLPair(atomic_number_i, atomic_number_j, distance,
	                      inner_cutoff, outer_cutoff, 0.0);
}

ZBLPairConstants MakeZBLPairConstants(int atomic_number_i, int atomic_number_j)
{
	if (atomic_number_i <= 0 || atomic_number_j <= 0)
		ERROR("ZBL atomic numbers should be positive.");
	const double ev_angstrom_per_e2 = 14.3996454784255;
	const double screening_inv =
		2.134563 * (std::pow(static_cast<double>(atomic_number_i), 0.23) +
		            std::pow(static_cast<double>(atomic_number_j), 0.23));
	const double prefactor = ev_angstrom_per_e2 *
		static_cast<double>(atomic_number_i) *
		static_cast<double>(atomic_number_j);
	return ZBLPairConstants{screening_inv, prefactor};
}

ZBLPairValue ComputeZBLPairCached(const ZBLPairConstants& constants,
                                  double distance,
                                  double inner_cutoff,
                                  double outer_cutoff)
{
	if (inner_cutoff < 0.0)
		ERROR("ZBL inner cutoff should be non-negative.");
	if (outer_cutoff <= 0.0)
		ERROR("ZBL outer cutoff should be positive.");
	if (distance <= 0.0)
		ERROR("ZBL pair distance should be positive.");
	if (outer_cutoff <= inner_cutoff)
		ERROR("ZBL cutoffs should satisfy 0 <= inner < outer.");

	if (distance >= outer_cutoff)
		return ZBLPairValue{0.0, 0.0};

	const double coefficients[4] = {0.18175, 0.50986, 0.28022, 0.02817};
	const double exponents[4] = {3.1998, 0.94229, 0.4029, 0.20162};
	const double x = constants.screening_inv * distance;

	double phi = 0.0;
	double dphi_dr = 0.0;
	for (int i = 0; i < 4; ++i) {
		const double exp_value = std::exp(-exponents[i] * x);
		phi += coefficients[i] * exp_value;
		dphi_dr -= coefficients[i] * exponents[i] *
			constants.screening_inv * exp_value;
	}

	const double base_energy = constants.prefactor * phi / distance;
	const double base_dEdr = constants.prefactor *
		(dphi_dr / distance - phi / (distance * distance));

	double switch_value = 1.0;
	double switch_derivative = 0.0;
	if (distance > inner_cutoff) {
		const double pi_factor = std::acos(-1.0) / (outer_cutoff - inner_cutoff);
		switch_value = 0.5 * std::cos(pi_factor * (distance - inner_cutoff)) + 0.5;
		switch_derivative = -0.5 * pi_factor *
			std::sin(pi_factor * (distance - inner_cutoff));
	}

	ZBLPairValue value;
	value.energy = switch_value * base_energy;
	value.dEdr = switch_value * base_dEdr + switch_derivative * base_energy;
	return value;
}

ZBLPairValue ComputeZBLPair(int atomic_number_i,
                            int atomic_number_j,
                            double distance,
                            double inner_cutoff,
                            double outer_cutoff,
                            double typewise_cutoff_factor)
{
	if (atomic_number_i <= 0 || atomic_number_j <= 0)
		ERROR("ZBL atomic numbers should be positive.");
	if (inner_cutoff < 0.0)
		ERROR("ZBL inner cutoff should be non-negative.");
	if (outer_cutoff <= 0.0)
		ERROR("ZBL outer cutoff should be positive.");
	if (distance <= 0.0)
		ERROR("ZBL pair distance should be positive.");

	if (UseTypewiseZBLCutoff(typewise_cutoff_factor)) {
		inner_cutoff = 0.0;
		outer_cutoff = ZBLTypewiseOuterCutoff(atomic_number_i, atomic_number_j,
		                                      outer_cutoff, typewise_cutoff_factor);
	} else if (outer_cutoff <= inner_cutoff) {
		ERROR("ZBL cutoffs should satisfy 0 <= inner < outer.");
	}

	if (distance >= outer_cutoff)
		return ZBLPairValue{0.0, 0.0};

	return ComputeZBLPairCached(MakeZBLPairConstants(atomic_number_i, atomic_number_j),
	                            distance, inner_cutoff, outer_cutoff);
}
