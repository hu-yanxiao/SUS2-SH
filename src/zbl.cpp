#include "zbl.h"

#include <algorithm>
#include <cmath>

#include "common/utils.h"

namespace {

void ZeroMatrix(Matrix3& matrix)
{
	for (int a = 0; a < 3; ++a)
		for (int b = 0; b < 3; ++b)
			matrix[a][b] = 0.0;
}

} // namespace

ZBLEFS::ZBLEFS()
	: energy(0.0)
{
	ZeroMatrix(stresses);
}

ZBLEFS::ZBLEFS(int atom_count)
	: energy(0.0)
{
	Resize(atom_count);
}

void ZBLEFS::Resize(int atom_count)
{
	energy = 0.0;
	forces.assign(atom_count, Vector3(0.0, 0.0, 0.0));
	ZeroMatrix(stresses);
}

ZBLPotential::ZBLPotential()
	: enabled_(false),
	  inner_cutoff_(DefaultZBLInnerCutoff()),
	  outer_cutoff_(DefaultZBLOuterCutoff()),
	  max_outer_cutoff_(DefaultZBLOuterCutoff()),
	  typewise_cutoff_enabled_(false),
	  typewise_cutoff_factor_(DefaultZBLTypewiseCutoffFactor())
{
}

bool ZBLPotential::Enabled() const
{
	return enabled_;
}

double ZBLPotential::InnerCutoff() const
{
	return inner_cutoff_;
}

double ZBLPotential::OuterCutoff() const
{
	return outer_cutoff_;
}

double ZBLPotential::MaxOuterCutoff() const
{
	return max_outer_cutoff_;
}

bool ZBLPotential::TypewiseCutoffEnabled() const
{
	return typewise_cutoff_enabled_;
}

double ZBLPotential::TypewiseCutoffFactor() const
{
	return typewise_cutoff_factor_;
}

const std::vector<int>& ZBLPotential::AtomicNumbers() const
{
	return atomic_numbers_;
}

void ZBLPotential::Clear()
{
	enabled_ = false;
	atomic_numbers_.clear();
	inner_cutoff_ = DefaultZBLInnerCutoff();
	outer_cutoff_ = DefaultZBLOuterCutoff();
	max_outer_cutoff_ = DefaultZBLOuterCutoff();
	typewise_cutoff_enabled_ = false;
	typewise_cutoff_factor_ = DefaultZBLTypewiseCutoffFactor();
	pair_inner_cutoffs_.clear();
	pair_outer_cutoffs_.clear();
	pair_outer_sq_.clear();
	pair_constants_.clear();
}

void ZBLPotential::Configure(const std::vector<int>& atomic_numbers,
                             double inner_cutoff,
                             double outer_cutoff,
                             bool typewise_cutoff_enabled,
                             double typewise_cutoff_factor)
{
	if (atomic_numbers.empty())
		ERROR("ZBL requires one atomic number per model species.");
	if (inner_cutoff < 0.0)
		ERROR("ZBL inner cutoff should be non-negative.");
	if (outer_cutoff <= 0.0)
		ERROR("ZBL outer cutoff should be positive.");
	if (typewise_cutoff_enabled) {
		if (typewise_cutoff_factor < 0.5)
			ERROR("ZBL typewise cutoff factor should be at least 0.5.");
	} else if (outer_cutoff <= inner_cutoff) {
		ERROR("ZBL cutoffs should satisfy 0 <= inner < outer.");
	}
	for (int atomic_number : atomic_numbers) {
		if (atomic_number <= 0)
			ERROR("ZBL atomic numbers should be positive.");
		if (typewise_cutoff_enabled)
			ZBLCovalentRadius(atomic_number);
	}

	atomic_numbers_ = atomic_numbers;
	inner_cutoff_ = typewise_cutoff_enabled ? 0.0 : inner_cutoff;
	outer_cutoff_ = outer_cutoff;
	typewise_cutoff_enabled_ = typewise_cutoff_enabled;
	typewise_cutoff_factor_ = typewise_cutoff_factor;
	FillZBLPairCutoffTables(atomic_numbers_, inner_cutoff, outer_cutoff,
	                        typewise_cutoff_enabled_,
	                        typewise_cutoff_enabled_ ? typewise_cutoff_factor_ : 0.0,
	                        pair_inner_cutoffs_, pair_outer_cutoffs_, pair_outer_sq_);
	max_outer_cutoff_ = 0.0;
	for (double pair_outer : pair_outer_cutoffs_)
		max_outer_cutoff_ = std::max(max_outer_cutoff_, pair_outer);
	if (max_outer_cutoff_ <= 0.0)
		ERROR("ZBL max pair outer cutoff should be positive.");
	pair_constants_.resize(static_cast<size_t>(atomic_numbers_.size()) *
	                       atomic_numbers_.size());
	for (int i = 0; i < static_cast<int>(atomic_numbers_.size()); ++i)
		for (int j = 0; j < static_cast<int>(atomic_numbers_.size()); ++j)
			pair_constants_[static_cast<size_t>(i) * atomic_numbers_.size() + j] =
				MakeZBLPairConstants(atomic_numbers_[i], atomic_numbers_[j]);
	enabled_ = true;
}

ZBLEFS ZBLPotential::Compute(const Configuration& cfg) const
{
	Neighborhoods neighborhoods(cfg, max_outer_cutoff_);
	return Compute(cfg, neighborhoods);
}

ZBLEFS ZBLPotential::Compute(const Configuration& cfg, const Neighborhoods& neighborhoods) const
{
	ZBLEFS zbl(cfg.size());
	if (!enabled_)
		return zbl;

	if (static_cast<int>(atomic_numbers_.size()) <= 0)
		ERROR("ZBL atomic number map is empty.");
	const int species_count = static_cast<int>(atomic_numbers_.size());
	if (static_cast<int>(pair_outer_cutoffs_.size()) != species_count * species_count ||
	    static_cast<int>(pair_inner_cutoffs_.size()) != species_count * species_count ||
	    static_cast<int>(pair_constants_.size()) != species_count * species_count)
		ERROR("ZBL pair cutoff table is not initialized.");
	if (neighborhoods.cutoff + 1.0e-12 < max_outer_cutoff_)
		ERROR("ZBL neighborhood cutoff is smaller than the maximum pair outer cutoff.");

	for (int ind = 0; ind < cfg.size(); ++ind) {
		const int type_i = cfg.type(ind);
		if (type_i < 0 || type_i >= species_count)
			ERROR("ZBL atom type is outside the atomic number map.");
		const Neighborhood& nbh = neighborhoods[ind];

		for (int j = 0; j < nbh.count; ++j) {
			const int neighbor = nbh.inds[j];
			const int type_j = cfg.type(neighbor);
			if (type_j < 0 || type_j >= species_count)
				ERROR("ZBL neighbor atom type is outside the atomic number map.");
			const int pair_index = type_i * species_count + type_j;
			if (nbh.dists[j] >= pair_outer_cutoffs_[pair_index])
				continue;
			const ZBLPairValue pair =
				ComputeZBLPairCached(pair_constants_[pair_index],
				                     nbh.dists[j],
				                     pair_inner_cutoffs_[pair_index],
				                     pair_outer_cutoffs_[pair_index]);
			if (pair.energy == 0.0 && pair.dEdr == 0.0)
				continue;

			const double inv_r = 1.0 / nbh.dists[j];
			Vector3 force = nbh.vecs[j] * (0.5 * pair.dEdr * inv_r);
			zbl.energy += 0.5 * pair.energy;
			zbl.forces[ind] += force;
			zbl.forces[neighbor] -= force;
			for (int a = 0; a < 3; ++a)
				for (int b = 0; b < 3; ++b)
					zbl.stresses[a][b] -= force[a] * nbh.vecs[j][b];
		}
	}
	return zbl;
}

void ZBLPotential::AddTo(Configuration& cfg) const
{
	Neighborhoods neighborhoods(cfg, max_outer_cutoff_);
	AddTo(cfg, neighborhoods);
}

void ZBLPotential::AddTo(Configuration& cfg, const Neighborhoods& neighborhoods) const
{
	if (!enabled_)
		return;
	if (static_cast<int>(atomic_numbers_.size()) <= 0)
		ERROR("ZBL atomic number map is empty.");
	const int species_count = static_cast<int>(atomic_numbers_.size());
	if (static_cast<int>(pair_outer_cutoffs_.size()) != species_count * species_count ||
	    static_cast<int>(pair_inner_cutoffs_.size()) != species_count * species_count ||
	    static_cast<int>(pair_constants_.size()) != species_count * species_count)
		ERROR("ZBL pair cutoff table is not initialized.");
	if (neighborhoods.cutoff + 1.0e-12 < max_outer_cutoff_)
		ERROR("ZBL neighborhood cutoff is smaller than the maximum pair outer cutoff.");

	const bool add_energy = cfg.has_energy();
	const bool add_forces = cfg.has_forces();
	const bool add_stresses = cfg.has_stresses();
	for (int ind = 0; ind < cfg.size(); ++ind) {
		const int type_i = cfg.type(ind);
		if (type_i < 0 || type_i >= species_count)
			ERROR("ZBL atom type is outside the atomic number map.");
		const Neighborhood& nbh = neighborhoods[ind];

		for (int j = 0; j < nbh.count; ++j) {
			const int neighbor = nbh.inds[j];
			const int type_j = cfg.type(neighbor);
			if (type_j < 0 || type_j >= species_count)
				ERROR("ZBL neighbor atom type is outside the atomic number map.");
			const int pair_index = type_i * species_count + type_j;
			if (nbh.dists[j] >= pair_outer_cutoffs_[pair_index])
				continue;
			const ZBLPairValue pair =
				ComputeZBLPairCached(pair_constants_[pair_index],
				                     nbh.dists[j],
				                     pair_inner_cutoffs_[pair_index],
				                     pair_outer_cutoffs_[pair_index]);
			if (pair.energy == 0.0 && pair.dEdr == 0.0)
				continue;

			const double inv_r = 1.0 / nbh.dists[j];
			const Vector3 force = nbh.vecs[j] * (0.5 * pair.dEdr * inv_r);
			if (add_energy)
				cfg.energy += 0.5 * pair.energy;
			if (add_forces) {
				cfg.force(ind) += force;
				cfg.force(neighbor) -= force;
			}
			if (add_stresses)
				for (int a = 0; a < 3; ++a)
					for (int b = 0; b < 3; ++b)
						cfg.stresses[a][b] -= force[a] * nbh.vecs[j][b];
		}
	}
}
