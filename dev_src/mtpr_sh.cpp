/* SUS2-SH real spherical-harmonic evaluation and analytic coefficient gradients. */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "mtpr.h"

using namespace std;

namespace {

const double kPi = std::acos(-1.0);
const double kSqrt2 = std::sqrt(2.0);
const int kMaxSHL = 6;
const int kMaxSHComponents = (kMaxSHL + 1) * (kMaxSHL + 1);

const double kRealY00 = 0.5 / std::sqrt(kPi);
const double kRealY1 = 0.5 * std::sqrt(3.0 / kPi);
const double kRealY2A = 0.5 * std::sqrt(15.0 / kPi);
const double kRealY20 = 0.25 * std::sqrt(5.0 / kPi);
const double kRealY22 = 0.25 * std::sqrt(15.0 / kPi);
const double kRealY33 = 0.125 * std::sqrt(70.0 / kPi);
const double kRealY32 = 0.5 * std::sqrt(105.0 / kPi);
const double kRealY31 = 0.125 * std::sqrt(42.0 / kPi);
const double kRealY30 = 0.25 * std::sqrt(7.0 / kPi);
const double kRealY3p2 = 0.25 * std::sqrt(105.0 / kPi);
const double kRealY44m = 0.75 * std::sqrt(35.0 / kPi);
const double kRealY43 = 0.375 * std::sqrt(70.0 / kPi);
const double kRealY42m = 0.75 * std::sqrt(5.0 / kPi);
const double kRealY41 = 0.375 * std::sqrt(10.0 / kPi);
const double kRealY40 = 0.1875 / std::sqrt(kPi);
const double kRealY42 = 0.375 * std::sqrt(5.0 / kPi);
const double kRealY44 = 0.1875 * std::sqrt(35.0 / kPi);

int SHFlatIndex(int l, int m)
{
	return l * l + (m + l);
}

double ParitySign(int m)
{
	return (std::abs(m) % 2) == 0 ? 1.0 : -1.0;
}

double RealSHValue(const std::complex<double>* values, int l, int m)
{
	if (m == 0)
		return values[SHFlatIndex(l, 0)].real();
	if (m > 0)
		return kSqrt2 * ParitySign(m) * values[SHFlatIndex(l, m)].real();
	return kSqrt2 * ParitySign(m) * values[SHFlatIndex(l, -m)].imag();
}

double RealSHDer(const std::complex<double>* ders, int l, int m, int a)
{
	if (m == 0)
		return ders[3 * SHFlatIndex(l, 0) + a].real();
	if (m > 0)
		return kSqrt2 * ParitySign(m) * ders[3 * SHFlatIndex(l, m) + a].real();
	return kSqrt2 * ParitySign(m) * ders[3 * SHFlatIndex(l, -m) + a].imag();
}

double RealSHValue(const double* values, int l, int m)
{
	return values[SHFlatIndex(l, m)];
}

double RealSHDer(const double* ders, int l, int m, int a)
{
	return ders[3 * SHFlatIndex(l, m) + a];
}

bool SHUsesPrecomputedLmpTable(AnyRadialBasis* radial_basis)
{
	if (radial_basis == nullptr)
		return false;
	const std::string type = radial_basis->GetRBTypeString();
	return type == "RBChebyshev_sss_lmp"
	    || type == "RBChebyshev_sss_rational_lmp"
	    || type == "RBChebyshev_ss_lmp"
	    || type == "RBChebyshev_ssw_lmp"
	    || type == "RBChebyshev_sssw_lmp"
	    || type == "RBChebyshev_s_lmp"
	    || type == "RBLaguerre_log1p_lmp"
	    || type == "RBLaguerre_log1p_pos_lmp"
	    || type == "RBLaguerre_log1p_noenv_lmp"
	    || type == "RBJacobi_sss_lmp"
	    || type == "RBJacobi_sss_noweight_lmp";
}

bool DisableTwoLayerEdgePrimitiveCache()
{
	const char* value = std::getenv("SUS2_SH_DISABLE_TWO_LAYER_EDGE_CACHE");
	return value != nullptr && value[0] != '\0' && value[0] != '0';
}

void InterpolateSHLmpMuTable(Array3D& value_table,
                             Array3D& der_table,
                             double inv_dr,
                             int species_count,
                             int type_central,
                             int type_outer,
                             double r,
                             int radial_func_count,
                             double* values,
                             double* ders)
{
	if (value_table.size1 != species_count * species_count
	    || value_table.size2 < 2
	    || value_table.size3 < radial_func_count)
		ERROR("SUS2-SH _lmp radial value table is not initialized");
	if (ders != nullptr
	    && (der_table.size1 != value_table.size1
	        || der_table.size2 != value_table.size2
	        || der_table.size3 < radial_func_count))
		ERROR("SUS2-SH _lmp radial derivative table is not initialized");
	if (type_central < 0 || type_central >= species_count
	    || type_outer < 0 || type_outer >= species_count)
		ERROR("SUS2-SH _lmp radial table type index is out of range");
	const int pair_index = species_count * type_central + type_outer;
	double grid_pos = r * inv_dr;
	int r_list = static_cast<int>(std::floor(grid_pos));
	const int last_interval = value_table.size2 - 2;
	if (r_list < 0)
		r_list = 0;
	if (r_list > last_interval)
		r_list = last_interval;
	const int r_next = r_list + 1;
	double ddr = grid_pos - r_list;
	if (ddr < 0.0)
		ddr = 0.0;
	if (ddr > 1.0)
		ddr = 1.0;
	for (int mu = 0; mu < radial_func_count; ++mu) {
		const double v1 = value_table(pair_index, r_list, mu);
		const double v2 = value_table(pair_index, r_next, mu);
		values[mu] = v1 + ddr * (v2 - v1);
		if (ders != nullptr) {
			const double d1 = der_table(pair_index, r_list, mu);
			const double d2 = der_table(pair_index, r_next, mu);
			ders[mu] = d1 + ddr * (d2 - d1);
		}
	}
}

double SHInvPower(int l, double r)
{
	const double inv_r = 1.0 / r;
	const double inv_r2 = inv_r * inv_r;
	if (l == 0)
		return 1.0;
	if (l == 1)
		return inv_r;
	if (l == 2)
		return inv_r2;
	if (l == 3)
		return inv_r2 * inv_r;
	if (l == 4)
		return inv_r2 * inv_r2;
	return std::pow(r, -l);
}

void AddSHPolynomial(int l,
	                     int m,
	                     double coeff,
	                     const std::complex<double>& poly,
	                     const std::complex<double> dpoly[3],
                     const Vector3& rvec,
                     double r,
                     std::complex<double>* values,
                     std::complex<double>* ders)
{
	const int idx = SHFlatIndex(l, m);
	const double inv_r = 1.0 / r;
	const double inv_pow = SHInvPower(l, r);
	const double inv_pow_der = (l == 0) ? 0.0 : -static_cast<double>(l) * inv_pow * inv_r * inv_r;
	values[idx] = coeff * poly * inv_pow;
	for (int a = 0; a < 3; ++a)
		ders[3 * idx + a] = coeff * (dpoly[a] * inv_pow + poly * inv_pow_der * rvec[a]);
}

void AddSHPolynomialValueOnly(int l,
                              int m,
                              double coeff,
                              const std::complex<double>& poly,
                              double r,
                              std::complex<double>* values)
{
	values[SHFlatIndex(l, m)] = coeff * poly * SHInvPower(l, r);
}

void AddRealSH(int l,
               int m,
               double coeff,
               double poly,
               double dpx,
               double dpy,
               double dpz,
               const Vector3& rvec,
               double r,
               double* values,
               double* ders)
{
	const int idx = SHFlatIndex(l, m);
	const double inv_r = 1.0 / r;
	const double inv_pow = SHInvPower(l, r);
	const double inv_pow_der = (l == 0) ? 0.0 : -static_cast<double>(l) * inv_pow * inv_r * inv_r;
	values[idx] = coeff * poly * inv_pow;
	ders[3 * idx + 0] = coeff * (dpx * inv_pow + poly * inv_pow_der * rvec[0]);
	ders[3 * idx + 1] = coeff * (dpy * inv_pow + poly * inv_pow_der * rvec[1]);
	ders[3 * idx + 2] = coeff * (dpz * inv_pow + poly * inv_pow_der * rvec[2]);
}

void AddRealSHValueOnly(int l, int m, double coeff, double poly, double r, double* values)
{
	values[SHFlatIndex(l, m)] = coeff * poly * SHInvPower(l, r);
}

double Factorial(int n)
{
	double value = 1.0;
	for (int i = 2; i <= n; ++i)
		value *= static_cast<double>(i);
	return value;
}

double ComplexSHNorm(int l, int m)
{
	return std::sqrt((2.0 * l + 1.0) / (4.0 * kPi) * Factorial(l - m) / Factorial(l + m));
}

void EvalRealSHFromSolid(const Vector3& rvec,
                         double r,
                         int lmax,
                         double* values,
                         double* ders)
{
	const int count = (lmax + 1) * (lmax + 1);
	for (int i = 0; i < count; ++i) {
		values[i] = 0.0;
		for (int a = 0; a < 3; ++a)
			ders[3 * i + a] = 0.0;
	}

	std::complex<double> solid[kMaxSHComponents];
	std::complex<double> solid_ders[3 * kMaxSHComponents];
	for (int i = 0; i < kMaxSHComponents; ++i) {
		solid[i] = std::complex<double>(0.0, 0.0);
		for (int a = 0; a < 3; ++a)
			solid_ders[3 * i + a] = std::complex<double>(0.0, 0.0);
	}

	const double x = rvec[0];
	const double y = rvec[1];
	const double z = rvec[2];
	const double r2 = r * r;
	const std::complex<double> u(x, y);
	const std::complex<double> du[3] = {
		std::complex<double>(1.0, 0.0),
		std::complex<double>(0.0, 1.0),
		std::complex<double>(0.0, 0.0)
	};

	solid[SHFlatIndex(0, 0)] = std::complex<double>(1.0, 0.0);
	for (int m = 1; m <= lmax; ++m) {
		const int prev = SHFlatIndex(m - 1, m - 1);
		const int idx = SHFlatIndex(m, m);
		const double coeff = -static_cast<double>(2 * m - 1);
		solid[idx] = coeff * u * solid[prev];
		for (int a = 0; a < 3; ++a)
			solid_ders[3 * idx + a] = coeff * (du[a] * solid[prev] + u * solid_ders[3 * prev + a]);
	}
	for (int m = 0; m <= lmax; ++m) {
		const int diag = SHFlatIndex(m, m);
		if (m + 1 <= lmax) {
			const int idx = SHFlatIndex(m + 1, m);
			const double coeff = static_cast<double>(2 * m + 1);
			solid[idx] = coeff * z * solid[diag];
			for (int a = 0; a < 3; ++a)
				solid_ders[3 * idx + a] = coeff * (
					(a == 2 ? solid[diag] : std::complex<double>(0.0, 0.0))
					+ z * solid_ders[3 * diag + a]);
		}
		for (int l = m + 2; l <= lmax; ++l) {
			const int idx = SHFlatIndex(l, m);
			const int prev1 = SHFlatIndex(l - 1, m);
			const int prev2 = SHFlatIndex(l - 2, m);
			const double acoef = static_cast<double>(2 * l - 1);
			const double bcoef = static_cast<double>(l + m - 1);
			const double denom = static_cast<double>(l - m);
			solid[idx] = (acoef * z * solid[prev1] - bcoef * r2 * solid[prev2]) / denom;
			for (int a = 0; a < 3; ++a) {
				const double dr2 = 2.0 * rvec[a];
				solid_ders[3 * idx + a] = (
					acoef * ((a == 2 ? solid[prev1] : std::complex<double>(0.0, 0.0))
					         + z * solid_ders[3 * prev1 + a])
					- bcoef * (dr2 * solid[prev2] + r2 * solid_ders[3 * prev2 + a])) / denom;
			}
		}
	}

	for (int l = 0; l <= lmax; ++l) {
		const double inv_pow = SHInvPower(l, r);
		const double inv_pow_der = (l == 0) ? 0.0 : -static_cast<double>(l) * inv_pow / (r * r);
		for (int m = 0; m <= l; ++m) {
			const int cidx = SHFlatIndex(l, m);
			const double norm = ComplexSHNorm(l, m);
			const std::complex<double> y_complex = norm * solid[cidx] * inv_pow;
			std::complex<double> dy_complex[3];
			for (int a = 0; a < 3; ++a)
				dy_complex[a] = norm * (solid_ders[3 * cidx + a] * inv_pow
				                         + solid[cidx] * inv_pow_der * rvec[a]);

			if (m == 0) {
				const int ridx = SHFlatIndex(l, 0);
				values[ridx] = y_complex.real();
				for (int a = 0; a < 3; ++a)
					ders[3 * ridx + a] = dy_complex[a].real();
			} else {
				const double sign = ParitySign(m);
				const int pidx = SHFlatIndex(l, m);
				const int nidx = SHFlatIndex(l, -m);
				values[pidx] = kSqrt2 * sign * y_complex.real();
				values[nidx] = kSqrt2 * sign * y_complex.imag();
				for (int a = 0; a < 3; ++a) {
					ders[3 * pidx + a] = kSqrt2 * sign * dy_complex[a].real();
					ders[3 * nidx + a] = kSqrt2 * sign * dy_complex[a].imag();
				}
				}
		}
	}
}

void EvalRealSHValuesOnlyFromSolid(const Vector3& rvec,
                                   double r,
                                   int lmax,
                                   double* values)
{
	double ders[3 * kMaxSHComponents];
	EvalRealSHFromSolid(rvec, r, lmax, values, ders);
}

void EvalComplexSH(const Vector3& rvec,
                   double r,
                   int lmax,
                   std::complex<double>* values,
                   std::complex<double>* ders)
{
	const int count = (lmax + 1) * (lmax + 1);
	for (int i = 0; i < count; ++i) {
		values[i] = std::complex<double>(0.0, 0.0);
		for (int a = 0; a < 3; ++a)
			ders[3 * i + a] = std::complex<double>(0.0, 0.0);
	}

	const double x = rvec[0];
	const double y = rvec[1];
	const double z = rvec[2];
	const std::complex<double> I(0.0, 1.0);
	const std::complex<double> u(x, y);
	const std::complex<double> v(x, -y);
	std::complex<double> dpoly[3];

	dpoly[0] = dpoly[1] = dpoly[2] = 0.0;
	AddSHPolynomial(0, 0, 0.5 / std::sqrt(kPi), 1.0, dpoly, rvec, r, values, ders);
	if (lmax == 0)
		return;

	dpoly[0] = 1.0; dpoly[1] = -I; dpoly[2] = 0.0;
	AddSHPolynomial(1, -1, 0.5 * std::sqrt(3.0 / (2.0 * kPi)), v, dpoly, rvec, r, values, ders);
	dpoly[0] = 0.0; dpoly[1] = 0.0; dpoly[2] = 1.0;
	AddSHPolynomial(1, 0, 0.5 * std::sqrt(3.0 / kPi), z, dpoly, rvec, r, values, ders);
	dpoly[0] = 1.0; dpoly[1] = I; dpoly[2] = 0.0;
	AddSHPolynomial(1, 1, -0.5 * std::sqrt(3.0 / (2.0 * kPi)), u, dpoly, rvec, r, values, ders);
	if (lmax == 1)
		return;

	const std::complex<double> u2 = u * u;
	const std::complex<double> v2 = v * v;
	dpoly[0] = 2.0 * v; dpoly[1] = -2.0 * I * v; dpoly[2] = 0.0;
	AddSHPolynomial(2, -2, 0.25 * std::sqrt(15.0 / (2.0 * kPi)), v2, dpoly, rvec, r, values, ders);
	dpoly[0] = z; dpoly[1] = -I * z; dpoly[2] = v;
	AddSHPolynomial(2, -1, 0.5 * std::sqrt(15.0 / (2.0 * kPi)), z * v, dpoly, rvec, r, values, ders);
	const std::complex<double> p20(2.0 * z * z - x * x - y * y, 0.0);
	dpoly[0] = -2.0 * x; dpoly[1] = -2.0 * y; dpoly[2] = 4.0 * z;
	AddSHPolynomial(2, 0, 0.25 * std::sqrt(5.0 / kPi), p20, dpoly, rvec, r, values, ders);
	dpoly[0] = z; dpoly[1] = I * z; dpoly[2] = u;
	AddSHPolynomial(2, 1, -0.5 * std::sqrt(15.0 / (2.0 * kPi)), z * u, dpoly, rvec, r, values, ders);
	dpoly[0] = 2.0 * u; dpoly[1] = 2.0 * I * u; dpoly[2] = 0.0;
	AddSHPolynomial(2, 2, 0.25 * std::sqrt(15.0 / (2.0 * kPi)), u2, dpoly, rvec, r, values, ders);
	if (lmax == 2)
		return;

	const std::complex<double> u3 = u2 * u;
	const std::complex<double> v3 = v2 * v;
	dpoly[0] = 3.0 * v2; dpoly[1] = -3.0 * I * v2; dpoly[2] = 0.0;
	AddSHPolynomial(3, -3, 0.125 * std::sqrt(35.0 / kPi), v3, dpoly, rvec, r, values, ders);
	dpoly[0] = 2.0 * z * v; dpoly[1] = -2.0 * I * z * v; dpoly[2] = v2;
	AddSHPolynomial(3, -2, 0.25 * std::sqrt(105.0 / (2.0 * kPi)), z * v2, dpoly, rvec, r, values, ders);
	const std::complex<double> a31(4.0 * z * z - x * x - y * y, 0.0);
	dpoly[0] = a31 - 2.0 * x * v; dpoly[1] = -I * a31 - 2.0 * y * v; dpoly[2] = 8.0 * z * v;
	AddSHPolynomial(3, -1, 0.125 * std::sqrt(21.0 / kPi), v * a31, dpoly, rvec, r, values, ders);
	const std::complex<double> p30(z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y), 0.0);
	dpoly[0] = -6.0 * x * z; dpoly[1] = -6.0 * y * z; dpoly[2] = 6.0 * z * z - 3.0 * x * x - 3.0 * y * y;
	AddSHPolynomial(3, 0, 0.25 * std::sqrt(7.0 / kPi), p30, dpoly, rvec, r, values, ders);
	dpoly[0] = a31 - 2.0 * x * u; dpoly[1] = I * a31 - 2.0 * y * u; dpoly[2] = 8.0 * z * u;
	AddSHPolynomial(3, 1, -0.125 * std::sqrt(21.0 / kPi), u * a31, dpoly, rvec, r, values, ders);
	dpoly[0] = 2.0 * z * u; dpoly[1] = 2.0 * I * z * u; dpoly[2] = u2;
	AddSHPolynomial(3, 2, 0.25 * std::sqrt(105.0 / (2.0 * kPi)), z * u2, dpoly, rvec, r, values, ders);
	dpoly[0] = 3.0 * u2; dpoly[1] = 3.0 * I * u2; dpoly[2] = 0.0;
	AddSHPolynomial(3, 3, -0.125 * std::sqrt(35.0 / kPi), u3, dpoly, rvec, r, values, ders);
	if (lmax == 3)
		return;

	const double rho2 = x * x + y * y;
	const std::complex<double> u4 = u3 * u;
	const std::complex<double> v4 = v3 * v;

	dpoly[0] = 4.0 * v3; dpoly[1] = -4.0 * I * v3; dpoly[2] = 0.0;
	AddSHPolynomial(4, -4, 0.1875 * std::sqrt(35.0 / (2.0 * kPi)), v4, dpoly, rvec, r, values, ders);
	dpoly[0] = 3.0 * z * v2; dpoly[1] = -3.0 * I * z * v2; dpoly[2] = v3;
	AddSHPolynomial(4, -3, 0.375 * std::sqrt(35.0 / kPi), z * v3, dpoly, rvec, r, values, ders);
	const std::complex<double> a42(6.0 * z * z - rho2, 0.0);
	dpoly[0] = 2.0 * v * a42 - 2.0 * x * v2;
	dpoly[1] = -2.0 * I * v * a42 - 2.0 * y * v2;
	dpoly[2] = 12.0 * z * v2;
	AddSHPolynomial(4, -2, 0.375 * std::sqrt(5.0 / (2.0 * kPi)), v2 * a42, dpoly, rvec, r, values, ders);
	const std::complex<double> a41(4.0 * z * z - 3.0 * rho2, 0.0);
	dpoly[0] = z * (a41 - 6.0 * x * v);
	dpoly[1] = z * (-I * a41 - 6.0 * y * v);
	dpoly[2] = v * (12.0 * z * z - 3.0 * rho2);
	AddSHPolynomial(4, -1, 0.375 * std::sqrt(5.0 / kPi), z * v * a41, dpoly, rvec, r, values, ders);
	const std::complex<double> p40(8.0 * z * z * z * z - 24.0 * z * z * rho2 + 3.0 * rho2 * rho2, 0.0);
	dpoly[0] = 12.0 * x * (rho2 - 4.0 * z * z);
	dpoly[1] = 12.0 * y * (rho2 - 4.0 * z * z);
	dpoly[2] = 16.0 * z * (2.0 * z * z - 3.0 * rho2);
	AddSHPolynomial(4, 0, 0.1875 / std::sqrt(kPi), p40, dpoly, rvec, r, values, ders);
	dpoly[0] = z * (a41 - 6.0 * x * u);
	dpoly[1] = z * (I * a41 - 6.0 * y * u);
	dpoly[2] = u * (12.0 * z * z - 3.0 * rho2);
	AddSHPolynomial(4, 1, -0.375 * std::sqrt(5.0 / kPi), z * u * a41, dpoly, rvec, r, values, ders);
	dpoly[0] = 2.0 * u * a42 - 2.0 * x * u2;
	dpoly[1] = 2.0 * I * u * a42 - 2.0 * y * u2;
	dpoly[2] = 12.0 * z * u2;
	AddSHPolynomial(4, 2, 0.375 * std::sqrt(5.0 / (2.0 * kPi)), u2 * a42, dpoly, rvec, r, values, ders);
	dpoly[0] = 3.0 * z * u2; dpoly[1] = 3.0 * I * z * u2; dpoly[2] = u3;
	AddSHPolynomial(4, 3, -0.375 * std::sqrt(35.0 / kPi), z * u3, dpoly, rvec, r, values, ders);
	dpoly[0] = 4.0 * u3; dpoly[1] = 4.0 * I * u3; dpoly[2] = 0.0;
	AddSHPolynomial(4, 4, 0.1875 * std::sqrt(35.0 / (2.0 * kPi)), u4, dpoly, rvec, r, values, ders);
}

void EvalComplexSHValuesOnly(const Vector3& rvec,
                             double r,
                             int lmax,
                             std::complex<double>* values)
{
	const int count = (lmax + 1) * (lmax + 1);
	for (int i = 0; i < count; ++i)
		values[i] = std::complex<double>(0.0, 0.0);

	const double x = rvec[0];
	const double y = rvec[1];
	const double z = rvec[2];
	const std::complex<double> u(x, y);
	const std::complex<double> v(x, -y);

	AddSHPolynomialValueOnly(0, 0, 0.5 / std::sqrt(kPi), 1.0, r, values);
	if (lmax == 0)
		return;

	AddSHPolynomialValueOnly(1, -1, 0.5 * std::sqrt(3.0 / (2.0 * kPi)), v, r, values);
	AddSHPolynomialValueOnly(1, 0, 0.5 * std::sqrt(3.0 / kPi), z, r, values);
	AddSHPolynomialValueOnly(1, 1, -0.5 * std::sqrt(3.0 / (2.0 * kPi)), u, r, values);
	if (lmax == 1)
		return;

	const std::complex<double> u2 = u * u;
	const std::complex<double> v2 = v * v;
	AddSHPolynomialValueOnly(2, -2, 0.25 * std::sqrt(15.0 / (2.0 * kPi)), v2, r, values);
	AddSHPolynomialValueOnly(2, -1, 0.5 * std::sqrt(15.0 / (2.0 * kPi)), z * v, r, values);
	const std::complex<double> p20(2.0 * z * z - x * x - y * y, 0.0);
	AddSHPolynomialValueOnly(2, 0, 0.25 * std::sqrt(5.0 / kPi), p20, r, values);
	AddSHPolynomialValueOnly(2, 1, -0.5 * std::sqrt(15.0 / (2.0 * kPi)), z * u, r, values);
	AddSHPolynomialValueOnly(2, 2, 0.25 * std::sqrt(15.0 / (2.0 * kPi)), u2, r, values);
	if (lmax == 2)
		return;

	const std::complex<double> u3 = u2 * u;
	const std::complex<double> v3 = v2 * v;
	AddSHPolynomialValueOnly(3, -3, 0.125 * std::sqrt(35.0 / kPi), v3, r, values);
	AddSHPolynomialValueOnly(3, -2, 0.25 * std::sqrt(105.0 / (2.0 * kPi)), z * v2, r, values);
	const std::complex<double> a31(4.0 * z * z - x * x - y * y, 0.0);
	AddSHPolynomialValueOnly(3, -1, 0.125 * std::sqrt(21.0 / kPi), v * a31, r, values);
	const std::complex<double> p30(z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y), 0.0);
	AddSHPolynomialValueOnly(3, 0, 0.25 * std::sqrt(7.0 / kPi), p30, r, values);
	AddSHPolynomialValueOnly(3, 1, -0.125 * std::sqrt(21.0 / kPi), u * a31, r, values);
	AddSHPolynomialValueOnly(3, 2, 0.25 * std::sqrt(105.0 / (2.0 * kPi)), z * u2, r, values);
	AddSHPolynomialValueOnly(3, 3, -0.125 * std::sqrt(35.0 / kPi), u3, r, values);
	if (lmax == 3)
		return;

	const double rho2 = x * x + y * y;
	const std::complex<double> u4 = u3 * u;
	const std::complex<double> v4 = v3 * v;
	AddSHPolynomialValueOnly(4, -4, 0.1875 * std::sqrt(35.0 / (2.0 * kPi)), v4, r, values);
	AddSHPolynomialValueOnly(4, -3, 0.375 * std::sqrt(35.0 / kPi), z * v3, r, values);
	const std::complex<double> a42(6.0 * z * z - rho2, 0.0);
	AddSHPolynomialValueOnly(4, -2, 0.375 * std::sqrt(5.0 / (2.0 * kPi)), v2 * a42, r, values);
	const std::complex<double> a41(4.0 * z * z - 3.0 * rho2, 0.0);
	AddSHPolynomialValueOnly(4, -1, 0.375 * std::sqrt(5.0 / kPi), z * v * a41, r, values);
	const std::complex<double> p40(8.0 * z * z * z * z - 24.0 * z * z * rho2 + 3.0 * rho2 * rho2, 0.0);
	AddSHPolynomialValueOnly(4, 0, 0.1875 / std::sqrt(kPi), p40, r, values);
	AddSHPolynomialValueOnly(4, 1, -0.375 * std::sqrt(5.0 / kPi), z * u * a41, r, values);
	AddSHPolynomialValueOnly(4, 2, 0.375 * std::sqrt(5.0 / (2.0 * kPi)), u2 * a42, r, values);
	AddSHPolynomialValueOnly(4, 3, -0.375 * std::sqrt(35.0 / kPi), z * u3, r, values);
	AddSHPolynomialValueOnly(4, 4, 0.1875 * std::sqrt(35.0 / (2.0 * kPi)), u4, r, values);
}

void EvalRealSH(const Vector3& rvec,
                double r,
                int lmax,
                double* values,
                double* ders)
{
	if (lmax > 4) {
		EvalRealSHFromSolid(rvec, r, lmax, values, ders);
		return;
	}

	const int count = (lmax + 1) * (lmax + 1);
	for (int i = 0; i < count; ++i) {
		values[i] = 0.0;
		for (int a = 0; a < 3; ++a)
			ders[3 * i + a] = 0.0;
	}

	const double x = rvec[0];
	const double y = rvec[1];
	const double z = rvec[2];
	const double x2 = x * x;
	const double y2 = y * y;
	const double z2 = z * z;

	AddRealSH(0, 0, kRealY00, 1.0, 0.0, 0.0, 0.0, rvec, r, values, ders);
	if (lmax == 0)
		return;

	AddRealSH(1, -1, kRealY1, y, 0.0, 1.0, 0.0, rvec, r, values, ders);
	AddRealSH(1, 0, kRealY1, z, 0.0, 0.0, 1.0, rvec, r, values, ders);
	AddRealSH(1, 1, kRealY1, x, 1.0, 0.0, 0.0, rvec, r, values, ders);
	if (lmax == 1)
		return;

	AddRealSH(2, -2, kRealY2A, x * y, y, x, 0.0, rvec, r, values, ders);
	AddRealSH(2, -1, kRealY2A, y * z, 0.0, z, y, rvec, r, values, ders);
	const double p20 = 2.0 * z2 - x2 - y2;
	AddRealSH(2, 0, kRealY20, p20, -2.0 * x, -2.0 * y, 4.0 * z, rvec, r, values, ders);
	AddRealSH(2, 1, kRealY2A, x * z, z, 0.0, x, rvec, r, values, ders);
	AddRealSH(2, 2, kRealY22, x2 - y2, 2.0 * x, -2.0 * y, 0.0, rvec, r, values, ders);
	if (lmax == 2)
		return;

	const double a31 = 4.0 * z2 - x2 - y2;
	const double p3m3 = 3.0 * x2 * y - y * y2;
	AddRealSH(3, -3, kRealY33, p3m3, 6.0 * x * y, 3.0 * x2 - 3.0 * y2, 0.0, rvec, r, values, ders);
	AddRealSH(3, -2, kRealY32, x * y * z, y * z, x * z, x * y, rvec, r, values, ders);
	AddRealSH(3, -1, kRealY31, y * a31, -2.0 * x * y, a31 - 2.0 * y2, 8.0 * y * z, rvec, r, values, ders);
	const double p30 = z * (2.0 * z2 - 3.0 * x2 - 3.0 * y2);
	AddRealSH(3, 0, kRealY30, p30, -6.0 * x * z, -6.0 * y * z, 6.0 * z2 - 3.0 * x2 - 3.0 * y2, rvec, r, values, ders);
	AddRealSH(3, 1, kRealY31, x * a31, a31 - 2.0 * x2, -2.0 * x * y, 8.0 * x * z, rvec, r, values, ders);
	const double p32 = z * (x2 - y2);
	AddRealSH(3, 2, kRealY3p2, p32, 2.0 * x * z, -2.0 * y * z, x2 - y2, rvec, r, values, ders);
	const double p33 = x * x2 - 3.0 * x * y2;
	AddRealSH(3, 3, kRealY33, p33, 3.0 * x2 - 3.0 * y2, -6.0 * x * y, 0.0, rvec, r, values, ders);
	if (lmax == 3)
		return;

	const double rho2 = x2 + y2;
	const double a42 = 6.0 * z2 - rho2;
	const double a41 = 4.0 * z2 - 3.0 * rho2;
	const double p44base = x2 - y2;
	const double p4m4 = x * y * p44base;
	AddRealSH(4, -4, kRealY44m, p4m4, y * (3.0 * x2 - y2), x * (x2 - 3.0 * y2), 0.0, rvec, r, values, ders);
	AddRealSH(4, -3, kRealY43, z * p3m3, 6.0 * x * y * z, z * (3.0 * x2 - 3.0 * y2), p3m3, rvec, r, values, ders);
	AddRealSH(4, -2, kRealY42m, x * y * a42, y * a42 - 2.0 * x2 * y, x * a42 - 2.0 * x * y2, 12.0 * x * y * z, rvec, r, values, ders);
	AddRealSH(4, -1, kRealY41, y * z * a41, -6.0 * x * y * z, z * (a41 - 6.0 * y2), y * (12.0 * z2 - 3.0 * rho2), rvec, r, values, ders);
	const double p40 = 8.0 * z2 * z2 - 24.0 * z2 * rho2 + 3.0 * rho2 * rho2;
	AddRealSH(4, 0, kRealY40, p40, 12.0 * x * (rho2 - 4.0 * z2), 12.0 * y * (rho2 - 4.0 * z2), 16.0 * z * (2.0 * z2 - 3.0 * rho2), rvec, r, values, ders);
	AddRealSH(4, 1, kRealY41, x * z * a41, z * (a41 - 6.0 * x2), -6.0 * x * y * z, x * (12.0 * z2 - 3.0 * rho2), rvec, r, values, ders);
	AddRealSH(4, 2, kRealY42, p44base * a42, 2.0 * x * a42 - 2.0 * x * p44base, -2.0 * y * a42 - 2.0 * y * p44base, 12.0 * z * p44base, rvec, r, values, ders);
	AddRealSH(4, 3, kRealY43, z * p33, z * (3.0 * x2 - 3.0 * y2), -6.0 * x * y * z, p33, rvec, r, values, ders);
	const double p44 = x2 * x2 - 6.0 * x2 * y2 + y2 * y2;
	AddRealSH(4, 4, kRealY44, p44, 4.0 * x * x2 - 12.0 * x * y2, -12.0 * x2 * y + 4.0 * y * y2, 0.0, rvec, r, values, ders);
}

void EvalRealSHValuesOnly(const Vector3& rvec,
                          double r,
                          int lmax,
                          double* values)
{
	if (lmax > 4) {
		EvalRealSHValuesOnlyFromSolid(rvec, r, lmax, values);
		return;
	}

	const int count = (lmax + 1) * (lmax + 1);
	for (int i = 0; i < count; ++i)
		values[i] = 0.0;

	const double x = rvec[0];
	const double y = rvec[1];
	const double z = rvec[2];
	const double x2 = x * x;
	const double y2 = y * y;
	const double z2 = z * z;

	AddRealSHValueOnly(0, 0, kRealY00, 1.0, r, values);
	if (lmax == 0)
		return;
	AddRealSHValueOnly(1, -1, kRealY1, y, r, values);
	AddRealSHValueOnly(1, 0, kRealY1, z, r, values);
	AddRealSHValueOnly(1, 1, kRealY1, x, r, values);
	if (lmax == 1)
		return;
	AddRealSHValueOnly(2, -2, kRealY2A, x * y, r, values);
	AddRealSHValueOnly(2, -1, kRealY2A, y * z, r, values);
	AddRealSHValueOnly(2, 0, kRealY20, 2.0 * z2 - x2 - y2, r, values);
	AddRealSHValueOnly(2, 1, kRealY2A, x * z, r, values);
	AddRealSHValueOnly(2, 2, kRealY22, x2 - y2, r, values);
	if (lmax == 2)
		return;
	const double a31 = 4.0 * z2 - x2 - y2;
	const double p3m3 = 3.0 * x2 * y - y * y2;
	const double p33 = x * x2 - 3.0 * x * y2;
	AddRealSHValueOnly(3, -3, kRealY33, p3m3, r, values);
	AddRealSHValueOnly(3, -2, kRealY32, x * y * z, r, values);
	AddRealSHValueOnly(3, -1, kRealY31, y * a31, r, values);
	AddRealSHValueOnly(3, 0, kRealY30, z * (2.0 * z2 - 3.0 * x2 - 3.0 * y2), r, values);
	AddRealSHValueOnly(3, 1, kRealY31, x * a31, r, values);
	AddRealSHValueOnly(3, 2, kRealY3p2, z * (x2 - y2), r, values);
	AddRealSHValueOnly(3, 3, kRealY33, p33, r, values);
	if (lmax == 3)
		return;
	const double rho2 = x2 + y2;
	const double a42 = 6.0 * z2 - rho2;
	const double a41 = 4.0 * z2 - 3.0 * rho2;
	const double p44base = x2 - y2;
	AddRealSHValueOnly(4, -4, kRealY44m, x * y * p44base, r, values);
	AddRealSHValueOnly(4, -3, kRealY43, z * p3m3, r, values);
	AddRealSHValueOnly(4, -2, kRealY42m, x * y * a42, r, values);
	AddRealSHValueOnly(4, -1, kRealY41, y * z * a41, r, values);
	AddRealSHValueOnly(4, 0, kRealY40, 8.0 * z2 * z2 - 24.0 * z2 * rho2 + 3.0 * rho2 * rho2, r, values);
	AddRealSHValueOnly(4, 1, kRealY41, x * z * a41, r, values);
	AddRealSHValueOnly(4, 2, kRealY42, p44base * a42, r, values);
	AddRealSHValueOnly(4, 3, kRealY43, z * p33, r, values);
	AddRealSHValueOnly(4, 4, kRealY44, x2 * x2 - 6.0 * x2 * y2 + y2 * y2, r, values);
}

} // namespace

void MLMTPR::PrepareTwoLayerGateNeighborMuBuffers(int type_outer,
                                                  double center_type_coeff,
                                                  double outer_type_coeff,
                                                  double gate_residual)
{
	if (static_cast<int>(two_layer_gate_additive_mu_buffer_.size()) != radial_func_count)
		two_layer_gate_additive_mu_buffer_.resize(radial_func_count);
	if (static_cast<int>(two_layer_gate_type_scale_mu_buffer_.size()) != radial_func_count)
		two_layer_gate_type_scale_mu_buffer_.resize(radial_func_count);
	if (static_cast<int>(two_layer_gate_scaled_mu_vals_buffer_.size()) != radial_func_count)
		two_layer_gate_scaled_mu_vals_buffer_.resize(radial_func_count);
	if (static_cast<int>(two_layer_gate_scaled_mu_ders_buffer_.size()) != radial_func_count)
		two_layer_gate_scaled_mu_ders_buffer_.resize(radial_func_count);

	double* additive_by_mu = two_layer_gate_additive_mu_buffer_.data();
	double* type_scale_by_mu = two_layer_gate_type_scale_mu_buffer_.data();
	if (!two_layer_gate_enabled_) {
		const double pair_type_scale = center_type_coeff * outer_type_coeff;
		for (int mu = 0; mu < radial_func_count; ++mu) {
			additive_by_mu[mu] = 0.0;
			type_scale_by_mu[mu] = pair_type_scale;
		}
		return;
	}

	const int local_offset = type_outer * radial_func_count;
	const int coeff_offset = TwoLayerGateAdditiveCoeffOffset() + local_offset;
	const bool use_regression_coeffs =
		coeff_offset >= 0
		&& coeff_offset + radial_func_count <= static_cast<int>(regression_coeffs.size());
	const bool use_saved_coeffs =
		local_offset >= 0
		&& local_offset + radial_func_count
			<= static_cast<int>(two_layer_gate_additive_coeffs_.size());
	const double* additive_src = use_regression_coeffs
		? regression_coeffs.data() + coeff_offset
		: (use_saved_coeffs ? two_layer_gate_additive_coeffs_.data() + local_offset : nullptr);

	for (int mu = 0; mu < radial_func_count; ++mu) {
		const double additive_coeff =
			(additive_src == nullptr) ? TwoLayerGateAdditiveCoeff(type_outer, mu)
			                          : additive_src[mu];
		additive_by_mu[mu] = additive_coeff;
		type_scale_by_mu[mu] =
			center_type_coeff * (outer_type_coeff + additive_coeff * gate_residual);
	}
}

void MLMTPR::CalcSHBasisFuncs(const Neighborhood& nbh, double* bf_vals)
{
	CalcSHMomentValuesOnly(nbh);
	for (int i = 0; i < alpha_count; ++i)
		bf_vals[i] = basis_vals[i];
}

bool MLMTPR::UseSHProductRows() const
{
	if (sh_product_rows_.empty())
		return false;
	const char* env = std::getenv("SUS2_SH_PRODUCT_ROWS");
	if (env == nullptr)
		return false;
	const std::string value(env);
	return value == "1" || value == "true" || value == "True" || value == "on" || value == "ON";
}

bool MLMTPR::UseSHSiteDerivativeCache() const
{
	const char* env = std::getenv("SUS2_SH_SITE_DER_CACHE");
	if (env == nullptr)
		return false;
	const std::string value(env);
	return value != "0" && value != "false" && value != "False";
}

bool MLMTPR::UseSHAccumSkipSiteDers() const
{
	const char* env = std::getenv("SUS2_SH_ACCUM_SKIP_SITE_DERS");
	if (env == nullptr)
		return false;
	const std::string value(env);
	return value != "0" && value != "false" && value != "False";
}

bool MLMTPR::UseSHProductHVTReverse() const
{
	const char* env = std::getenv("SUS2_SH_PRODUCT_HVT_REVERSE");
	if (env == nullptr)
		return false;
	const std::string value(env);
	return value != "0" && value != "false" && value != "False";
}

namespace {

bool SHAccumProfileEnabledOnRank0()
{
	const char* env = std::getenv("SUS2_SH_ACCUM_PROFILE");
	if (env == nullptr)
		return false;
	const std::string value(env);
	if (value == "0" || value == "false" || value == "False")
		return false;
	const char* rank_envs[] = {
		"PMI_RANK",
		"OMPI_COMM_WORLD_RANK",
		"MV2_COMM_WORLD_RANK",
		"MPI_RANKID"
	};
	for (const char* rank_env : rank_envs) {
		const char* rank_value = std::getenv(rank_env);
		if (rank_value != nullptr)
			return std::atoi(rank_value) == 0;
	}
	return true;
}

int SHAccumProfileInterval()
{
	const char* env = std::getenv("SUS2_SH_ACCUM_PROFILE_INTERVAL");
	if (env == nullptr)
		return 10000;
	const int value = std::atoi(env);
	return value > 0 ? value : 10000;
}

double SHAccumProfileNow()
{
	using clock = std::chrono::steady_clock;
	return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

struct SHAccumProfileState {
	long long calls = 0;
	long long neighbors = 0;
	double total_s = 0.0;
	double moment_s = 0.0;
	double energy_backprop_s = 0.0;
	double force_product_s = 0.0;
	double force_cache_s = 0.0;
	double tangent_forward_s = 0.0;
	double mixed_seed_s = 0.0;
	double hvt_reverse_s = 0.0;
	double scalar_grad_s = 0.0;
	double coeff_grad_s = 0.0;
};

SHAccumProfileState& SHAccumProfile()
{
	static SHAccumProfileState state;
	return state;
}

void RecordSHAccumProfile(double total_s,
                          double moment_s,
                          double energy_backprop_s,
                          double force_product_s,
                          double force_cache_s,
                          double tangent_forward_s,
                          double mixed_seed_s,
                          double hvt_reverse_s,
                          double scalar_grad_s,
                          double coeff_grad_s,
                          int neighbor_count)
{
	SHAccumProfileState& state = SHAccumProfile();
	state.calls += 1;
	state.neighbors += neighbor_count;
	state.total_s += total_s;
	state.moment_s += moment_s;
	state.energy_backprop_s += energy_backprop_s;
	state.force_product_s += force_product_s;
	state.force_cache_s += force_cache_s;
	state.tangent_forward_s += tangent_forward_s;
	state.mixed_seed_s += mixed_seed_s;
	state.hvt_reverse_s += hvt_reverse_s;
	state.scalar_grad_s += scalar_grad_s;
	state.coeff_grad_s += coeff_grad_s;
	const int interval = SHAccumProfileInterval();
	if (state.calls % interval != 0)
		return;
	const double inv_calls = 1.0 / static_cast<double>(state.calls);
	std::cout << "[SUS2_SH_ACCUM_PROFILE] calls=" << state.calls
	          << " neighbors=" << state.neighbors
	          << " avg_us_total=" << 1.0e6 * state.total_s * inv_calls
	          << " avg_us_moment=" << 1.0e6 * state.moment_s * inv_calls
	          << " avg_us_energy_backprop=" << 1.0e6 * state.energy_backprop_s * inv_calls
	          << " avg_us_force_product=" << 1.0e6 * state.force_product_s * inv_calls
	          << " avg_us_force_cache=" << 1.0e6 * state.force_cache_s * inv_calls
	          << " avg_us_tangent_forward=" << 1.0e6 * state.tangent_forward_s * inv_calls
	          << " avg_us_mixed_seed=" << 1.0e6 * state.mixed_seed_s * inv_calls
	          << " avg_us_hvt_reverse=" << 1.0e6 * state.hvt_reverse_s * inv_calls
	          << " avg_us_scalar_grad=" << 1.0e6 * state.scalar_grad_s * inv_calls
	          << " avg_us_coeff_grad=" << 1.0e6 * state.coeff_grad_s * inv_calls
	          << std::endl;
}

} // namespace

void MLMTPR::TraceSHProductProgramOnce()
{
	if (sh_product_rows_trace_printed_)
		return;
	const char* env = std::getenv("SUS2_SH_PRODUCT_ROWS_TRACE");
	if (env == nullptr)
		return;
	const std::string value(env);
	if (value == "0" || value == "false" || value == "False")
		return;

	int terminal_rows = 0;
	int terminal_terms = 0;
	for (const SHProductRow& row : sh_product_rows_) {
		if (!row.terminal_scalar)
			continue;
		++terminal_rows;
		terminal_terms += row.term_count;
	}
	std::cout << "SUS2-SH product rows enabled: rows=" << sh_product_rows_.size()
	          << " terms=" << sh_product_row_terms_.size()
	          << " terminal_scalar_rows=" << terminal_rows
	          << " terminal_scalar_terms=" << terminal_terms << std::endl;
	sh_product_rows_trace_printed_ = true;
}

void MLMTPR::TraceSHSiteDerivativeCacheOnce(int neighbor_count,
                                            int sh_count,
                                            int radial_func_count)
{
	if (sh_site_der_cache_trace_printed_)
		return;
	const char* env = std::getenv("SUS2_SH_SITE_DER_CACHE_TRACE");
	if (env == nullptr)
		return;
	const std::string value(env);
	if (value == "0" || value == "false" || value == "False")
		return;

	std::cout << "SUS2-SH site derivative cache enabled: neighbors=" << neighbor_count
	          << " sh_components=" << sh_count
	          << " radial_functions=" << radial_func_count << std::endl;
	sh_site_der_cache_trace_printed_ = true;
}

void MLMTPR::ApplySHProductRowsForward()
{
	TraceSHProductProgramOnce();
	for (const SHProductRow& row : sh_product_rows_) {
		double value = 0.0;
		const int end = row.term_begin + row.term_count;
		for (int t = row.term_begin; t < end; ++t) {
			const SHProductRowTerm& term = sh_product_row_terms_[t];
			value += term.coeff * moment_vals[term.left] * moment_vals[term.right];
		}
		moment_vals[row.target] += value;
	}
}

void MLMTPR::ApplySHProductRowsDers(const Neighborhood& nbh)
{
	TraceSHProductProgramOnce();
	for (const SHProductRow& row : sh_product_rows_) {
		double value = 0.0;
		const int end = row.term_begin + row.term_count;
		for (int t = row.term_begin; t < end; ++t) {
			const SHProductRowTerm& term = sh_product_row_terms_[t];
			value += term.coeff * moment_vals[term.left] * moment_vals[term.right];
		}
		moment_vals[row.target] += value;

		if (row.terminal_scalar) {
			const int basis_index = 1 + row.scalar_index;
			basis_vals[basis_index] = moment_vals[row.target];
			for (int j = 0; j < nbh.count; ++j) {
				for (int a = 0; a < 3; ++a) {
					double der = 0.0;
					for (int t = row.term_begin; t < end; ++t) {
						const SHProductRowTerm& term = sh_product_row_terms_[t];
						der += term.coeff * (
							moment_ders(term.left, j, a) * moment_vals[term.right]
							+ moment_vals[term.left] * moment_ders(term.right, j, a));
					}
					basis_ders(basis_index, j, a) = der;
				}
			}
		} else {
			for (int j = 0; j < nbh.count; ++j) {
				for (int a = 0; a < 3; ++a) {
					double der = 0.0;
					for (int t = row.term_begin; t < end; ++t) {
						const SHProductRowTerm& term = sh_product_row_terms_[t];
						der += term.coeff * (
							moment_ders(term.left, j, a) * moment_vals[term.right]
							+ moment_vals[term.left] * moment_ders(term.right, j, a));
					}
					moment_ders(row.target, j, a) += der;
				}
			}
		}
	}
}

void MLMTPR::AccumulateSHProductRowsForward(const std::vector<double>& input_values,
                                            std::vector<double>& output_values) const
{
	for (const SHProductRow& row : sh_product_rows_) {
		double value = 0.0;
		const int end = row.term_begin + row.term_count;
		for (int t = row.term_begin; t < end; ++t) {
			const SHProductRowTerm& term = sh_product_row_terms_[t];
			value += term.coeff * (
				input_values[term.left] * moment_vals[term.right]
				+ moment_vals[term.left] * input_values[term.right]);
		}
		output_values[row.target] += value;
	}
}

void MLMTPR::BackpropSHProductRows(std::vector<double>& adjoints) const
{
	for (int r = static_cast<int>(sh_product_rows_.size()) - 1; r >= 0; --r) {
		const SHProductRow& row = sh_product_rows_[r];
		const double adj_target = adjoints[row.target];
		if (adj_target == 0.0)
			continue;
		for (int t = row.term_begin + row.term_count - 1; t >= row.term_begin; --t) {
			const SHProductRowTerm& term = sh_product_row_terms_[t];
			adjoints[term.left] += term.coeff * moment_vals[term.right] * adj_target;
			adjoints[term.right] += term.coeff * moment_vals[term.left] * adj_target;
		}
	}
}

void MLMTPR::ClearTwoLayerEdgePrimitiveCache()
{
	two_layer_edge_cache_ready_ = false;
	two_layer_edge_cache_has_derivatives_ = false;
	two_layer_edge_cache_has_param_derivatives_ = false;
	two_layer_full_edge_cache_for_next_calc_ = false;
	two_layer_reuse_full_edge_cache_once_ = false;
	two_layer_forward_final_moment_cache_ready_ = false;
	two_layer_edge_cache_atom_count_ = 0;
	two_layer_edge_cache_eval_block_count_ = 0;
	two_layer_edge_cache_sh_count_ = 0;
	two_layer_edge_cache_radial_func_count_ = 0;
	two_layer_edge_cache_gate_mu_count_ = 0;
	two_layer_edge_cache_rb_size_ = 0;
	active_two_layer_edge_cache_atom_index_ = -1;
	two_layer_gate_values_from_edge_cache_ready_ = false;
}

bool MLMTPR::HasTwoLayerEdgePrimitiveCache(int cache_atom_index,
                                           bool need_derivatives,
                                           bool need_param_derivatives) const
{
	if (cache_atom_index < 0)
		cache_atom_index = active_two_layer_edge_cache_atom_index_;
	if (!two_layer_edge_cache_ready_ || cache_atom_index < 0)
		return false;
	if (cache_atom_index >= two_layer_edge_cache_atom_count_)
		return false;
	if (need_derivatives && !two_layer_edge_cache_has_derivatives_)
		return false;
	if (need_derivatives
	    && need_param_derivatives
	    && !two_layer_edge_cache_has_param_derivatives_)
		return false;
	if (two_layer_edge_cache_eval_block_count_ !=
	    static_cast<int>(radial_eval_to_scaling_block_.size()))
		return false;
	if (two_layer_edge_cache_sh_count_ != (sh_l_max_ + 1) * (sh_l_max_ + 1))
		return false;
	if (two_layer_edge_cache_radial_func_count_ != radial_func_count)
		return false;
	if (two_layer_edge_cache_rb_size_ != p_RadialBasis->rb_size)
		return false;
	if (static_cast<size_t>(cache_atom_index + 1)
	    >= two_layer_edge_offsets_cache_.size())
		return false;
	const size_t edge_count = two_layer_edge_offsets_cache_.back();
	const size_t expected_sh_size =
		edge_count * static_cast<size_t>(two_layer_edge_cache_sh_count_);
	const size_t expected_mu_size =
		edge_count * static_cast<size_t>(radial_func_count);
	const size_t expected_gate_mu_size =
		edge_count * static_cast<size_t>(two_layer_gate_required_mu_indices_.size());
	if (two_layer_edge_sh_values_cache_.size() < expected_sh_size
	    || two_layer_edge_mu_vals_cache_.size() < expected_mu_size)
		return false;
	if (need_derivatives
	    && (two_layer_edge_sh_ders_cache_.size()
	            < expected_sh_size * static_cast<size_t>(3)
	        || two_layer_edge_mu_ders_cache_.size() < expected_mu_size))
		return false;
	if (need_derivatives && need_param_derivatives) {
		const size_t expected_radial_val_size =
			edge_count
			* static_cast<size_t>(two_layer_edge_cache_eval_block_count_)
			* static_cast<size_t>(two_layer_edge_cache_rb_size_);
		const size_t expected_radial_der_size =
			expected_radial_val_size * static_cast<size_t>(5);
		if (two_layer_edge_radial_vals_cache_.size() < expected_radial_val_size
		    || two_layer_edge_radial_ders_cache_.size() < expected_radial_der_size
		    || two_layer_edge_mu_ders_s_cache_.size() < expected_mu_size
		    || two_layer_edge_mu_ders_ss_cache_.size() < expected_mu_size
		    || two_layer_edge_mu_coord_ders_s_cache_.size() < expected_mu_size
		    || two_layer_edge_mu_coord_ders_ss_cache_.size() < expected_mu_size)
			return false;
	}
	if (TwoLayerGateUsesSharedRadial()) {
		if (two_layer_edge_cache_gate_mu_count_
		    != static_cast<int>(two_layer_gate_required_mu_indices_.size()))
			return false;
		if (two_layer_edge_gate_mu_vals_cache_.size() < expected_gate_mu_size)
			return false;
		if (need_derivatives
		    && two_layer_edge_gate_mu_ders_cache_.size() < expected_gate_mu_size)
			return false;
		if (need_derivatives
		    && need_param_derivatives
		    && (two_layer_edge_gate_mu_ders_s_cache_.size() < expected_gate_mu_size
		        || two_layer_edge_gate_mu_ders_ss_cache_.size() < expected_gate_mu_size
		        || two_layer_edge_gate_mu_coord_ders_s_cache_.size() < expected_gate_mu_size
		        || two_layer_edge_gate_mu_coord_ders_ss_cache_.size() < expected_gate_mu_size))
			return false;
	}
	return true;
}

size_t MLMTPR::TwoLayerEdgePrimitiveOffset(int cache_atom_index,
                                           int neighbor_index) const
{
	if (cache_atom_index < 0)
		cache_atom_index = active_two_layer_edge_cache_atom_index_;
	return two_layer_edge_offsets_cache_[cache_atom_index]
		+ static_cast<size_t>(neighbor_index);
}

int MLMTPR::TwoLayerGateMuDenseIndex(int mu) const
{
	if (mu < 0 || mu >= radial_func_count)
		ERROR("SUS2-SH two-layer gate mu index is out of range");
	if (static_cast<int>(two_layer_gate_mu_dense_index_.size()) != radial_func_count)
		ERROR("SUS2-SH two-layer gate mu dense map is not initialized");
	const int dense = two_layer_gate_mu_dense_index_[mu];
	if (dense < 0 || dense >= static_cast<int>(two_layer_gate_required_mu_indices_.size()))
		ERROR("SUS2-SH two-layer gate mu dense index is out of range");
	return dense;
}

void MLMTPR::BuildTwoLayerEdgePrimitiveCache(const Neighborhoods& neighborhoods,
                                             bool need_derivatives,
                                             bool need_param_derivatives)
{
	ClearTwoLayerEdgePrimitiveCache();
	if (DisableTwoLayerEdgePrimitiveCache())
		return;
	if (!is_sh_potential_ || !two_layer_gate_enabled_)
		return;
	const bool use_lmp_table = SHUsesPrecomputedLmpTable(p_RadialBasis);
	if (use_lmp_table && need_param_derivatives)
		ERROR("SUS2-SH _lmp radial tables do not provide nonlinear parameter derivatives.");
	if (sh_l_max_ < 0 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,6].");
	if (TwoLayerGateWeightCount() > 0
	    && (two_layer_gate_required_moments_.empty()
	        || two_layer_gate_required_moment_indices_.empty()))
		BuildTwoLayerGateProductProgram();
	const bool use_gate_radial = TwoLayerGateUsesSharedRadial();
	const int gate_count = TwoLayerGateWeightCount();
	const int cached_gate_moment_count =
		static_cast<int>(two_layer_gate_required_moment_indices_.size());
	const bool prepare_gate_values_from_cache =
		gate_count > 0 && cached_gate_moment_count > 0;

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int eval_block_count =
		static_cast<int>(radial_eval_to_scaling_block_.size());
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const int radial_val_stride = eval_block_count * R;
	const int radial_der_stride = radial_val_stride * 5;
	const int mu_stride = radial_func_count;
	const int gate_mu_stride = use_gate_radial
		? static_cast<int>(two_layer_gate_required_mu_indices_.size())
		: 0;
	const int atom_count = neighborhoods.size();
	const bool need_raw_radial_cache =
		need_derivatives && need_param_derivatives;
	size_t edge_count = 0;

	two_layer_edge_offsets_cache_.assign(static_cast<size_t>(atom_count) + 1, 0);
	for (int ind = 0; ind < atom_count; ++ind) {
		two_layer_edge_offsets_cache_[ind] = edge_count;
		edge_count += static_cast<size_t>(neighborhoods[ind].count);
	}
	two_layer_edge_offsets_cache_[atom_count] = edge_count;

	two_layer_edge_sh_values_cache_.resize(edge_count * sh_count);
	two_layer_edge_mu_vals_cache_.resize(edge_count * mu_stride);
	if (need_raw_radial_cache) {
		two_layer_edge_radial_vals_cache_.resize(edge_count * radial_val_stride);
	} else {
		two_layer_edge_radial_vals_cache_.clear();
	}
	if (need_derivatives) {
		two_layer_edge_sh_ders_cache_.resize(edge_count * 3 * sh_count);
		two_layer_edge_mu_ders_cache_.resize(edge_count * mu_stride);
		if (need_param_derivatives) {
			two_layer_edge_radial_ders_cache_.resize(edge_count * radial_der_stride);
			two_layer_edge_mu_ders_s_cache_.resize(edge_count * mu_stride);
			two_layer_edge_mu_ders_ss_cache_.resize(edge_count * mu_stride);
			two_layer_edge_mu_coord_ders_s_cache_.resize(edge_count * mu_stride);
			two_layer_edge_mu_coord_ders_ss_cache_.resize(edge_count * mu_stride);
		} else {
			two_layer_edge_radial_ders_cache_.clear();
			two_layer_edge_mu_ders_s_cache_.clear();
			two_layer_edge_mu_ders_ss_cache_.clear();
			two_layer_edge_mu_coord_ders_s_cache_.clear();
			two_layer_edge_mu_coord_ders_ss_cache_.clear();
		}
	} else {
		two_layer_edge_sh_ders_cache_.clear();
		two_layer_edge_radial_ders_cache_.clear();
		two_layer_edge_mu_ders_cache_.clear();
		two_layer_edge_mu_ders_s_cache_.clear();
		two_layer_edge_mu_ders_ss_cache_.clear();
		two_layer_edge_mu_coord_ders_s_cache_.clear();
		two_layer_edge_mu_coord_ders_ss_cache_.clear();
	}
	if (use_gate_radial) {
		two_layer_edge_gate_mu_vals_cache_.resize(edge_count * gate_mu_stride);
		if (need_derivatives) {
			two_layer_edge_gate_mu_ders_cache_.resize(edge_count * gate_mu_stride);
			if (need_param_derivatives) {
				two_layer_edge_gate_mu_ders_s_cache_.resize(edge_count * gate_mu_stride);
				two_layer_edge_gate_mu_ders_ss_cache_.resize(edge_count * gate_mu_stride);
				two_layer_edge_gate_mu_coord_ders_s_cache_.resize(edge_count * gate_mu_stride);
				two_layer_edge_gate_mu_coord_ders_ss_cache_.resize(edge_count * gate_mu_stride);
			} else {
				two_layer_edge_gate_mu_ders_s_cache_.clear();
				two_layer_edge_gate_mu_ders_ss_cache_.clear();
				two_layer_edge_gate_mu_coord_ders_s_cache_.clear();
				two_layer_edge_gate_mu_coord_ders_ss_cache_.clear();
			}
		}
	}
	if (prepare_gate_values_from_cache) {
		two_layer_gate_values_.resize(atom_count);
		two_layer_gate_scalar_values_cache_.resize(
			static_cast<size_t>(atom_count) * gate_count);
		two_layer_gate_moment_values_cache_.resize(
			static_cast<size_t>(atom_count) * cached_gate_moment_count);
	}

	const int radial_coeff_base = C + 2 * C * C * K_;
	const int shared_type_offset = radial_coeff_base + R;
	std::vector<double> radial_vals_scratch;
	std::vector<double> radial_ders_scratch;
	if (!need_raw_radial_cache)
		radial_vals_scratch.resize(radial_val_stride);
	if (need_derivatives && !need_param_derivatives)
		radial_ders_scratch.resize(radial_der_stride);
	std::vector<double> gate_lmp_vals_scratch;
	std::vector<double> gate_lmp_ders_scratch;
	if (use_lmp_table && use_gate_radial) {
		gate_lmp_vals_scratch.resize(radial_func_count);
		if (need_derivatives)
			gate_lmp_ders_scratch.resize(radial_func_count);
	}

	for (int ind = 0; ind < atom_count; ++ind) {
		const Neighborhood& nbh = neighborhoods[ind];
		const int type_central = nbh.my_type;
		if (type_central >= species_count)
			throw MlipException("Too few species count in the MTP potential!");
		if (prepare_gate_values_from_cache)
			for (int moment_index : two_layer_gate_required_moment_indices_)
				moment_vals[moment_index] = 0.0;
		const double center_type_coeff = prepare_gate_values_from_cache
			? regression_coeffs[shared_type_offset + type_central]
			: 0.0;
		for (int j = 0; j < nbh.count; ++j) {
			const size_t edge = two_layer_edge_offsets_cache_[ind]
				+ static_cast<size_t>(j);
			const Vector3& rvec = nbh.vecs[j];
			const double r = nbh.dists[j];
			const int type_outer = nbh.types[j];
			double* sh_values =
				two_layer_edge_sh_values_cache_.data() + edge * sh_count;
			double* sh_ders = need_derivatives
				? two_layer_edge_sh_ders_cache_.data() + edge * 3 * sh_count
				: nullptr;
			double* rb_vals = need_raw_radial_cache
				? two_layer_edge_radial_vals_cache_.data()
					+ edge * radial_val_stride
				: radial_vals_scratch.data();
			double* rb_ders = need_derivatives
				? (need_param_derivatives
					? two_layer_edge_radial_ders_cache_.data()
						+ edge * radial_der_stride
					: radial_ders_scratch.data())
				: nullptr;
			double* mu_vals =
				two_layer_edge_mu_vals_cache_.data() + edge * mu_stride;
			double* mu_ders = need_derivatives
				? two_layer_edge_mu_ders_cache_.data() + edge * mu_stride
				: nullptr;
			double* mu_ders_s = need_derivatives && need_param_derivatives
				? two_layer_edge_mu_ders_s_cache_.data() + edge * mu_stride
				: nullptr;
			double* mu_ders_ss = need_derivatives && need_param_derivatives
				? two_layer_edge_mu_ders_ss_cache_.data() + edge * mu_stride
				: nullptr;
			double* mu_coord_ders_s = need_derivatives && need_param_derivatives
				? two_layer_edge_mu_coord_ders_s_cache_.data() + edge * mu_stride
				: nullptr;
				double* mu_coord_ders_ss = need_derivatives && need_param_derivatives
					? two_layer_edge_mu_coord_ders_ss_cache_.data() + edge * mu_stride
					: nullptr;
			double* gate_mu_vals = use_gate_radial
				? two_layer_edge_gate_mu_vals_cache_.data() + edge * gate_mu_stride
				: nullptr;
			double* gate_mu_ders = use_gate_radial && need_derivatives
				? two_layer_edge_gate_mu_ders_cache_.data() + edge * gate_mu_stride
				: nullptr;
			double* gate_mu_ders_s =
				use_gate_radial && need_derivatives && need_param_derivatives
				? two_layer_edge_gate_mu_ders_s_cache_.data() + edge * gate_mu_stride
				: nullptr;
			double* gate_mu_ders_ss =
				use_gate_radial && need_derivatives && need_param_derivatives
				? two_layer_edge_gate_mu_ders_ss_cache_.data() + edge * gate_mu_stride
				: nullptr;
			double* gate_mu_coord_ders_s =
				use_gate_radial && need_derivatives && need_param_derivatives
				? two_layer_edge_gate_mu_coord_ders_s_cache_.data() + edge * gate_mu_stride
				: nullptr;
			double* gate_mu_coord_ders_ss =
				use_gate_radial && need_derivatives && need_param_derivatives
				? two_layer_edge_gate_mu_coord_ders_ss_cache_.data() + edge * gate_mu_stride
				: nullptr;

				if (need_derivatives)
					EvalRealSH(rvec, r, sh_l_max_, sh_values, sh_ders);
			else
				EvalRealSHValuesOnly(rvec, r, sh_l_max_, sh_values);

			if (use_lmp_table) {
				InterpolateSHLmpMuTable(radial_list,
				                         radial_der_list,
				                         inv_dr,
				                         C,
				                         type_central,
				                         type_outer,
				                         r,
				                         radial_func_count,
				                         mu_vals,
				                         need_derivatives ? mu_ders : nullptr);
				if (use_gate_radial) {
					InterpolateSHLmpMuTable(two_layer_gate_radial_list,
					                         two_layer_gate_radial_der_list,
					                         inv_dr,
					                         C,
					                         type_central,
					                         type_outer,
					                         r,
					                         radial_func_count,
					                         gate_lmp_vals_scratch.data(),
					                         need_derivatives ? gate_lmp_ders_scratch.data() : nullptr);
					for (int dense_mu = 0;
					     dense_mu < static_cast<int>(two_layer_gate_required_mu_indices_.size());
					     ++dense_mu) {
						const int mu = two_layer_gate_required_mu_indices_[dense_mu];
						gate_mu_vals[dense_mu] = gate_lmp_vals_scratch[mu];
						if (need_derivatives)
							gate_mu_ders[dense_mu] = gate_lmp_ders_scratch[mu];
					}
				}
			} else {
				for (int eval_block = 0; eval_block < eval_block_count; ++eval_block) {
					const int scaling_block = radial_eval_to_scaling_block_[eval_block];
					const int basis_k = radial_eval_to_basis_k_[eval_block];
					const double sigma =
						regression_coeffs[
							C + 2 * scaling_block * C * C + C * type_central
							+ type_outer];
					const double shift =
						regression_coeffs[
							C + 2 * scaling_block * C * C + C * C
							+ C * type_central + type_outer];
					if (need_derivatives)
						p_RadialBasis->RB_Calc(r, sigma, shift, basis_k);
					else
						p_RadialBasis->RB_CalcValsOnly(r, sigma, shift, basis_k);
					for (int xi = 0; xi < R; ++xi)
						rb_vals[eval_block * R + xi] =
							p_RadialBasis->rb_vals[xi] * scaling;
					if (need_derivatives)
						for (int xi = 0; xi < R * 5; ++xi)
							rb_ders[eval_block * R * 5 + xi] =
								p_RadialBasis->rb_ders[xi] * scaling;
				}

				for (int mu = 0; mu < radial_func_count; ++mu) {
					const int radial_base = mu_to_radial_eval_block_[mu] * R;
					const int deriv_base = 5 * radial_base;
					const int radial_offset = radial_coeff_base + mu * (R + C);
					double dot_val = 0.0;
					double dot_der = 0.0;
					double dot_s = 0.0;
					double dot_coord_s = 0.0;
					double dot_ss = 0.0;
					double dot_coord_ss = 0.0;
					for (int xi = 0; xi < R; ++xi) {
						const double coeff = regression_coeffs[radial_offset + xi];
						dot_val += coeff * rb_vals[radial_base + xi];
						if (need_derivatives) {
							dot_der += coeff * rb_ders[deriv_base + xi];
							if (need_param_derivatives) {
								dot_s += coeff * rb_ders[deriv_base + xi + R];
								dot_coord_s += coeff * rb_ders[deriv_base + xi + 2 * R];
								dot_ss += coeff * rb_ders[deriv_base + xi + 3 * R];
								dot_coord_ss +=
									coeff * rb_ders[deriv_base + xi + 4 * R];
							}
						}
					}
					mu_vals[mu] = dot_val;
					if (need_derivatives) {
						mu_ders[mu] = dot_der;
						if (need_param_derivatives) {
							mu_ders_s[mu] = dot_s;
							mu_coord_ders_s[mu] = dot_coord_s;
							mu_ders_ss[mu] = dot_ss;
							mu_coord_ders_ss[mu] = dot_coord_ss;
						}
					}
				}
				if (use_gate_radial) {
					for (int dense_mu = 0;
					     dense_mu < static_cast<int>(two_layer_gate_required_mu_indices_.size());
					     ++dense_mu) {
						const int mu = two_layer_gate_required_mu_indices_[dense_mu];
						const int radial_base = mu_to_radial_eval_block_[mu] * R;
						const int deriv_base = 5 * radial_base;
						const int radial_offset = TwoLayerGateRadialCoeffIndex(mu, 0);
						double dot_val = 0.0;
						double dot_der = 0.0;
						double dot_s = 0.0;
						double dot_coord_s = 0.0;
						double dot_ss = 0.0;
						double dot_coord_ss = 0.0;
						for (int xi = 0; xi < R; ++xi) {
							const double coeff = regression_coeffs[radial_offset + xi];
							dot_val += coeff * rb_vals[radial_base + xi];
							if (need_derivatives) {
								dot_der += coeff * rb_ders[deriv_base + xi];
								if (need_param_derivatives) {
									dot_s += coeff * rb_ders[deriv_base + xi + R];
									dot_coord_s +=
										coeff * rb_ders[deriv_base + xi + 2 * R];
									dot_ss += coeff * rb_ders[deriv_base + xi + 3 * R];
									dot_coord_ss +=
										coeff * rb_ders[deriv_base + xi + 4 * R];
								}
							}
						}
						gate_mu_vals[dense_mu] = dot_val;
						if (need_derivatives) {
							gate_mu_ders[dense_mu] = dot_der;
							if (need_param_derivatives) {
								gate_mu_ders_s[dense_mu] = dot_s;
								gate_mu_coord_ders_s[dense_mu] = dot_coord_s;
								gate_mu_ders_ss[dense_mu] = dot_ss;
								gate_mu_coord_ders_ss[dense_mu] = dot_coord_ss;
							}
						}
					}
				}
			}
			if (prepare_gate_values_from_cache) {
			const double outer_type_coeff =
				regression_coeffs[shared_type_offset + type_outer];
			const double type_scale = center_type_coeff * outer_type_coeff;
			const double* gate_mu_values = use_gate_radial ? gate_mu_vals : mu_vals;
			for (int bi = 0;
			     bi < static_cast<int>(two_layer_gate_required_basic_indices_.size());
			     ++bi) {
				const int basic_index = two_layer_gate_required_basic_indices_[bi];
				const int mu = basic_mu_cache_[basic_index];
				const int mu_index =
					use_gate_radial
						? two_layer_gate_required_basic_dense_mu_indices_[bi]
						: mu;
				const int sh_index = basic_sh_index_cache_[basic_index];
				moment_vals[basic_index] +=
					type_scale * gate_mu_values[mu_index] * sh_values[sh_index];
				}
			}
		}
		if (prepare_gate_values_from_cache) {
			for (int product_index : two_layer_gate_required_product_indices_) {
				const SHProduct& product = sh_products_[product_index];
				moment_vals[product.target] +=
					product.coeff * moment_vals[product.left]
					* moment_vals[product.right];
			}
			double f = 0.0;
			double* cached_scalars = two_layer_gate_scalar_values_cache_.data()
				+ static_cast<size_t>(ind) * gate_count;
			double* cached_moments = two_layer_gate_moment_values_cache_.data()
				+ static_cast<size_t>(ind) * cached_gate_moment_count;
			for (int i = 0; i < cached_gate_moment_count; ++i)
				cached_moments[i] =
					moment_vals[two_layer_gate_required_moment_indices_[i]];
			for (int q = 0; q < gate_count; ++q) {
				const int scalar_index = two_layer_gate_scalar_indices_[q];
				if (scalar_index < 0 || scalar_index >= alpha_scalar_moments)
					ERROR("SUS2-SH two-layer gate scalar index is out of range");
				const int moment = alpha_moment_mapping[scalar_index];
				cached_scalars[q] = moment_vals[moment];
				f += TwoLayerGateWeight(q) * cached_scalars[q];
			}
			two_layer_gate_values_[ind] =
				TwoLayerGateUsesDirectScale()
					? (two_layer_gate_bias_ + f)
					: (1.0 + f);
		}
	}
	if (prepare_gate_values_from_cache)
		two_layer_gate_values_from_edge_cache_ready_ = true;

	two_layer_edge_cache_ready_ = true;
	two_layer_edge_cache_has_derivatives_ = need_derivatives;
	two_layer_edge_cache_has_param_derivatives_ =
		need_derivatives && need_param_derivatives;
	two_layer_edge_cache_atom_count_ = atom_count;
	two_layer_edge_cache_eval_block_count_ = eval_block_count;
	two_layer_edge_cache_sh_count_ = sh_count;
	two_layer_edge_cache_radial_func_count_ = radial_func_count;
	two_layer_edge_cache_gate_mu_count_ = gate_mu_stride;
	two_layer_edge_cache_rb_size_ = R;
}

void MLMTPR::CalcSHMomentValuesOnly(const Neighborhood& nbh)
{
	const bool use_lmp_table = SHUsesPrecomputedLmpTable(p_RadialBasis);
	if (sh_l_max_ < 0 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,6].");

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	if (type_central >= species_count)
		throw MlipException("Too few species count in the MTP potential!");

	std::fill(moment_vals, moment_vals + alpha_moments_count, 0.0);

	double sh_values[kMaxSHComponents];
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& radial_values = grad_mu_contract_vals_;
	const bool use_edge_cache = HasTwoLayerEdgePrimitiveCache(-1, false);
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const int mu_stride = radial_func_count;

	const int radial_coeff_base = C + 2 * C * C * K_;
	const int shared_type_offset = radial_coeff_base + R;

	for (int j = 0; j < nbh.count; ++j) {
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const int type_outer = nbh.types[j];
		const double* sh_values_use = sh_values;
		const double* radial_values_use = radial_values.data();
		if (use_edge_cache) {
			const size_t edge = TwoLayerEdgePrimitiveOffset(-1, j);
			sh_values_use =
				two_layer_edge_sh_values_cache_.data() + edge * sh_count;
			radial_values_use =
				two_layer_edge_mu_vals_cache_.data() + edge * mu_stride;
	} else {
		EvalRealSHValuesOnly(rvec, r, sh_l_max_, sh_values);

			if (use_lmp_table) {
				InterpolateSHLmpMuTable(radial_list,
				                         radial_der_list,
				                         inv_dr,
				                         C,
				                         type_central,
				                         type_outer,
				                         r,
				                         radial_func_count,
				                         radial_values.data(),
				                         nullptr);
			} else {
				for (int eval_block = 0; eval_block < static_cast<int>(radial_eval_to_scaling_block_.size()); ++eval_block) {
					const int scaling_block = radial_eval_to_scaling_block_[eval_block];
					const int basis_k = radial_eval_to_basis_k_[eval_block];
					p_RadialBasis->RB_CalcValsOnly(
						r,
						regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer],
						regression_coeffs[C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer],
						basis_k);
					for (int xi = 0; xi < R; ++xi)
						rb_vals[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
				}
				for (int mu = 0; mu < radial_func_count; ++mu) {
					const int radial_base = mu_to_radial_eval_block_[mu] * R;
					const int radial_offset = radial_coeff_base + mu * (R + C);
					double radial_value = 0.0;
					for (int xi = 0; xi < R; ++xi)
						radial_value += regression_coeffs[radial_offset + xi] * rb_vals[radial_base + xi];
					radial_values[mu] = radial_value;
				}
			}
		}

			const double center_type_coeff = regression_coeffs[shared_type_offset + type_central];
			const double outer_type_coeff = regression_coeffs[shared_type_offset + type_outer];
			const double pair_type_scale = center_type_coeff * outer_type_coeff;
			const double gate_residual = TwoLayerGateNeighborResidual(nbh, j);
			PrepareTwoLayerGateNeighborMuBuffers(
				type_outer, center_type_coeff, outer_type_coeff, gate_residual);
			const double* gate_type_scale_by_mu =
				two_layer_gate_type_scale_mu_buffer_.data();
			double* gate_scaled_radial_by_mu =
				two_layer_gate_scaled_mu_vals_buffer_.data();
			for (int mu = 0; mu < radial_func_count; ++mu)
				gate_scaled_radial_by_mu[mu] =
					gate_type_scale_by_mu[mu] * radial_values_use[mu];

				for (int i = 0; i < alpha_index_basic_count; ++i) {
					const int mu = basic_mu_cache_[i];
					const int sh_index = basic_sh_index_cache_[i];
					moment_vals[i] +=
						gate_scaled_radial_by_mu[mu] * sh_values_use[sh_index];
				}
	}

	if (UseSHProductRows()) {
		ApplySHProductRowsForward();
	} else {
		for (size_t p = 0; p < sh_products_.size(); ++p) {
			const SHProduct& product = sh_products_[p];
			moment_vals[product.target] +=
				product.coeff * moment_vals[product.left] * moment_vals[product.right];
		}
	}

	basis_vals[0] = 1.0;
	for (int i = 0; i < alpha_scalar_moments; ++i)
		basis_vals[1 + i] = moment_vals[alpha_moment_mapping[i]];
}

void MLMTPR::CalcSHMomentValuesWithSiteDerivativeCache(const Neighborhood& nbh)
{
	const bool use_lmp_table = SHUsesPrecomputedLmpTable(p_RadialBasis);
	if (sh_l_max_ < 0 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,6].");

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	if (type_central >= species_count)
		throw MlipException("Too few species count in the MTP potential!");

	const int eval_block_count = static_cast<int>(radial_eval_to_scaling_block_.size());
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const int mu_stride = radial_func_count;
	const size_t neighbor_count = static_cast<size_t>(nbh.count);

	TraceSHSiteDerivativeCacheOnce(nbh.count, sh_count, radial_func_count);

	grad_neighbor_sh_values_cache_.resize(neighbor_count * sh_count);
	grad_neighbor_sh_ders_cache_.resize(neighbor_count * 3 * sh_count);
	grad_neighbor_mu_contract_vals_cache_.resize(neighbor_count * mu_stride);
	grad_neighbor_mu_contract_ders_cache_.resize(neighbor_count * mu_stride);

	std::fill(moment_vals, moment_vals + alpha_moments_count, 0.0);

	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& rb_ders = basis_radial_ders_buffer_;
	const bool use_edge_cache = HasTwoLayerEdgePrimitiveCache(-1, true, false);

	const int radial_coeff_base = C + 2 * C * C * K_;
	const int shared_type_offset = radial_coeff_base + R;

	for (int j = 0; j < nbh.count; ++j) {
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const int type_outer = nbh.types[j];
		double* cached_sh_values = grad_neighbor_sh_values_cache_.data() + static_cast<size_t>(j) * sh_count;
		double* cached_sh_ders = grad_neighbor_sh_ders_cache_.data() + static_cast<size_t>(j) * 3 * sh_count;
		double* cached_radial_values = grad_neighbor_mu_contract_vals_cache_.data() + static_cast<size_t>(j) * mu_stride;
		double* cached_radial_derivatives = grad_neighbor_mu_contract_ders_cache_.data() + static_cast<size_t>(j) * mu_stride;
		if (use_edge_cache) {
			const size_t edge = TwoLayerEdgePrimitiveOffset(-1, j);
			std::copy(two_layer_edge_sh_values_cache_.data() + edge * sh_count,
			          two_layer_edge_sh_values_cache_.data() + (edge + 1) * sh_count,
			          cached_sh_values);
			std::copy(two_layer_edge_sh_ders_cache_.data() + edge * 3 * sh_count,
			          two_layer_edge_sh_ders_cache_.data() + (edge + 1) * 3 * sh_count,
			          cached_sh_ders);
			std::copy(two_layer_edge_mu_vals_cache_.data() + edge * mu_stride,
			          two_layer_edge_mu_vals_cache_.data() + (edge + 1) * mu_stride,
			          cached_radial_values);
			std::copy(two_layer_edge_mu_ders_cache_.data() + edge * mu_stride,
			          two_layer_edge_mu_ders_cache_.data() + (edge + 1) * mu_stride,
			          cached_radial_derivatives);
	} else {
		EvalRealSH(rvec, r, sh_l_max_, cached_sh_values, cached_sh_ders);

			if (use_lmp_table) {
				InterpolateSHLmpMuTable(radial_list,
				                         radial_der_list,
				                         inv_dr,
				                         C,
				                         type_central,
				                         type_outer,
				                         r,
				                         radial_func_count,
				                         cached_radial_values,
				                         cached_radial_derivatives);
			} else {
				for (int eval_block = 0; eval_block < eval_block_count; ++eval_block) {
					const int scaling_block = radial_eval_to_scaling_block_[eval_block];
					const int basis_k = radial_eval_to_basis_k_[eval_block];
					p_RadialBasis->RB_Calc(
						r,
						regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer],
						regression_coeffs[C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer],
						basis_k);
					for (int xi = 0; xi < R; ++xi) {
						rb_vals[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
						rb_ders[eval_block * R + xi] = p_RadialBasis->rb_ders[xi] * scaling;
					}
				}
				for (int mu = 0; mu < radial_func_count; ++mu) {
					const int radial_base = mu_to_radial_eval_block_[mu] * R;
					const int radial_offset = radial_coeff_base + mu * (R + C);
					double radial_value = 0.0;
					double radial_der = 0.0;
					for (int xi = 0; xi < R; ++xi) {
						const double coeff = regression_coeffs[radial_offset + xi];
						radial_value += coeff * rb_vals[radial_base + xi];
						radial_der += coeff * rb_ders[radial_base + xi];
					}
					cached_radial_values[mu] = radial_value;
					cached_radial_derivatives[mu] = radial_der;
				}
			}
		}

			const double center_type_coeff = regression_coeffs[shared_type_offset + type_central];
			const double outer_type_coeff = regression_coeffs[shared_type_offset + type_outer];
			const double pair_type_scale = center_type_coeff * outer_type_coeff;
			const double gate_residual = TwoLayerGateNeighborResidual(nbh, j);
			PrepareTwoLayerGateNeighborMuBuffers(
				type_outer, center_type_coeff, outer_type_coeff, gate_residual);
			const double* gate_type_scale_by_mu =
				two_layer_gate_type_scale_mu_buffer_.data();
			double* gate_scaled_radial_by_mu =
				two_layer_gate_scaled_mu_vals_buffer_.data();
			for (int mu = 0; mu < radial_func_count; ++mu)
				gate_scaled_radial_by_mu[mu] =
					gate_type_scale_by_mu[mu] * cached_radial_values[mu];

			for (int i = 0; i < alpha_index_basic_count; ++i) {
				const int mu = basic_mu_cache_[i];
				const int sh_index = basic_sh_index_cache_[i];
				moment_vals[i] += gate_scaled_radial_by_mu[mu] * cached_sh_values[sh_index];
			}
	}

	if (UseSHProductRows()) {
		ApplySHProductRowsForward();
	} else {
		for (size_t p = 0; p < sh_products_.size(); ++p) {
			const SHProduct& product = sh_products_[p];
			moment_vals[product.target] +=
				product.coeff * moment_vals[product.left] * moment_vals[product.right];
		}
	}

	basis_vals[0] = 1.0;
	for (int i = 0; i < alpha_scalar_moments; ++i)
		basis_vals[1 + i] = moment_vals[alpha_moment_mapping[i]];
}

void MLMTPR::CalcSHMomentValuesWithGradientCache(const Neighborhood& nbh)
{
	if (SHUsesPrecomputedLmpTable(p_RadialBasis))
		ERROR("SUS2-SH does not use precomputed LAMMPS radial tables in training.");
	if (sh_l_max_ < 0 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,6].");

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	if (type_central >= species_count)
		throw MlipException("Too few species count in the MTP potential!");

	const int eval_block_count = static_cast<int>(radial_eval_to_scaling_block_.size());
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const int radial_val_stride = eval_block_count * R;
	const int radial_der_stride = radial_val_stride * 5;
	const int mu_stride = radial_func_count;
	const size_t neighbor_count = static_cast<size_t>(nbh.count);
	const bool use_edge_cache = HasTwoLayerEdgePrimitiveCache(-1, true, true);
	if (!use_edge_cache) {
		grad_neighbor_sh_values_cache_.resize(neighbor_count * sh_count);
		grad_neighbor_sh_ders_cache_.resize(neighbor_count * 3 * sh_count);
		grad_neighbor_radial_vals_cache_.resize(neighbor_count * radial_val_stride);
		grad_neighbor_radial_ders_cache_.resize(neighbor_count * radial_der_stride);
		grad_neighbor_mu_contract_vals_cache_.resize(neighbor_count * mu_stride);
		grad_neighbor_mu_contract_ders_cache_.resize(neighbor_count * mu_stride);
		grad_neighbor_mu_contract_ders_s_cache_.resize(neighbor_count * mu_stride);
		grad_neighbor_mu_contract_ders_ss_cache_.resize(neighbor_count * mu_stride);
		grad_neighbor_mu_contract_coord_ders_s_cache_.resize(neighbor_count * mu_stride);
		grad_neighbor_mu_contract_coord_ders_ss_cache_.resize(neighbor_count * mu_stride);
	}

	std::fill(moment_vals, moment_vals + alpha_moments_count, 0.0);

	const int radial_coeff_base = C + 2 * C * C * K_;
	const int shared_type_offset = radial_coeff_base + R;
	const double center_type_coeff = regression_coeffs[shared_type_offset + type_central];

	for (int j = 0; j < nbh.count; ++j) {
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const int type_outer = nbh.types[j];
		const double* cached_sh_values = nullptr;
		const double* cached_radial_values = nullptr;
		if (use_edge_cache) {
			const size_t edge = TwoLayerEdgePrimitiveOffset(-1, j);
			cached_sh_values =
				two_layer_edge_sh_values_cache_.data() + edge * sh_count;
			cached_radial_values =
				two_layer_edge_mu_vals_cache_.data() + edge * mu_stride;
		} else {
			double* cached_sh_values_write =
				grad_neighbor_sh_values_cache_.data() + static_cast<size_t>(j) * sh_count;
			double* cached_sh_ders =
				grad_neighbor_sh_ders_cache_.data() + static_cast<size_t>(j) * 3 * sh_count;
			double* cached_rb_vals =
				grad_neighbor_radial_vals_cache_.data()
				+ static_cast<size_t>(j) * radial_val_stride;
			double* cached_rb_ders =
				grad_neighbor_radial_ders_cache_.data()
				+ static_cast<size_t>(j) * radial_der_stride;
			double* cached_radial_values_write =
				grad_neighbor_mu_contract_vals_cache_.data()
				+ static_cast<size_t>(j) * mu_stride;
			double* cached_radial_derivatives =
				grad_neighbor_mu_contract_ders_cache_.data()
				+ static_cast<size_t>(j) * mu_stride;
			double* cached_radial_s_derivatives =
				grad_neighbor_mu_contract_ders_s_cache_.data()
				+ static_cast<size_t>(j) * mu_stride;
			double* cached_radial_ss_derivatives =
				grad_neighbor_mu_contract_ders_ss_cache_.data()
				+ static_cast<size_t>(j) * mu_stride;
			double* cached_radial_coord_s_derivatives =
				grad_neighbor_mu_contract_coord_ders_s_cache_.data()
				+ static_cast<size_t>(j) * mu_stride;
			double* cached_radial_coord_ss_derivatives =
				grad_neighbor_mu_contract_coord_ders_ss_cache_.data()
				+ static_cast<size_t>(j) * mu_stride;
			cached_sh_values = cached_sh_values_write;
			cached_radial_values = cached_radial_values_write;
			EvalRealSH(rvec, r, sh_l_max_, cached_sh_values_write, cached_sh_ders);

			for (int eval_block = 0; eval_block < eval_block_count; ++eval_block) {
				const int scaling_block = radial_eval_to_scaling_block_[eval_block];
				const int basis_k = radial_eval_to_basis_k_[eval_block];
				p_RadialBasis->RB_Calc(
					r,
					regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer],
					regression_coeffs[C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer],
					basis_k);
				for (int xi = 0; xi < R; ++xi)
					cached_rb_vals[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
				for (int xi = 0; xi < R * 5; ++xi)
					cached_rb_ders[eval_block * R * 5 + xi] = p_RadialBasis->rb_ders[xi] * scaling;
			}

			for (int mu = 0; mu < radial_func_count; ++mu) {
				const int radial_base = mu_to_radial_eval_block_[mu] * R;
				const int deriv_base = 5 * radial_base;
				const int radial_offset = radial_coeff_base + mu * (R + C);
				double dot_val = 0.0;
				double dot_der = 0.0;
				double dot_s = 0.0;
				double dot_coord_s = 0.0;
				double dot_ss = 0.0;
				double dot_coord_ss = 0.0;
				for (int xi = 0; xi < R; ++xi) {
					const double coeff = regression_coeffs[radial_offset + xi];
					dot_val += coeff * cached_rb_vals[radial_base + xi];
					dot_der += coeff * cached_rb_ders[deriv_base + xi];
					dot_s += coeff * cached_rb_ders[deriv_base + xi + R];
					dot_coord_s += coeff * cached_rb_ders[deriv_base + xi + 2 * R];
					dot_ss += coeff * cached_rb_ders[deriv_base + xi + 3 * R];
					dot_coord_ss += coeff * cached_rb_ders[deriv_base + xi + 4 * R];
				}
				cached_radial_values_write[mu] = dot_val;
				cached_radial_derivatives[mu] = dot_der;
				cached_radial_s_derivatives[mu] = dot_s;
				cached_radial_coord_s_derivatives[mu] = dot_coord_s;
				cached_radial_ss_derivatives[mu] = dot_ss;
				cached_radial_coord_ss_derivatives[mu] = dot_coord_ss;
			}
			}

				const double outer_type_coeff = regression_coeffs[shared_type_offset + type_outer];
				const double pair_type_scale = center_type_coeff * outer_type_coeff;
				const double gate_residual = TwoLayerGateNeighborResidual(nbh, j);
				PrepareTwoLayerGateNeighborMuBuffers(
					type_outer, center_type_coeff, outer_type_coeff, gate_residual);
				const double* gate_type_scale_by_mu =
					two_layer_gate_type_scale_mu_buffer_.data();
				double* gate_scaled_radial_by_mu =
					two_layer_gate_scaled_mu_vals_buffer_.data();
				for (int mu = 0; mu < radial_func_count; ++mu)
					gate_scaled_radial_by_mu[mu] =
						gate_type_scale_by_mu[mu] * cached_radial_values[mu];
				for (int i = 0; i < alpha_index_basic_count; ++i) {
					const int mu = basic_mu_cache_[i];
					const int sh_index = basic_sh_index_cache_[i];
					moment_vals[i] += gate_scaled_radial_by_mu[mu] * cached_sh_values[sh_index];
				}
	}

	if (UseSHProductRows()) {
		ApplySHProductRowsForward();
	} else {
		for (size_t p = 0; p < sh_products_.size(); ++p) {
			const SHProduct& product = sh_products_[p];
			moment_vals[product.target] +=
				product.coeff * moment_vals[product.left] * moment_vals[product.right];
		}
	}

	basis_vals[0] = 1.0;
	for (int i = 0; i < alpha_scalar_moments; ++i)
		basis_vals[1 + i] = moment_vals[alpha_moment_mapping[i]];
}

void MLMTPR::CalcSHBasisFuncsDers(const Neighborhood& nbh)
{
	const bool use_lmp_table = SHUsesPrecomputedLmpTable(p_RadialBasis);
	if (sh_l_max_ < 0 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,6].");

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	if (type_central >= species_count)
		throw MlipException("Too few species count in the MTP potential!");

	if (nbh.count != moment_ders.size2 || alpha_moments_count != moment_ders.size1)
		moment_ders.resize(alpha_moments_count, nbh.count, 3);
	std::fill(moment_vals, moment_vals + alpha_moments_count, 0.0);
	moment_ders.set(0);

	double sh_values[kMaxSHComponents];
	double sh_ders[3 * kMaxSHComponents];
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& rb_ders = basis_radial_ders_buffer_;
	std::vector<double>& radial_values = grad_mu_contract_vals_;
	std::vector<double>& radial_derivatives = grad_mu_contract_ders_;
	const int radial_coeff_base = C + 2 * C * C * K_;
	const int shared_type_offset = radial_coeff_base + R;
	const bool use_edge_cache = HasTwoLayerEdgePrimitiveCache(-1, true, false);
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const int mu_stride = radial_func_count;

	for (int j = 0; j < nbh.count; ++j) {
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const double inv_r = 1.0 / r;
		const int type_outer = nbh.types[j];
		const double* sh_values_use = sh_values;
		const double* sh_ders_use = sh_ders;
		const double* radial_values_use = radial_values.data();
		const double* radial_derivatives_use = radial_derivatives.data();
		if (use_edge_cache) {
			const size_t edge = TwoLayerEdgePrimitiveOffset(-1, j);
			sh_values_use =
				two_layer_edge_sh_values_cache_.data() + edge * sh_count;
			sh_ders_use =
				two_layer_edge_sh_ders_cache_.data() + edge * 3 * sh_count;
			radial_values_use =
				two_layer_edge_mu_vals_cache_.data() + edge * mu_stride;
			radial_derivatives_use =
				two_layer_edge_mu_ders_cache_.data() + edge * mu_stride;
	} else {
		EvalRealSH(rvec, r, sh_l_max_, sh_values, sh_ders);

			if (use_lmp_table) {
				InterpolateSHLmpMuTable(radial_list,
				                         radial_der_list,
				                         inv_dr,
				                         C,
				                         type_central,
				                         type_outer,
				                         r,
				                         radial_func_count,
				                         radial_values.data(),
				                         radial_derivatives.data());
			} else {
				for (int eval_block = 0; eval_block < static_cast<int>(radial_eval_to_scaling_block_.size()); ++eval_block) {
					const int scaling_block = radial_eval_to_scaling_block_[eval_block];
					const int basis_k = radial_eval_to_basis_k_[eval_block];
					p_RadialBasis->RB_Calc(
						r,
						regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer],
						regression_coeffs[C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer],
						basis_k);
					for (int xi = 0; xi < R; ++xi) {
						rb_vals[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
						rb_ders[eval_block * R + xi] = p_RadialBasis->rb_ders[xi] * scaling;
					}
				}
				for (int mu = 0; mu < radial_func_count; ++mu) {
					const int radial_base = mu_to_radial_eval_block_[mu] * R;
					const int radial_offset = radial_coeff_base + mu * (R + C);
					double radial_value = 0.0;
					double radial_der = 0.0;
					for (int xi = 0; xi < R; ++xi) {
						const double coeff = regression_coeffs[radial_offset + xi];
						radial_value += coeff * rb_vals[radial_base + xi];
						radial_der += coeff * rb_ders[radial_base + xi];
					}
					radial_values[mu] = radial_value;
					radial_derivatives[mu] = radial_der;
				}
			}
		}

			const double center_type_coeff = regression_coeffs[shared_type_offset + type_central];
			const double outer_type_coeff = regression_coeffs[shared_type_offset + type_outer];
			const double pair_type_scale = center_type_coeff * outer_type_coeff;
			const double gate_residual = TwoLayerGateNeighborResidual(nbh, j);
			PrepareTwoLayerGateNeighborMuBuffers(
				type_outer, center_type_coeff, outer_type_coeff, gate_residual);
			const double* gate_type_scale_by_mu =
				two_layer_gate_type_scale_mu_buffer_.data();
			double* gate_scaled_radial_by_mu =
				two_layer_gate_scaled_mu_vals_buffer_.data();
			double* gate_scaled_der_by_mu =
				two_layer_gate_scaled_mu_ders_buffer_.data();
			for (int mu = 0; mu < radial_func_count; ++mu) {
				const double type_scale = gate_type_scale_by_mu[mu];
				gate_scaled_radial_by_mu[mu] = type_scale * radial_values_use[mu];
				gate_scaled_der_by_mu[mu] = type_scale * radial_derivatives_use[mu];
			}

					for (int i = 0; i < alpha_index_basic_count; ++i) {
						const int mu = basic_mu_cache_[i];
					const int sh_index = basic_sh_index_cache_[i];
					const int sh_der_index = basic_sh_der_index_cache_[i];
					const double radial_value = gate_scaled_radial_by_mu[mu];
					const double radial_der = gate_scaled_der_by_mu[mu];

					const double y = sh_values_use[sh_index];
					moment_vals[i] += radial_value * y;
					for (int a = 0; a < 3; ++a) {
						const double dy = sh_ders_use[sh_der_index + a];
						moment_ders(i, j, a) += radial_der * rvec[a] * inv_r * y + radial_value * dy;
			}
		}
	}

	basis_vals[0] = 1.0;
	if (basis_ders.size1 != alpha_count || basis_ders.size2 != nbh.count)
		basis_ders.resize(alpha_count, nbh.count, 3);
	if (active_two_layer_gate_values_ == nullptr)
		basis_ders.set(0);

	const bool use_product_rows = UseSHProductRows();
	if (use_product_rows) {
		ApplySHProductRowsDers(nbh);
	} else {
		for (size_t p = 0; p < sh_products_.size(); ++p) {
			const SHProduct& product = sh_products_[p];
			const double left_value = moment_vals[product.left];
			const double right_value = moment_vals[product.right];
			moment_vals[product.target] += product.coeff * left_value * right_value;
			for (int j = 0; j < nbh.count; ++j)
				for (int a = 0; a < 3; ++a)
					moment_ders(product.target, j, a) += product.coeff * (
						moment_ders(product.left, j, a) * right_value + left_value * moment_ders(product.right, j, a));
		}
	}

	for (int i = 0; i < alpha_scalar_moments; ++i) {
		if (use_product_rows && sh_scalar_terminal_product_[i])
			continue;
		const int node = alpha_moment_mapping[i];
		basis_vals[1 + i] = moment_vals[node];
		for (int j = 0; j < nbh.count; ++j)
			for (int a = 0; a < 3; ++a)
				basis_ders(1 + i, j, a) = moment_ders(node, j, a);
	}
}

void MLMTPR::CalcTwoLayerGateScalarValuesOnly(
	const Neighborhood& nbh,
	std::vector<double>& gate_scalar_values,
	int cache_atom_index)
{
	const bool use_lmp_table = SHUsesPrecomputedLmpTable(p_RadialBasis);
	if (sh_l_max_ < 0 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,6].");
	if (two_layer_gate_required_moments_.empty()
	    || two_layer_gate_required_moment_indices_.empty())
		BuildTwoLayerGateProductProgram();

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	if (type_central >= species_count)
		throw MlipException("Too few species count in the MTP potential!");

	for (int moment_index : two_layer_gate_required_moment_indices_)
		moment_vals[moment_index] = 0.0;
	double sh_values[kMaxSHComponents];
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& radial_values = grad_mu_contract_vals_;
	const int radial_coeff_base = C + 2 * C * C * K_;
	const int shared_type_offset = radial_coeff_base + R;
	const double center_type_coeff = regression_coeffs[shared_type_offset + type_central];
	const bool use_edge_cache =
		HasTwoLayerEdgePrimitiveCache(cache_atom_index, false);
	const bool use_gate_radial = TwoLayerGateUsesSharedRadial();
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const int mu_stride = radial_func_count;
	const int gate_mu_stride = use_gate_radial
		? static_cast<int>(two_layer_gate_required_mu_indices_.size())
		: 0;

	for (int j = 0; j < nbh.count; ++j) {
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const int type_outer = nbh.types[j];
		const double* sh_values_use = sh_values;
		const double* radial_values_use = radial_values.data();
		if (use_edge_cache) {
			const size_t edge = TwoLayerEdgePrimitiveOffset(cache_atom_index, j);
			sh_values_use =
				two_layer_edge_sh_values_cache_.data() + edge * sh_count;
			radial_values_use =
				(use_gate_radial
					? two_layer_edge_gate_mu_vals_cache_.data()
					: two_layer_edge_mu_vals_cache_.data())
				+ edge * static_cast<size_t>(
					use_gate_radial ? gate_mu_stride : mu_stride);
	} else {
		EvalRealSHValuesOnly(rvec, r, sh_l_max_, sh_values);

			if (use_lmp_table) {
				InterpolateSHLmpMuTable(use_gate_radial ? two_layer_gate_radial_list : radial_list,
				                         use_gate_radial ? two_layer_gate_radial_der_list : radial_der_list,
				                         inv_dr,
				                         C,
				                         type_central,
				                         type_outer,
				                         r,
				                         radial_func_count,
				                         radial_values.data(),
				                         nullptr);
			} else {
				for (int eval_block : two_layer_gate_required_radial_eval_blocks_) {
					const int scaling_block = radial_eval_to_scaling_block_[eval_block];
					const int basis_k = radial_eval_to_basis_k_[eval_block];
					p_RadialBasis->RB_CalcValsOnly(
						r,
						regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer],
						regression_coeffs[C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer],
						basis_k);
					for (int xi = 0; xi < R; ++xi)
						rb_vals[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
				}
				for (int mu : two_layer_gate_required_mu_indices_) {
					const int radial_base = mu_to_radial_eval_block_[mu] * R;
					const int radial_offset = TwoLayerGateRadialCoeffIndex(mu, 0);
					double radial_value = 0.0;
					for (int xi = 0; xi < R; ++xi)
						radial_value += regression_coeffs[radial_offset + xi] * rb_vals[radial_base + xi];
					radial_values[mu] = radial_value;
				}
			}
		}

		const double outer_type_coeff = regression_coeffs[shared_type_offset + type_outer];
		const double type_scale = center_type_coeff * outer_type_coeff;
		for (int bi = 0;
		     bi < static_cast<int>(two_layer_gate_required_basic_indices_.size());
		     ++bi) {
			const int basic_index = two_layer_gate_required_basic_indices_[bi];
			const int mu = basic_mu_cache_[basic_index];
			const int mu_index =
				(use_edge_cache && use_gate_radial)
					? two_layer_gate_required_basic_dense_mu_indices_[bi]
					: mu;
			const int sh_index = basic_sh_index_cache_[basic_index];
			moment_vals[basic_index] +=
				type_scale * radial_values_use[mu_index] * sh_values_use[sh_index];
		}
	}

	for (int product_index : two_layer_gate_required_product_indices_) {
		const SHProduct& product = sh_products_[product_index];
		moment_vals[product.target] +=
			product.coeff * moment_vals[product.left] * moment_vals[product.right];
	}

	gate_scalar_values.assign(TwoLayerGateWeightCount(), 0.0);
	for (int q = 0; q < TwoLayerGateWeightCount(); ++q) {
		const int scalar_index = two_layer_gate_scalar_indices_[q];
		gate_scalar_values[q] = moment_vals[alpha_moment_mapping[scalar_index]];
	}
}

void MLMTPR::CalcTwoLayerGateScalarDers(
	const Neighborhood& nbh,
	std::vector<double>& gate_scalar_ders)
{
	const bool use_lmp_table = SHUsesPrecomputedLmpTable(p_RadialBasis);
	if (sh_l_max_ < 0 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,6].");
	if (two_layer_gate_required_moments_.empty()
	    || two_layer_gate_required_moment_indices_.empty())
		BuildTwoLayerGateProductProgram();

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	if (type_central >= species_count)
		throw MlipException("Too few species count in the MTP potential!");

	const int moment_stride = alpha_moments_count;
	const int coord_stride = 3 * moment_stride;
	std::fill(moment_vals, moment_vals + alpha_moments_count, 0.0);
	sh_gate_moment_ders_.assign(
		static_cast<size_t>(nbh.count) * coord_stride, 0.0);
	gate_scalar_ders.assign(
		static_cast<size_t>(TwoLayerGateWeightCount()) * nbh.count * 3, 0.0);

	double sh_values[kMaxSHComponents];
	double sh_ders[3 * kMaxSHComponents];
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& rb_ders = basis_radial_ders_buffer_;
	std::vector<double>& radial_values = grad_mu_contract_vals_;
	std::vector<double>& radial_derivatives = grad_mu_contract_ders_;
	const int radial_coeff_base = C + 2 * C * C * K_;
	const int shared_type_offset = radial_coeff_base + R;
	const double center_type_coeff = regression_coeffs[shared_type_offset + type_central];
	const bool use_gate_radial = TwoLayerGateUsesSharedRadial();

	for (int j = 0; j < nbh.count; ++j) {
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const double inv_r = 1.0 / r;
		const int type_outer = nbh.types[j];
		EvalRealSH(rvec, r, sh_l_max_, sh_values, sh_ders);

		if (use_lmp_table) {
			InterpolateSHLmpMuTable(use_gate_radial ? two_layer_gate_radial_list : radial_list,
			                         use_gate_radial ? two_layer_gate_radial_der_list : radial_der_list,
			                         inv_dr,
			                         C,
			                         type_central,
			                         type_outer,
			                         r,
			                         radial_func_count,
			                         radial_values.data(),
			                         radial_derivatives.data());
		} else {
			for (int eval_block : two_layer_gate_required_radial_eval_blocks_) {
				const int scaling_block = radial_eval_to_scaling_block_[eval_block];
				const int basis_k = radial_eval_to_basis_k_[eval_block];
				p_RadialBasis->RB_Calc(
					r,
					regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer],
					regression_coeffs[C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer],
					basis_k);
				for (int xi = 0; xi < R; ++xi) {
					rb_vals[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
					rb_ders[eval_block * R + xi] = p_RadialBasis->rb_ders[xi] * scaling;
				}
			}
			for (int mu : two_layer_gate_required_mu_indices_) {
				const int radial_base = mu_to_radial_eval_block_[mu] * R;
				const int radial_offset = TwoLayerGateRadialCoeffIndex(mu, 0);
				double radial_value = 0.0;
				double radial_der = 0.0;
				for (int xi = 0; xi < R; ++xi) {
					const double coeff = regression_coeffs[radial_offset + xi];
					radial_value += coeff * rb_vals[radial_base + xi];
					radial_der += coeff * rb_ders[radial_base + xi];
				}
				radial_values[mu] = radial_value;
				radial_derivatives[mu] = radial_der;
			}
		}

		const double outer_type_coeff = regression_coeffs[shared_type_offset + type_outer];
		const double type_scale = center_type_coeff * outer_type_coeff;
		double* moment_der = sh_gate_moment_ders_.data()
			+ static_cast<size_t>(j) * coord_stride;
		for (int basic_index : two_layer_gate_required_basic_indices_) {
			const int mu = basic_mu_cache_[basic_index];
			const int sh_index = basic_sh_index_cache_[basic_index];
			const int sh_der_index = basic_sh_der_index_cache_[basic_index];
			const double radial_value = radial_values[mu];
			const double radial_der = radial_derivatives[mu];
			const double y = sh_values[sh_index];
			moment_vals[basic_index] += type_scale * radial_value * y;
			for (int a = 0; a < 3; ++a) {
				const double dy = sh_ders[sh_der_index + a];
				moment_der[a * moment_stride + basic_index] +=
					type_scale * (radial_der * rvec[a] * inv_r * y + radial_value * dy);
			}
		}
	}

	for (int product_index : two_layer_gate_required_product_indices_) {
		const SHProduct& product = sh_products_[product_index];
		const double left_value = moment_vals[product.left];
		const double right_value = moment_vals[product.right];
		moment_vals[product.target] += product.coeff * left_value * right_value;
		for (int j = 0; j < nbh.count; ++j) {
			double* moment_der = sh_gate_moment_ders_.data()
				+ static_cast<size_t>(j) * coord_stride;
			for (int a = 0; a < 3; ++a) {
				moment_der[a * moment_stride + product.target] += product.coeff * (
					moment_der[a * moment_stride + product.left] * right_value
					+ left_value * moment_der[a * moment_stride + product.right]);
			}
		}
	}

	for (int q = 0; q < TwoLayerGateWeightCount(); ++q) {
		const int scalar_index = two_layer_gate_scalar_indices_[q];
		const int moment = alpha_moment_mapping[scalar_index];
		for (int j = 0; j < nbh.count; ++j) {
			const double* moment_der = sh_gate_moment_ders_.data()
				+ static_cast<size_t>(j) * coord_stride;
			for (int a = 0; a < 3; ++a)
				gate_scalar_ders[(static_cast<size_t>(q) * nbh.count + j) * 3 + a] =
					moment_der[a * moment_stride + moment];
		}
	}
}

void MLMTPR::CalcTwoLayerGateWeightedScalarDers(
	const Neighborhood& nbh,
	std::vector<Vector3>& gate_scalar_ders,
	int cache_atom_index)
{
	const bool use_lmp_table = SHUsesPrecomputedLmpTable(p_RadialBasis);
	if (sh_l_max_ < 0 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,6].");
	if (two_layer_gate_required_moments_.empty()
	    || two_layer_gate_required_moment_indices_.empty())
		BuildTwoLayerGateProductProgram();

	gate_scalar_ders.assign(
		static_cast<size_t>(nbh.count), Vector3(0.0, 0.0, 0.0));
	if (TwoLayerGateWeightCount() == 0)
		return;

	bool has_weight = false;
	for (int q = 0; q < TwoLayerGateWeightCount(); ++q)
		if (TwoLayerGateWeight(q) != 0.0) {
			has_weight = true;
			break;
		}
	if (!has_weight)
		return;

	bool loaded_moment_cache = false;
	if (cache_atom_index >= 0 && alpha_moments_count > 0) {
		const int cached_moment_count =
			static_cast<int>(two_layer_gate_required_moment_indices_.size());
		const size_t offset =
			static_cast<size_t>(cache_atom_index) * cached_moment_count;
		if (cached_moment_count > 0
		    && offset + cached_moment_count <=
			two_layer_gate_moment_values_cache_.size()) {
			const double* cached_moments =
				two_layer_gate_moment_values_cache_.data() + offset;
			for (int i = 0; i < cached_moment_count; ++i)
				moment_vals[two_layer_gate_required_moment_indices_[i]] =
					cached_moments[i];
			loaded_moment_cache = true;
		}
	}
	if (!loaded_moment_cache)
		CalcTwoLayerGateScalarValuesOnly(nbh,
		                                  sh_gate_scalar_values_,
		                                  cache_atom_index);

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	if (type_central >= species_count)
		throw MlipException("Too few species count in the MTP potential!");

	site_energy_ders_wrt_moments_.assign(alpha_moments_count, 0.0);
	for (int q = 0; q < TwoLayerGateWeightCount(); ++q) {
		const double weight = TwoLayerGateWeight(q);
		if (weight == 0.0)
			continue;
		const int scalar_index = two_layer_gate_scalar_indices_[q];
		site_energy_ders_wrt_moments_[alpha_moment_mapping[scalar_index]] += weight;
	}

	for (int p = static_cast<int>(two_layer_gate_required_product_indices_.size()) - 1;
	     p >= 0; --p) {
		const SHProduct& product =
			sh_products_[two_layer_gate_required_product_indices_[p]];
		const double adj_target = site_energy_ders_wrt_moments_[product.target];
		if (adj_target == 0.0)
			continue;
		site_energy_ders_wrt_moments_[product.left] +=
			product.coeff * moment_vals[product.right] * adj_target;
		site_energy_ders_wrt_moments_[product.right] +=
			product.coeff * moment_vals[product.left] * adj_target;
	}

	const int radial_coeff_base = C + 2 * C * C * K_;
	const int shared_type_offset = radial_coeff_base + R;
	const double center_type_coeff = regression_coeffs[shared_type_offset + type_central];

	double sh_values[kMaxSHComponents];
	double sh_ders[3 * kMaxSHComponents];
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& rb_ders = basis_radial_ders_buffer_;
	std::vector<double>& radial_values = grad_mu_contract_vals_;
	std::vector<double>& radial_derivatives = grad_mu_contract_ders_;
	const bool use_edge_cache =
		HasTwoLayerEdgePrimitiveCache(cache_atom_index, true, false);
	const bool use_gate_radial = TwoLayerGateUsesSharedRadial();
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const int mu_stride = radial_func_count;
	const int gate_mu_stride = use_gate_radial
		? static_cast<int>(two_layer_gate_required_mu_indices_.size())
		: 0;

	for (int j = 0; j < nbh.count; ++j) {
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const double inv_r = 1.0 / r;
		const int type_outer = nbh.types[j];
		const double* sh_values_use = sh_values;
		const double* sh_ders_use = sh_ders;
		const double* radial_values_use = radial_values.data();
		const double* radial_derivatives_use = radial_derivatives.data();
					if (use_edge_cache) {
						const size_t edge = TwoLayerEdgePrimitiveOffset(cache_atom_index, j);
						sh_values_use =
							two_layer_edge_sh_values_cache_.data() + edge * sh_count;
						sh_ders_use =
							two_layer_edge_sh_ders_cache_.data() + edge * 3 * sh_count;
						radial_values_use =
							(use_gate_radial
								? two_layer_edge_gate_mu_vals_cache_.data()
								: two_layer_edge_mu_vals_cache_.data())
							+ edge * static_cast<size_t>(
								use_gate_radial ? gate_mu_stride : mu_stride);
						radial_derivatives_use =
							(use_gate_radial
								? two_layer_edge_gate_mu_ders_cache_.data()
								: two_layer_edge_mu_ders_cache_.data())
							+ edge * static_cast<size_t>(
								use_gate_radial ? gate_mu_stride : mu_stride);
		} else {
			EvalRealSH(rvec, r, sh_l_max_, sh_values, sh_ders);

			if (use_lmp_table) {
				InterpolateSHLmpMuTable(use_gate_radial ? two_layer_gate_radial_list : radial_list,
				                         use_gate_radial ? two_layer_gate_radial_der_list : radial_der_list,
				                         inv_dr,
				                         C,
				                         type_central,
				                         type_outer,
				                         r,
				                         radial_func_count,
				                         radial_values.data(),
				                         radial_derivatives.data());
			} else {
				for (int eval_block : two_layer_gate_required_radial_eval_blocks_) {
						const int scaling_block = radial_eval_to_scaling_block_[eval_block];
						const int basis_k = radial_eval_to_basis_k_[eval_block];
						p_RadialBasis->RB_Calc(
							r,
							regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer],
							regression_coeffs[C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer],
							basis_k);
						for (int xi = 0; xi < R; ++xi) {
							rb_vals[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
							rb_ders[eval_block * R + xi] = p_RadialBasis->rb_ders[xi] * scaling;
						}
					}

					for (int mu : two_layer_gate_required_mu_indices_) {
						const int radial_base = mu_to_radial_eval_block_[mu] * R;
						const int radial_offset = TwoLayerGateRadialCoeffIndex(mu, 0);
						double radial_value = 0.0;
						double radial_der = 0.0;
						for (int xi = 0; xi < R; ++xi) {
							const double coeff = regression_coeffs[radial_offset + xi];
							radial_value += coeff * rb_vals[radial_base + xi];
							radial_der += coeff * rb_ders[radial_base + xi];
						}
						radial_values[mu] = radial_value;
						radial_derivatives[mu] = radial_der;
					}
			}
		}

		const double outer_type_coeff = regression_coeffs[shared_type_offset + type_outer];
		const double type_scale = center_type_coeff * outer_type_coeff;
		Vector3 gate_der(0.0, 0.0, 0.0);
		for (int bi = 0;
		     bi < static_cast<int>(two_layer_gate_required_basic_indices_.size());
		     ++bi) {
			const int basic_index = two_layer_gate_required_basic_indices_[bi];
			const double adjoint = site_energy_ders_wrt_moments_[basic_index];
			if (adjoint == 0.0)
				continue;
			const int mu = basic_mu_cache_[basic_index];
			const int mu_index =
				(use_edge_cache && use_gate_radial)
					? two_layer_gate_required_basic_dense_mu_indices_[bi]
					: mu;
			const int sh_index = basic_sh_index_cache_[basic_index];
			const int sh_der_index = basic_sh_der_index_cache_[basic_index];
			const double radial_value = radial_values_use[mu_index];
			const double radial_der = radial_derivatives_use[mu_index];
			const double y = sh_values_use[sh_index];
			for (int a = 0; a < 3; ++a) {
				const double dy = sh_ders_use[sh_der_index + a];
				gate_der[a] += adjoint * type_scale
					* (radial_der * rvec[a] * inv_r * y + radial_value * dy);
			}
		}
		gate_scalar_ders[j] = gate_der;
	}
}

void MLMTPR::AccumulateTwoLayerGateScalarParamGrad(
	const Neighborhood& nbh,
	std::vector<double>& out_grad_accumulator,
	double gate_adjoint,
	const Vector3* gate_der_weights,
	int cache_atom_index,
	const double* gate_moment_tangents,
	double gate_moment_tangent_scale)
{
	bool has_der_weights = false;
	if (gate_der_weights != nullptr) {
		for (int j = 0; j < nbh.count; ++j)
			if (gate_der_weights[j].NormSq() != 0.0) {
				has_der_weights = true;
				break;
			}
	}
	if ((gate_adjoint == 0.0 && !has_der_weights)
	    || TwoLayerGateWeightCount() == 0)
		return;
	if (SHUsesPrecomputedLmpTable(p_RadialBasis))
		ERROR("SUS2-SH coefficient gradients require direct radial basis evaluation.");
	if (sh_l_max_ < 0 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,6].");
	if (two_layer_gate_required_moments_.empty()
	    || two_layer_gate_required_moment_indices_.empty())
		BuildTwoLayerGateProductProgram();

	bool loaded_moment_cache = false;
	if (cache_atom_index >= 0 && alpha_moments_count > 0) {
		const int cached_moment_count =
			static_cast<int>(two_layer_gate_required_moment_indices_.size());
		const size_t offset =
			static_cast<size_t>(cache_atom_index) * cached_moment_count;
		if (cached_moment_count > 0
		    && offset + cached_moment_count <=
			two_layer_gate_moment_values_cache_.size()) {
			const double* cached_moments =
				two_layer_gate_moment_values_cache_.data() + offset;
			for (int i = 0; i < cached_moment_count; ++i)
				moment_vals[two_layer_gate_required_moment_indices_[i]] =
					cached_moments[i];
			loaded_moment_cache = true;
		}
	}
	if (!loaded_moment_cache)
		CalcTwoLayerGateScalarValuesOnly(nbh,
		                                  sh_gate_scalar_values_,
		                                  cache_atom_index);
	out_grad_accumulator.resize(CoeffCount());

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	if (type_central >= species_count)
		throw MlipException("Too few species count in the MTP potential!");

	if (static_cast<int>(site_energy_ders_wrt_moments_.size())
	    != alpha_moments_count)
		site_energy_ders_wrt_moments_.resize(alpha_moments_count);
	if (static_cast<int>(grad_dloss_dsenders_.size()) != alpha_moments_count)
		grad_dloss_dsenders_.resize(alpha_moments_count);
	if (static_cast<int>(grad_dloss_dmom_.size()) != alpha_moments_count)
		grad_dloss_dmom_.resize(alpha_moments_count);
	for (int moment_index : two_layer_gate_required_moment_indices_) {
		site_energy_ders_wrt_moments_[moment_index] = 0.0;
		grad_dloss_dsenders_[moment_index] = 0.0;
		grad_dloss_dmom_[moment_index] = 0.0;
	}
	for (int q = 0; q < TwoLayerGateWeightCount(); ++q) {
		const double weight = TwoLayerGateWeight(q);
		if (weight == 0.0)
			continue;
		const int scalar_index = two_layer_gate_scalar_indices_[q];
		const int moment = alpha_moment_mapping[scalar_index];
		site_energy_ders_wrt_moments_[moment] += weight;
	}

	for (int p = static_cast<int>(two_layer_gate_required_product_indices_.size()) - 1;
	     p >= 0; --p) {
		const SHProduct& product =
			sh_products_[two_layer_gate_required_product_indices_[p]];
		const double adj_target = site_energy_ders_wrt_moments_[product.target];
		if (adj_target == 0.0)
			continue;
		site_energy_ders_wrt_moments_[product.left] +=
			product.coeff * moment_vals[product.right] * adj_target;
		site_energy_ders_wrt_moments_[product.right] +=
			product.coeff * moment_vals[product.left] * adj_target;
	}

	const int radial_coeff_base = C + 2 * C * C * K_;
	const int shared_type_offset = radial_coeff_base + R;
	const int eval_block_count =
		static_cast<int>(radial_eval_to_scaling_block_.size());
	const int radial_val_stride = eval_block_count * R;
	const int radial_der_stride = radial_val_stride * 5;
	const double center_type_coeff =
		regression_coeffs[shared_type_offset + type_central];

	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& rb_ders = radial_ders_buffer_;
	std::vector<double>& radial_values = grad_mu_contract_vals_;
	std::vector<double>& radial_derivatives = grad_mu_contract_ders_;
	std::vector<double>& radial_s_derivatives = grad_mu_contract_ders_s_;
	std::vector<double>& radial_ss_derivatives = grad_mu_contract_ders_ss_;
	std::vector<double>& radial_coord_s_derivatives = grad_mu_contract_coord_ders_s_;
	std::vector<double>& radial_coord_ss_derivatives = grad_mu_contract_coord_ders_ss_;
	if (static_cast<int>(rb_vals.size()) < radial_val_stride)
		rb_vals.resize(radial_val_stride);
	if (static_cast<int>(rb_ders.size()) < radial_der_stride)
		rb_ders.resize(radial_der_stride);
	if (static_cast<int>(radial_values.size()) < radial_func_count)
		radial_values.resize(radial_func_count);
	if (static_cast<int>(radial_derivatives.size()) < radial_func_count)
		radial_derivatives.resize(radial_func_count);
	if (static_cast<int>(radial_s_derivatives.size()) < radial_func_count)
		radial_s_derivatives.resize(radial_func_count);
	if (static_cast<int>(radial_ss_derivatives.size()) < radial_func_count)
		radial_ss_derivatives.resize(radial_func_count);
	if (static_cast<int>(radial_coord_s_derivatives.size()) < radial_func_count)
		radial_coord_s_derivatives.resize(radial_func_count);
	if (static_cast<int>(radial_coord_ss_derivatives.size()) < radial_func_count)
		radial_coord_ss_derivatives.resize(radial_func_count);
	if (static_cast<int>(grad_radial_coeff_value_accum_.size()) != radial_func_count)
		grad_radial_coeff_value_accum_.resize(radial_func_count);
	if (static_cast<int>(grad_radial_coeff_coord_accum_.size()) != radial_func_count)
		grad_radial_coeff_coord_accum_.resize(radial_func_count);

	double sh_values[kMaxSHComponents];
	double sh_ders[3 * kMaxSHComponents];
	const bool use_edge_cache =
		HasTwoLayerEdgePrimitiveCache(cache_atom_index, true);
	const bool use_gate_radial = TwoLayerGateUsesSharedRadial();
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const int mu_stride = radial_func_count;
	const int gate_mu_stride = use_gate_radial
		? static_cast<int>(two_layer_gate_required_mu_indices_.size())
		: 0;
	if (has_der_weights) {
		if (gate_moment_tangents != nullptr) {
			for (int moment_index : two_layer_gate_required_moment_indices_)
				grad_dloss_dsenders_[moment_index] =
					gate_moment_tangent_scale
					* gate_moment_tangents[moment_index];
		} else {
			for (int j = 0; j < nbh.count; ++j) {
				const Vector3& dir = gate_der_weights[j];
				if (dir.NormSq() == 0.0)
					continue;
				const Vector3& rvec = nbh.vecs[j];
				const double r = nbh.dists[j];
				const double inv_r = 1.0 / r;
				const int type_outer = nbh.types[j];
				const double wr =
					(dir[0] * rvec[0] + dir[1] * rvec[1] + dir[2] * rvec[2])
					* inv_r;
				const double* sh_values_use = sh_values;
				const double* sh_ders_use = sh_ders;
				const double* radial_values_use = radial_values.data();
				const double* radial_derivatives_use = radial_derivatives.data();
				if (use_edge_cache) {
					const size_t edge = TwoLayerEdgePrimitiveOffset(cache_atom_index, j);
					sh_values_use =
						two_layer_edge_sh_values_cache_.data() + edge * sh_count;
						sh_ders_use =
							two_layer_edge_sh_ders_cache_.data() + edge * 3 * sh_count;
							radial_values_use =
								(use_gate_radial
									? two_layer_edge_gate_mu_vals_cache_.data()
									: two_layer_edge_mu_vals_cache_.data())
								+ edge * static_cast<size_t>(
									use_gate_radial ? gate_mu_stride : mu_stride);
							radial_derivatives_use =
								(use_gate_radial
									? two_layer_edge_gate_mu_ders_cache_.data()
									: two_layer_edge_mu_ders_cache_.data())
								+ edge * static_cast<size_t>(
									use_gate_radial ? gate_mu_stride : mu_stride);
					} else {
					EvalRealSH(rvec, r, sh_l_max_, sh_values, sh_ders);

					for (int eval_block : two_layer_gate_required_radial_eval_blocks_) {
						const int scaling_block = radial_eval_to_scaling_block_[eval_block];
						const int basis_k = radial_eval_to_basis_k_[eval_block];
						p_RadialBasis->RB_Calc(
							r,
							regression_coeffs[
								C + 2 * scaling_block * C * C + C * type_central + type_outer],
							regression_coeffs[
								C + 2 * scaling_block * C * C + C * C
								+ C * type_central + type_outer],
							basis_k);
						for (int xi = 0; xi < R; ++xi) {
							rb_vals[eval_block * R + xi] =
								p_RadialBasis->rb_vals[xi] * scaling;
								rb_ders[eval_block * R * 5 + xi] =
									p_RadialBasis->rb_ders[xi] * scaling;
							}
						}
						for (int mu : two_layer_gate_required_mu_indices_) {
							const int radial_base = mu_to_radial_eval_block_[mu] * R;
							const int deriv_base = 5 * radial_base;
							const int radial_offset = TwoLayerGateRadialCoeffIndex(mu, 0);
							double radial_value = 0.0;
							double radial_der = 0.0;
							for (int xi = 0; xi < R; ++xi) {
								const double coeff = regression_coeffs[radial_offset + xi];
							radial_value += coeff * rb_vals[radial_base + xi];
							radial_der += coeff * rb_ders[deriv_base + xi];
						}
						radial_values[mu] = radial_value;
						radial_derivatives[mu] = radial_der;
					}
				}

					const double outer_type_coeff =
						regression_coeffs[shared_type_offset + type_outer];
					const double type_scale = center_type_coeff * outer_type_coeff;
					for (int bi = 0;
					     bi < static_cast<int>(two_layer_gate_required_basic_indices_.size());
					     ++bi) {
						const int basic_index = two_layer_gate_required_basic_indices_[bi];
						const int mu = basic_mu_cache_[basic_index];
						const int mu_index =
							(use_edge_cache && use_gate_radial)
								? two_layer_gate_required_basic_dense_mu_indices_[bi]
								: mu;
						const int sh_index = basic_sh_index_cache_[basic_index];
						const int sh_der_index = basic_sh_der_index_cache_[basic_index];
						const double radial_value = radial_values_use[mu_index];
						const double radial_der = radial_derivatives_use[mu_index];
						const double y = sh_values_use[sh_index];
						const double wdy =
							dir[0] * sh_ders_use[sh_der_index]
							+ dir[1] * sh_ders_use[sh_der_index + 1]
						+ dir[2] * sh_ders_use[sh_der_index + 2];
					grad_dloss_dsenders_[basic_index] +=
						type_scale * (radial_der * wr * y + radial_value * wdy);
				}
			}

			for (int product_index : two_layer_gate_required_product_indices_) {
				const SHProduct& product = sh_products_[product_index];
				grad_dloss_dsenders_[product.target] += product.coeff * (
					grad_dloss_dsenders_[product.left] * moment_vals[product.right]
					+ moment_vals[product.left] * grad_dloss_dsenders_[product.right]);
			}
		}

		for (int product_index : two_layer_gate_required_product_indices_) {
			const SHProduct& product = sh_products_[product_index];
			const double adj_target = site_energy_ders_wrt_moments_[product.target];
			if (adj_target == 0.0)
				continue;
			grad_dloss_dmom_[product.left] +=
				grad_dloss_dsenders_[product.right] * adj_target * product.coeff;
			grad_dloss_dmom_[product.right] +=
				grad_dloss_dsenders_[product.left] * adj_target * product.coeff;
		}

		for (int p = static_cast<int>(two_layer_gate_required_product_indices_.size()) - 1;
		     p >= 0; --p) {
			const SHProduct& product =
				sh_products_[two_layer_gate_required_product_indices_[p]];
			const double adj_target = grad_dloss_dmom_[product.target];
			if (adj_target == 0.0)
				continue;
			grad_dloss_dmom_[product.left] +=
				product.coeff * moment_vals[product.right] * adj_target;
			grad_dloss_dmom_[product.right] +=
				product.coeff * moment_vals[product.left] * adj_target;
		}
	}

	if (gate_adjoint != 0.0)
		for (int basic_index : two_layer_gate_required_basic_indices_)
			grad_dloss_dmom_[basic_index] +=
				gate_adjoint * site_energy_ders_wrt_moments_[basic_index];

	for (int j = 0; j < nbh.count; ++j) {
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const double inv_r = 1.0 / r;
		const int type_outer = nbh.types[j];
		const double wx = has_der_weights ? gate_der_weights[j][0] : 0.0;
		const double wy = has_der_weights ? gate_der_weights[j][1] : 0.0;
		const double wz = has_der_weights ? gate_der_weights[j][2] : 0.0;
		const double wr = has_der_weights
			? (wx * rvec[0] + wy * rvec[1] + wz * rvec[2]) * inv_r
			: 0.0;
		const double* sh_values_use = sh_values;
		const double* sh_ders_use = sh_ders;
		const double* rb_vals_use = rb_vals.data();
		const double* rb_ders_use = rb_ders.data();
		const double* radial_values_use = radial_values.data();
		const double* radial_derivatives_use = radial_derivatives.data();
		const double* radial_s_derivatives_use = radial_s_derivatives.data();
		const double* radial_ss_derivatives_use = radial_ss_derivatives.data();
		const double* radial_coord_s_derivatives_use =
			radial_coord_s_derivatives.data();
		const double* radial_coord_ss_derivatives_use =
			radial_coord_ss_derivatives.data();
		if (use_edge_cache) {
			const size_t edge = TwoLayerEdgePrimitiveOffset(cache_atom_index, j);
			sh_values_use =
				two_layer_edge_sh_values_cache_.data() + edge * sh_count;
			sh_ders_use =
				two_layer_edge_sh_ders_cache_.data() + edge * 3 * sh_count;
			rb_vals_use =
				two_layer_edge_radial_vals_cache_.data()
				+ edge * radial_val_stride;
			rb_ders_use =
				two_layer_edge_radial_ders_cache_.data()
				+ edge * radial_der_stride;
			radial_values_use =
				(use_gate_radial
					? two_layer_edge_gate_mu_vals_cache_.data()
					: two_layer_edge_mu_vals_cache_.data())
				+ edge * static_cast<size_t>(
					use_gate_radial ? gate_mu_stride : mu_stride);
			radial_derivatives_use =
				(use_gate_radial
					? two_layer_edge_gate_mu_ders_cache_.data()
					: two_layer_edge_mu_ders_cache_.data())
				+ edge * static_cast<size_t>(
					use_gate_radial ? gate_mu_stride : mu_stride);
			radial_s_derivatives_use =
				(use_gate_radial
					? two_layer_edge_gate_mu_ders_s_cache_.data()
					: two_layer_edge_mu_ders_s_cache_.data())
				+ edge * static_cast<size_t>(
					use_gate_radial ? gate_mu_stride : mu_stride);
			radial_ss_derivatives_use =
				(use_gate_radial
					? two_layer_edge_gate_mu_ders_ss_cache_.data()
					: two_layer_edge_mu_ders_ss_cache_.data())
				+ edge * static_cast<size_t>(
					use_gate_radial ? gate_mu_stride : mu_stride);
			radial_coord_s_derivatives_use =
				(use_gate_radial
					? two_layer_edge_gate_mu_coord_ders_s_cache_.data()
					: two_layer_edge_mu_coord_ders_s_cache_.data())
				+ edge * static_cast<size_t>(
					use_gate_radial ? gate_mu_stride : mu_stride);
			radial_coord_ss_derivatives_use =
				(use_gate_radial
					? two_layer_edge_gate_mu_coord_ders_ss_cache_.data()
					: two_layer_edge_mu_coord_ders_ss_cache_.data())
				+ edge * static_cast<size_t>(
					use_gate_radial ? gate_mu_stride : mu_stride);
		} else {
			EvalRealSH(rvec, r, sh_l_max_, sh_values, sh_ders);

			for (int eval_block : two_layer_gate_required_radial_eval_blocks_) {
				const int scaling_block = radial_eval_to_scaling_block_[eval_block];
				const int basis_k = radial_eval_to_basis_k_[eval_block];
				p_RadialBasis->RB_Calc(
					r,
					regression_coeffs[
						C + 2 * scaling_block * C * C + C * type_central + type_outer],
					regression_coeffs[
						C + 2 * scaling_block * C * C + C * C
						+ C * type_central + type_outer],
					basis_k);
				for (int xi = 0; xi < R; ++xi)
					rb_vals[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
				for (int xi = 0; xi < R * 5; ++xi)
					rb_ders[eval_block * R * 5 + xi] =
						p_RadialBasis->rb_ders[xi] * scaling;
			}

			for (int mu : two_layer_gate_required_mu_indices_) {
				const int radial_base = mu_to_radial_eval_block_[mu] * R;
				const int deriv_base = 5 * radial_base;
				const int radial_offset = TwoLayerGateRadialCoeffIndex(mu, 0);
				double radial_value = 0.0;
				double radial_der = 0.0;
				double radial_s = 0.0;
				double radial_coord_s = 0.0;
				double radial_ss = 0.0;
				double radial_coord_ss = 0.0;
				for (int xi = 0; xi < R; ++xi) {
					const double coeff = regression_coeffs[radial_offset + xi];
					radial_value += coeff * rb_vals[radial_base + xi];
					radial_der += coeff * rb_ders[deriv_base + xi];
					radial_s += coeff * rb_ders[deriv_base + xi + R];
					radial_coord_s += coeff * rb_ders[deriv_base + xi + 2 * R];
					radial_ss += coeff * rb_ders[deriv_base + xi + 3 * R];
					radial_coord_ss += coeff * rb_ders[deriv_base + xi + 4 * R];
				}
				radial_values[mu] = radial_value;
				radial_derivatives[mu] = radial_der;
				radial_s_derivatives[mu] = radial_s;
				radial_ss_derivatives[mu] = radial_ss;
				radial_coord_s_derivatives[mu] = radial_coord_s;
				radial_coord_ss_derivatives[mu] = radial_coord_ss;
			}
		}

		const double outer_type_coeff = regression_coeffs[shared_type_offset + type_outer];
			const double type_scale = center_type_coeff * outer_type_coeff;
			double* radial_coeff_value_accum = grad_radial_coeff_value_accum_.data();
			double* radial_coeff_coord_accum = grad_radial_coeff_coord_accum_.data();
			for (int mu : two_layer_gate_required_mu_indices_) {
				radial_coeff_value_accum[mu] = 0.0;
				radial_coeff_coord_accum[mu] = 0.0;
			}

		for (int bi = 0;
		     bi < static_cast<int>(two_layer_gate_required_basic_indices_.size());
		     ++bi) {
			const int basic_index = two_layer_gate_required_basic_indices_[bi];
			const double value_weight = grad_dloss_dmom_[basic_index];
			const double coord_weight = has_der_weights
				? site_energy_ders_wrt_moments_[basic_index]
				: 0.0;
			if (value_weight == 0.0 && coord_weight == 0.0)
				continue;
			const int mu = basic_mu_cache_[basic_index];
			const int mu_index =
				(use_edge_cache && use_gate_radial)
					? two_layer_gate_required_basic_dense_mu_indices_[bi]
					: mu;
			const int scaling_block = basic_scaling_block_cache_[basic_index];
			const int sh_index = basic_sh_index_cache_[basic_index];
			const int sh_der_index = basic_sh_der_index_cache_[basic_index];
			const double y = sh_values_use[sh_index];
			double dy[3];
			dy[0] = sh_ders_use[sh_der_index];
			dy[1] = sh_ders_use[sh_der_index + 1];
			dy[2] = sh_ders_use[sh_der_index + 2];
			const double wdy = wx * dy[0] + wy * dy[1] + wz * dy[2];
			const double dot_val = radial_values_use[mu_index];
			const double dot_der = radial_derivatives_use[mu_index];
			const double dot_s = radial_s_derivatives_use[mu_index];
			const double dot_coord_s = radial_coord_s_derivatives_use[mu_index];
			const double dot_ss = radial_ss_derivatives_use[mu_index];
			const double dot_coord_ss = radial_coord_ss_derivatives_use[mu_index];
			const double base_direction = dot_der * wr * y + dot_val * wdy;
			const double sigma_direction = dot_coord_s * wr * y + dot_s * wdy;
			const double shift_direction = dot_coord_ss * wr * y + dot_ss * wdy;
			radial_coeff_value_accum[mu] +=
				value_weight * y + coord_weight * wdy;
			radial_coeff_coord_accum[mu] += coord_weight * y;

			const int sigma_coeff_offset =
				C + 2 * C * C * scaling_block + type_central * C + type_outer;
			out_grad_accumulator[shared_type_offset + type_central] +=
				value_weight * outer_type_coeff * dot_val * y
				+ coord_weight * outer_type_coeff * base_direction;
			out_grad_accumulator[shared_type_offset + type_outer] +=
				value_weight * center_type_coeff * dot_val * y
				+ coord_weight * center_type_coeff * base_direction;
			out_grad_accumulator[sigma_coeff_offset] +=
				value_weight * type_scale * dot_s * y
				+ coord_weight * type_scale * sigma_direction;
			out_grad_accumulator[sigma_coeff_offset + C * C] +=
				value_weight * type_scale * dot_ss * y
				+ coord_weight * type_scale * shift_direction;
		}

		for (int mu : two_layer_gate_required_mu_indices_) {
			const double value_accum = radial_coeff_value_accum[mu];
			const double coord_accum = radial_coeff_coord_accum[mu];
			if (value_accum == 0.0 && coord_accum == 0.0)
				continue;
			const int radial_base = mu_to_radial_eval_block_[mu] * R;
			const int deriv_base = 5 * radial_base;
			const int radial_offset = TwoLayerGateRadialCoeffIndex(mu, 0);
			for (int xi = 0; xi < R; ++xi)
				out_grad_accumulator[radial_offset + xi] +=
					type_scale * (
						rb_vals_use[radial_base + xi] * value_accum
						+ rb_ders_use[deriv_base + xi] * wr * coord_accum);
		}
	}
}

void MLMTPR::CalcSHBasisGateDers(const Neighborhood& nbh,
                                 std::vector<double>& gate_basis_ders)
{
	const bool use_lmp_table = SHUsesPrecomputedLmpTable(p_RadialBasis);
	if (sh_l_max_ < 0 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,6].");

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	if (type_central >= species_count)
		throw MlipException("Too few species count in the MTP potential!");

	const int moment_stride = alpha_moments_count;
	const int scalar_stride = alpha_scalar_moments;
	sh_gate_moment_ders_.assign(
		static_cast<size_t>(nbh.count) * moment_stride, 0.0);
	gate_basis_ders.assign(
		static_cast<size_t>(nbh.count) * scalar_stride, 0.0);

	double sh_values[kMaxSHComponents];
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& radial_values = grad_mu_contract_vals_;
	const bool use_edge_cache = HasTwoLayerEdgePrimitiveCache(-1, false);
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const int mu_stride = radial_func_count;
	const int radial_coeff_base = C + 2 * C * C * K_;
	const int shared_type_offset = radial_coeff_base + R;
	const double center_type_coeff = regression_coeffs[shared_type_offset + type_central];

	for (int j = 0; j < nbh.count; ++j) {
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const int type_outer = nbh.types[j];
		const double* sh_values_use = sh_values;
		const double* radial_values_use = radial_values.data();
		if (use_edge_cache) {
			const size_t edge = TwoLayerEdgePrimitiveOffset(-1, j);
			sh_values_use =
				two_layer_edge_sh_values_cache_.data() + edge * sh_count;
			radial_values_use =
				two_layer_edge_mu_vals_cache_.data() + edge * mu_stride;
	} else {
		EvalRealSHValuesOnly(rvec, r, sh_l_max_, sh_values);

			if (use_lmp_table) {
				InterpolateSHLmpMuTable(radial_list,
				                         radial_der_list,
				                         inv_dr,
				                         C,
				                         type_central,
				                         type_outer,
				                         r,
				                         radial_func_count,
				                         radial_values.data(),
				                         nullptr);
			} else {
				for (int eval_block = 0; eval_block < static_cast<int>(radial_eval_to_scaling_block_.size()); ++eval_block) {
					const int scaling_block = radial_eval_to_scaling_block_[eval_block];
					const int basis_k = radial_eval_to_basis_k_[eval_block];
					p_RadialBasis->RB_CalcValsOnly(
						r,
						regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer],
						regression_coeffs[C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer],
						basis_k);
					for (int xi = 0; xi < R; ++xi)
						rb_vals[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
				}
				for (int mu = 0; mu < radial_func_count; ++mu) {
					const int radial_base = mu_to_radial_eval_block_[mu] * R;
					const int radial_offset = radial_coeff_base + mu * (R + C);
					double radial_value = 0.0;
					for (int xi = 0; xi < R; ++xi)
						radial_value += regression_coeffs[radial_offset + xi] * rb_vals[radial_base + xi];
					radial_values[mu] = radial_value;
				}
			}
		}

		const double outer_type_coeff = regression_coeffs[shared_type_offset + type_outer];
		const double pair_type_scale = center_type_coeff * outer_type_coeff;
		double* gate_moment_der = sh_gate_moment_ders_.data()
			+ static_cast<size_t>(j) * moment_stride;
		for (int i = 0; i < alpha_index_basic_count; ++i) {
			const int mu = basic_mu_cache_[i];
			const int sh_index = basic_sh_index_cache_[i];
			gate_moment_der[i] +=
				pair_type_scale * radial_values_use[mu] * sh_values_use[sh_index];
		}
	}

	if (UseSHProductRows()) {
		for (const SHProductRow& row : sh_product_rows_) {
			const int end = row.term_begin + row.term_count;
			for (int t = row.term_begin; t < end; ++t) {
				const SHProductRowTerm& term = sh_product_row_terms_[t];
				for (int j = 0; j < nbh.count; ++j) {
					double* gate_moment_der = sh_gate_moment_ders_.data()
						+ static_cast<size_t>(j) * moment_stride;
					gate_moment_der[row.target] += term.coeff * (
						gate_moment_der[term.left] * moment_vals[term.right]
						+ moment_vals[term.left] * gate_moment_der[term.right]);
				}
			}
		}
	} else {
		for (size_t p = 0; p < sh_products_.size(); ++p) {
			const SHProduct& product = sh_products_[p];
			for (int j = 0; j < nbh.count; ++j) {
				double* gate_moment_der = sh_gate_moment_ders_.data()
					+ static_cast<size_t>(j) * moment_stride;
				gate_moment_der[product.target] += product.coeff * (
					gate_moment_der[product.left] * moment_vals[product.right]
					+ moment_vals[product.left] * gate_moment_der[product.right]);
			}
		}
	}

	for (int j = 0; j < nbh.count; ++j) {
		const double* gate_moment_der = sh_gate_moment_ders_.data()
			+ static_cast<size_t>(j) * moment_stride;
		double* gate_basis_der = gate_basis_ders.data()
			+ static_cast<size_t>(j) * scalar_stride;
		for (int i = 0; i < alpha_scalar_moments; ++i)
			gate_basis_der[i] = gate_moment_der[alpha_moment_mapping[i]];
	}
}

void MLMTPR::AccumulateSHBasisGateDers(
	const Neighborhood& nbh,
	std::vector<double>& gate_linear_adjoints)
{
	const bool use_lmp_table = SHUsesPrecomputedLmpTable(p_RadialBasis);
	if (sh_l_max_ < 0 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,6].");
	if (alpha_scalar_moments == 0)
		return;
	if (gate_linear_adjoints.size() % static_cast<size_t>(alpha_scalar_moments) != 0)
		ERROR("SUS2-SH gate linear adjoint buffer has an incompatible size.");

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	if (type_central >= species_count)
		throw MlipException("Too few species count in the MTP potential!");

	const int moment_stride = alpha_moments_count;
	if (static_cast<int>(sh_gate_moment_ders_.size()) != moment_stride)
		sh_gate_moment_ders_.resize(moment_stride);
	const bool use_product_rows = !sh_product_rows_.empty();

	double sh_values[kMaxSHComponents];
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& radial_values = grad_mu_contract_vals_;
	const bool use_edge_cache = HasTwoLayerEdgePrimitiveCache(-1, false);
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const int mu_stride = radial_func_count;
	const int radial_coeff_base = C + 2 * C * C * K_;
	const int shared_type_offset = radial_coeff_base + R;
	const double center_type_coeff = regression_coeffs[shared_type_offset + type_central];

	for (int j = 0; j < nbh.count; ++j) {
		const int atom_index = nbh.inds[j];
		if (atom_index < 0
		    || static_cast<size_t>(atom_index + 1) * alpha_scalar_moments
		        > gate_linear_adjoints.size())
			ERROR("SUS2-SH two-layer gate linear component atom index is out of range");

		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const int type_outer = nbh.types[j];
		const double* sh_values_use = sh_values;
		const double* radial_values_use = radial_values.data();
		if (use_edge_cache) {
			const size_t edge = TwoLayerEdgePrimitiveOffset(-1, j);
			sh_values_use =
				two_layer_edge_sh_values_cache_.data() + edge * sh_count;
			radial_values_use =
				two_layer_edge_mu_vals_cache_.data() + edge * mu_stride;
	} else {
		EvalRealSHValuesOnly(rvec, r, sh_l_max_, sh_values);

			if (use_lmp_table) {
				InterpolateSHLmpMuTable(radial_list,
				                         radial_der_list,
				                         inv_dr,
				                         C,
				                         type_central,
				                         type_outer,
				                         r,
				                         radial_func_count,
				                         radial_values.data(),
				                         nullptr);
			} else {
				for (int eval_block = 0;
				     eval_block < static_cast<int>(radial_eval_to_scaling_block_.size());
				     ++eval_block) {
					const int scaling_block = radial_eval_to_scaling_block_[eval_block];
					const int basis_k = radial_eval_to_basis_k_[eval_block];
					p_RadialBasis->RB_CalcValsOnly(
						r,
						regression_coeffs[C + 2 * scaling_block * C * C
							+ C * type_central + type_outer],
						regression_coeffs[C + 2 * scaling_block * C * C + C * C
							+ C * type_central + type_outer],
						basis_k);
					for (int xi = 0; xi < R; ++xi)
						rb_vals[eval_block * R + xi] =
							p_RadialBasis->rb_vals[xi] * scaling;
				}
				for (int mu = 0; mu < radial_func_count; ++mu) {
					const int radial_base = mu_to_radial_eval_block_[mu] * R;
					const int radial_offset = radial_coeff_base + mu * (R + C);
					double radial_value = 0.0;
					for (int xi = 0; xi < R; ++xi)
						radial_value += regression_coeffs[radial_offset + xi]
							* rb_vals[radial_base + xi];
					radial_values[mu] = radial_value;
				}
			}
		}

		const double outer_type_coeff = regression_coeffs[shared_type_offset + type_outer];
		const double pair_type_scale = center_type_coeff * outer_type_coeff;
		if (!use_product_rows) {
			std::fill(sh_gate_moment_ders_.begin(),
			          sh_gate_moment_ders_.end(),
			          0.0);
		}
		for (int i = 0; i < alpha_index_basic_count; ++i) {
			const int mu = basic_mu_cache_[i];
			const int sh_index = basic_sh_index_cache_[i];
			sh_gate_moment_ders_[i] =
				pair_type_scale * radial_values_use[mu] * sh_values_use[sh_index];
		}

		if (use_product_rows) {
			for (const SHProductRow& row : sh_product_rows_) {
				double value = 0.0;
				const int end = row.term_begin + row.term_count;
				for (int t = row.term_begin; t < end; ++t) {
					const SHProductRowTerm& term = sh_product_row_terms_[t];
					value += term.coeff * (
						sh_gate_moment_ders_[term.left] * moment_vals[term.right]
						+ moment_vals[term.left] * sh_gate_moment_ders_[term.right]);
				}
				sh_gate_moment_ders_[row.target] = value;
			}
		} else {
			for (size_t p = 0; p < sh_products_.size(); ++p) {
				const SHProduct& product = sh_products_[p];
				sh_gate_moment_ders_[product.target] += product.coeff * (
					sh_gate_moment_ders_[product.left] * moment_vals[product.right]
					+ moment_vals[product.left] * sh_gate_moment_ders_[product.right]);
			}
		}

		double* gate_adjoints = gate_linear_adjoints.data()
			+ static_cast<size_t>(atom_index) * alpha_scalar_moments;
		for (int i = 0; i < alpha_scalar_moments; ++i)
			gate_adjoints[i] += sh_gate_moment_ders_[alpha_moment_mapping[i]];
	}
}

void MLMTPR::CalcTwoLayerGateScalarDirectionalDerivatives(
	const Neighborhood& nbh,
	const std::vector<Vector3>& direction_weights,
	std::vector<double>& gate_scalar_tangents,
	std::vector<double>* gate_moment_tangents,
	int cache_atom_index)
{
	if (SHUsesPrecomputedLmpTable(p_RadialBasis))
		ERROR("SUS2-SH does not use precomputed LAMMPS radial tables in training.");
	if (static_cast<int>(direction_weights.size()) != nbh.count)
		ERROR("SUS2-SH gate directional derivative size does not match neighborhood size.");
	if (sh_l_max_ < 0 || sh_l_max_ > kMaxSHL)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,6].");
	if (two_layer_gate_required_moments_.empty()
	    || two_layer_gate_required_moment_indices_.empty())
		BuildTwoLayerGateProductProgram();

	std::fill(moment_vals, moment_vals + alpha_moments_count, 0.0);
	grad_dloss_dsenders_.assign(alpha_moments_count, 0.0);
	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	if (type_central >= species_count)
		throw MlipException("Too few species count in the MTP potential!");

	const int radial_coeff_base = C + 2 * C * C * K_;
	const int shared_type_offset = radial_coeff_base + R;
	const double center_type_coeff = regression_coeffs[shared_type_offset + type_central];

	double sh_values[kMaxSHComponents];
	double sh_ders[3 * kMaxSHComponents];
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& rb_ders = basis_radial_ders_buffer_;
	std::vector<double>& radial_values = grad_mu_contract_vals_;
	std::vector<double>& radial_derivatives = grad_mu_contract_ders_;
	const bool use_edge_cache =
		HasTwoLayerEdgePrimitiveCache(cache_atom_index, true, false);
	const bool use_gate_radial = TwoLayerGateUsesSharedRadial();
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const int mu_stride = radial_func_count;
	const int gate_mu_stride = use_gate_radial
		? static_cast<int>(two_layer_gate_required_mu_indices_.size())
		: 0;

	for (int j = 0; j < nbh.count; ++j) {
		const bool has_direction = direction_weights[j].NormSq() != 0.0;
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const int type_outer = nbh.types[j];
		const double* sh_values_use = sh_values;
		const double* sh_ders_use = sh_ders;
		const double* radial_values_use = radial_values.data();
		const double* radial_derivatives_use = radial_derivatives.data();
		if (use_edge_cache) {
			const size_t edge = TwoLayerEdgePrimitiveOffset(cache_atom_index, j);
			sh_values_use =
				two_layer_edge_sh_values_cache_.data() + edge * sh_count;
			sh_ders_use =
				two_layer_edge_sh_ders_cache_.data() + edge * 3 * sh_count;
			radial_values_use =
				(use_gate_radial
					? two_layer_edge_gate_mu_vals_cache_.data()
					: two_layer_edge_mu_vals_cache_.data())
				+ edge * static_cast<size_t>(
					use_gate_radial ? gate_mu_stride : mu_stride);
			radial_derivatives_use =
				(use_gate_radial
					? two_layer_edge_gate_mu_ders_cache_.data()
					: two_layer_edge_mu_ders_cache_.data())
				+ edge * static_cast<size_t>(
					use_gate_radial ? gate_mu_stride : mu_stride);
		} else {
			if (has_direction)
				EvalRealSH(rvec, r, sh_l_max_, sh_values, sh_ders);
			else
				EvalRealSHValuesOnly(rvec, r, sh_l_max_, sh_values);

			for (int eval_block : two_layer_gate_required_radial_eval_blocks_) {
				const int scaling_block = radial_eval_to_scaling_block_[eval_block];
				const int basis_k = radial_eval_to_basis_k_[eval_block];
				const double sigma =
					regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer];
				const double shift =
					regression_coeffs[
						C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer];
				if (has_direction)
					p_RadialBasis->RB_Calc(r, sigma, shift, basis_k);
				else
					p_RadialBasis->RB_CalcValsOnly(r, sigma, shift, basis_k);
				for (int xi = 0; xi < R; ++xi) {
					rb_vals[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
					if (has_direction)
						rb_ders[eval_block * R + xi] =
							p_RadialBasis->rb_ders[xi] * scaling;
					}
				}
			for (int mu : two_layer_gate_required_mu_indices_) {
				const int radial_base = mu_to_radial_eval_block_[mu] * R;
				const int radial_offset = TwoLayerGateRadialCoeffIndex(mu, 0);
				double radial_value = 0.0;
				double radial_der = 0.0;
				for (int xi = 0; xi < R; ++xi) {
					const double coeff = regression_coeffs[radial_offset + xi];
					radial_value += coeff * rb_vals[radial_base + xi];
					if (has_direction)
						radial_der += coeff * rb_ders[radial_base + xi];
				}
				radial_values[mu] = radial_value;
				if (has_direction)
					radial_derivatives[mu] = radial_der;
			}
		}

		const double outer_type_coeff = regression_coeffs[shared_type_offset + type_outer];
		const double type_scale = center_type_coeff * outer_type_coeff;
		const double inv_r = has_direction ? 1.0 / r : 0.0;
		for (int bi = 0;
		     bi < static_cast<int>(two_layer_gate_required_basic_indices_.size());
		     ++bi) {
			const int basic_index = two_layer_gate_required_basic_indices_[bi];
			const int mu = basic_mu_cache_[basic_index];
			const int mu_index =
				(use_edge_cache && use_gate_radial)
					? two_layer_gate_required_basic_dense_mu_indices_[bi]
					: mu;
			const int sh_index = basic_sh_index_cache_[basic_index];
			const int sh_der_index = basic_sh_der_index_cache_[basic_index];
			const double radial_value = radial_values_use[mu_index];
			const double y = sh_values_use[sh_index];
			moment_vals[basic_index] += type_scale * radial_value * y;
			if (has_direction) {
				const double radial_der = radial_derivatives_use[mu_index];
				double directional_derivative = 0.0;
				for (int a = 0; a < 3; ++a) {
					const double dy = sh_ders_use[sh_der_index + a];
					directional_derivative += direction_weights[j][a]
						* (radial_der * rvec[a] * inv_r * y + radial_value * dy);
				}
				grad_dloss_dsenders_[basic_index] +=
					type_scale * directional_derivative;
			}
		}
	}

	for (int product_index : two_layer_gate_required_product_indices_) {
		const SHProduct& product = sh_products_[product_index];
		const double left_value = moment_vals[product.left];
		const double right_value = moment_vals[product.right];
		moment_vals[product.target] += product.coeff * left_value * right_value;
		grad_dloss_dsenders_[product.target] += product.coeff * (
			grad_dloss_dsenders_[product.left] * right_value
			+ left_value * grad_dloss_dsenders_[product.right]);
	}

	if (gate_moment_tangents != nullptr)
		gate_moment_tangents->assign(
			grad_dloss_dsenders_.begin(), grad_dloss_dsenders_.end());

	gate_scalar_tangents.assign(TwoLayerGateWeightCount(), 0.0);
	for (int q = 0; q < TwoLayerGateWeightCount(); ++q) {
		const int scalar_index = two_layer_gate_scalar_indices_[q];
		gate_scalar_tangents[q] =
			grad_dloss_dsenders_[alpha_moment_mapping[scalar_index]];
	}
}

void MLMTPR::CalcSHResidualSiteEnergyDers(const Neighborhood& nbh)
{
	if (!TwoLayerResidualEnabled())
		ERROR("CalcSHResidualSiteEnergyDers called for a non-residual model");
	if (static_cast<int>(two_layer_residual_e0_coeffs_.size()) != alpha_scalar_moments)
		two_layer_residual_e0_coeffs_.assign(alpha_scalar_moments, 0.0);

	const bool saved_residual = two_layer_residual_enabled_;
	const std::vector<double>* saved_gate_values = active_two_layer_gate_values_;
	std::vector<double>* saved_gate_adjoints = active_two_layer_gate_adjoints_;
	const std::vector<double> saved_linear_coeffs = linear_coeffs;
	std::vector<double> saved_e0_coeffs(alpha_scalar_moments, 0.0);
	for (int i = 0; i < alpha_scalar_moments; ++i)
		saved_e0_coeffs[i] = TwoLayerResidualE0Coeff(i);

	// E0: one-cutoff unmodulated SH readout, including the one-body term.
	two_layer_residual_enabled_ = false;
	active_two_layer_gate_values_ = nullptr;
	active_two_layer_gate_adjoints_ = nullptr;
	for (int i = 0; i < alpha_scalar_moments; ++i)
		linear_coeffs[species_count + i] = saved_e0_coeffs[i];
	for (int i = 0; i < species_count; ++i)
		linear_coeffs[i] = 1.0;
	CalcSHSiteEnergyDers(nbh);
	const double e0_energy =
		buff_site_energy_ + saved_linear_coeffs[nbh.my_type] - 1.0;
	const double e0_energy0 = buff_site_energy_0;
	std::vector<Vector3> e0_ders = buff_site_energy_ders_;

	// E1: direct-gated final SH readout. The one-body constant belongs to E0.
	linear_coeffs = saved_linear_coeffs;
	for (int i = 0; i < species_count; ++i)
		linear_coeffs[i] = 1.0;
	active_two_layer_gate_values_ = saved_gate_values;
	active_two_layer_gate_adjoints_ = saved_gate_adjoints;
	CalcSHSiteEnergyDers(nbh);
	const double e1_one_body = regression_coeffs[nbh.my_type] + 1.0;
	buff_site_energy_ += e0_energy - e1_one_body;
	buff_site_energy_0 += e0_energy0;
	for (int j = 0; j < nbh.count; ++j)
		buff_site_energy_ders_[j] += e0_ders[j];

	two_layer_residual_enabled_ = saved_residual;
	active_two_layer_gate_values_ = saved_gate_values;
	active_two_layer_gate_adjoints_ = saved_gate_adjoints;
	linear_coeffs = saved_linear_coeffs;
}

void MLMTPR::CalcSHSiteEnergyDers(const Neighborhood& nbh)
{
	if (TwoLayerResidualEnabled()) {
		CalcSHResidualSiteEnergyDers(nbh);
		return;
	}
	const bool use_lmp_table = SHUsesPrecomputedLmpTable(p_RadialBasis);
	const bool use_edge_der_cache =
		HasTwoLayerEdgePrimitiveCache(-1, true, false);
	const bool use_site_der_cache =
		!use_edge_der_cache && UseSHSiteDerivativeCache();
	if (use_site_der_cache)
		CalcSHMomentValuesWithSiteDerivativeCache(nbh);
	else
		CalcSHMomentValuesOnly(nbh);
	buff_site_energy_ = 0.0;
	buff_site_energy_0 = 0.0;
	buff_site_energy_ders_.resize(nbh.count);
	FillWithZero(buff_site_energy_ders_);

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	const int radial_coeff_base = C + 2 * C * C * K_;
	const int shared_type_offset = radial_coeff_base + R;
	const double center_linear = linear_coeffs[nbh.my_type];

	site_energy_ders_wrt_moments_.resize(alpha_moments_count);
	std::fill(site_energy_ders_wrt_moments_.begin(), site_energy_ders_wrt_moments_.end(), 0.0);
	buff_site_energy_ += regression_coeffs[nbh.my_type] + center_linear;
	for (int i = 0; i < alpha_scalar_moments; ++i) {
		const double moment_coeff = linear_coeffs[species_count + i] * linear_mults[i];
		const double coeff = center_linear * moment_coeff;
		const double scalar_value = moment_vals[alpha_moment_mapping[i]];
		buff_site_energy_ += coeff * scalar_value;
		buff_site_energy_0 += coeff * scalar_value;
		max_linear[i] = std::max(max_linear[i], std::abs(linear_coeffs[species_count + i] * scalar_value));
		site_energy_ders_wrt_moments_[alpha_moment_mapping[i]] += moment_coeff;
	}

	if (UseSHProductRows()) {
		BackpropSHProductRows(site_energy_ders_wrt_moments_);
	} else {
		for (int p = static_cast<int>(sh_products_.size()) - 1; p >= 0; --p) {
			const SHProduct& product = sh_products_[p];
			const double adj_target = site_energy_ders_wrt_moments_[product.target];
			if (adj_target == 0.0)
				continue;
			site_energy_ders_wrt_moments_[product.left] +=
				product.coeff * moment_vals[product.right] * adj_target;
			site_energy_ders_wrt_moments_[product.right] +=
				product.coeff * moment_vals[product.left] * adj_target;
		}
	}

	double sh_values[kMaxSHComponents];
	double sh_ders[3 * kMaxSHComponents];
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& rb_ders = basis_radial_ders_buffer_;
	std::vector<double>& radial_values = grad_mu_contract_vals_;
	std::vector<double>& radial_derivatives = grad_mu_contract_ders_;
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const int mu_stride = radial_func_count;

	for (int j = 0; j < nbh.count; ++j) {
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const double inv_r = 1.0 / r;
		const int type_outer = nbh.types[j];
		const double* sh_values_use = sh_values;
		const double* sh_ders_use = sh_ders;
		const double* radial_values_use = radial_values.data();
		const double* radial_derivatives_use = radial_derivatives.data();
		if (use_edge_der_cache) {
			const size_t edge = TwoLayerEdgePrimitiveOffset(-1, j);
			sh_values_use =
				two_layer_edge_sh_values_cache_.data() + edge * sh_count;
			sh_ders_use =
				two_layer_edge_sh_ders_cache_.data() + edge * 3 * sh_count;
			radial_values_use =
				two_layer_edge_mu_vals_cache_.data() + edge * mu_stride;
			radial_derivatives_use =
				two_layer_edge_mu_ders_cache_.data() + edge * mu_stride;
		} else if (use_site_der_cache) {
			sh_values_use = grad_neighbor_sh_values_cache_.data() + static_cast<size_t>(j) * sh_count;
			sh_ders_use = grad_neighbor_sh_ders_cache_.data() + static_cast<size_t>(j) * 3 * sh_count;
			radial_values_use = grad_neighbor_mu_contract_vals_cache_.data() + static_cast<size_t>(j) * mu_stride;
			radial_derivatives_use = grad_neighbor_mu_contract_ders_cache_.data() + static_cast<size_t>(j) * mu_stride;
	} else {
		EvalRealSH(rvec, r, sh_l_max_, sh_values, sh_ders);

			if (use_lmp_table) {
				InterpolateSHLmpMuTable(radial_list,
				                         radial_der_list,
				                         inv_dr,
				                         C,
				                         type_central,
				                         type_outer,
				                         r,
				                         radial_func_count,
				                         radial_values.data(),
				                         radial_derivatives.data());
			} else {
				for (int eval_block = 0; eval_block < static_cast<int>(radial_eval_to_scaling_block_.size()); ++eval_block) {
					const int scaling_block = radial_eval_to_scaling_block_[eval_block];
					const int basis_k = radial_eval_to_basis_k_[eval_block];
					p_RadialBasis->RB_Calc(
						r,
						regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer],
						regression_coeffs[C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer],
						basis_k);
					for (int xi = 0; xi < R; ++xi) {
						rb_vals[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
						rb_ders[eval_block * R + xi] = p_RadialBasis->rb_ders[xi] * scaling;
					}
				}
				for (int mu = 0; mu < radial_func_count; ++mu) {
					const int radial_base = mu_to_radial_eval_block_[mu] * R;
					const int radial_offset = radial_coeff_base + mu * (R + C);
					double radial_value = 0.0;
					double radial_der = 0.0;
					for (int xi = 0; xi < R; ++xi) {
						const double coeff = regression_coeffs[radial_offset + xi];
						radial_value += coeff * rb_vals[radial_base + xi];
						radial_der += coeff * rb_ders[radial_base + xi];
					}
					radial_values[mu] = radial_value;
					radial_derivatives[mu] = radial_der;
				}
			}
		}

				const double center_type_coeff = regression_coeffs[shared_type_offset + type_central];
				const double outer_type_coeff = regression_coeffs[shared_type_offset + type_outer];
				const double pair_type_scale = center_type_coeff * outer_type_coeff;
				const double gate_residual = TwoLayerGateNeighborResidual(nbh, j);
				PrepareTwoLayerGateNeighborMuBuffers(
					type_outer, center_type_coeff, outer_type_coeff, gate_residual);
				const double* gate_additive_by_mu =
					two_layer_gate_additive_mu_buffer_.data();
				const double* gate_type_scale_by_mu =
					two_layer_gate_type_scale_mu_buffer_.data();
				double* gate_scaled_radial_by_mu =
					two_layer_gate_scaled_mu_vals_buffer_.data();
				double* gate_scaled_der_by_mu =
					two_layer_gate_scaled_mu_ders_buffer_.data();
				for (int mu = 0; mu < radial_func_count; ++mu) {
					const double type_scale = gate_type_scale_by_mu[mu];
					gate_scaled_radial_by_mu[mu] = type_scale * radial_values_use[mu];
					gate_scaled_der_by_mu[mu] = type_scale * radial_derivatives_use[mu];
				}
				double gate_adjoint = 0.0;

				for (int i = 0; i < alpha_index_basic_count; ++i) {
			const double adj = center_linear * site_energy_ders_wrt_moments_[i];
			if (adj == 0.0)
				continue;
			const int mu = basic_mu_cache_[i];
					const int sh_index = basic_sh_index_cache_[i];
					const int sh_der_index = basic_sh_der_index_cache_[i];
					const double radial_value = radial_values_use[mu];
					const double radial_der = radial_derivatives_use[mu];
					const double scaled_radial_value = gate_scaled_radial_by_mu[mu];
					const double scaled_radial_der = gate_scaled_der_by_mu[mu];
					const double y = sh_values_use[sh_index];
					const double additive_coeff = gate_additive_by_mu[mu];
					gate_adjoint += adj * center_type_coeff * additive_coeff * radial_value * y;
					for (int a = 0; a < 3; ++a) {
					const double dy = sh_ders_use[sh_der_index + a];
					buff_site_energy_ders_[j][a] +=
						adj * (scaled_radial_der * rvec[a] * inv_r * y
						       + scaled_radial_value * dy);
				}
			}
		AddTwoLayerGateAdjoint(nbh, j, gate_adjoint);
	}
}

void MLMTPR::AccumulateSHGateTangentGrad(const Neighborhood& nbh,
                                         std::vector<double>& out_grad_accumulator,
                                         const std::vector<double>& neighbor_gate_tangent,
                                         int cache_atom_index)
{
	if (SHUsesPrecomputedLmpTable(p_RadialBasis))
		ERROR("SUS2-SH coefficient gradients require direct radial basis evaluation.");
	if (static_cast<int>(neighbor_gate_tangent.size()) != nbh.count)
		ERROR("SUS2-SH gate tangent size does not match neighborhood size.");

	bool has_tangent = false;
	for (double value : neighbor_gate_tangent)
		if (value != 0.0) {
			has_tangent = true;
			break;
		}
	if (!has_tangent)
		return;

	out_grad_accumulator.resize(CoeffCount());

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	const int radial_coeff_base = C + 2 * C * C * K_;
	const int shared_type_offset = radial_coeff_base + R;
	const int coeff_count = LinearCoeffOffset();
	const double center_linear = linear_coeffs[type_central];
	const int eval_block_count = static_cast<int>(radial_eval_to_scaling_block_.size());
	const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
	const int radial_val_stride = eval_block_count * R;

	bool loaded_final_cache = false;
	if (cache_atom_index >= 0 && alpha_moments_count > 0) {
		const size_t offset =
			static_cast<size_t>(cache_atom_index) * alpha_moments_count;
		if (offset + alpha_moments_count <=
		    two_layer_final_moment_values_cache_.size()
		    && offset + alpha_moments_count <=
		    two_layer_final_moment_ders_cache_.size()) {
			std::copy(two_layer_final_moment_values_cache_.data() + offset,
			          two_layer_final_moment_values_cache_.data() + offset
			              + alpha_moments_count,
			          moment_vals);
			site_energy_ders_wrt_moments_.assign(
				two_layer_final_moment_ders_cache_.data() + offset,
				two_layer_final_moment_ders_cache_.data() + offset
				    + alpha_moments_count);
			loaded_final_cache = true;
		}
	}
	if (!loaded_final_cache) {
		CalcSHMomentValuesOnly(nbh);
		site_energy_ders_wrt_moments_.resize(alpha_moments_count);
		std::fill(site_energy_ders_wrt_moments_.begin(),
		          site_energy_ders_wrt_moments_.end(), 0.0);
		for (int i = 0; i < alpha_scalar_moments; ++i) {
			const int node = alpha_moment_mapping[i];
			const double moment_coeff = linear_coeffs[species_count + i] * linear_mults[i];
			site_energy_ders_wrt_moments_[node] += moment_coeff;
		}
		if (UseSHProductRows()) {
			BackpropSHProductRows(site_energy_ders_wrt_moments_);
		} else {
			for (int p = static_cast<int>(sh_products_.size()) - 1; p >= 0; --p) {
				const SHProduct& product = sh_products_[p];
				const double adj_target = site_energy_ders_wrt_moments_[product.target];
				if (adj_target == 0.0)
					continue;
				site_energy_ders_wrt_moments_[product.left] +=
					product.coeff * moment_vals[product.right] * adj_target;
				site_energy_ders_wrt_moments_[product.right] +=
					product.coeff * moment_vals[product.left] * adj_target;
			}
		}
	}
	grad_dloss_dsenders_.assign(alpha_moments_count, 0.0);
	grad_dloss_dmom_.assign(alpha_moments_count, 0.0);

	const int radial_der_stride = radial_val_stride * 5;
	const size_t neighbor_count = static_cast<size_t>(nbh.count);
	std::vector<double>& cached_sh_values_all = grad_neighbor_sh_values_cache_;
	std::vector<double>& cached_rb_vals_all = grad_neighbor_radial_vals_cache_;
	std::vector<double>& cached_radial_values_all = grad_neighbor_mu_contract_vals_cache_;
	std::vector<double>& cached_radial_s_all = grad_neighbor_mu_contract_ders_s_cache_;
	std::vector<double>& cached_radial_ss_all = grad_neighbor_mu_contract_ders_ss_cache_;
	std::vector<double>& rb_ders = radial_ders_buffer_;
	const double center_type_coeff = regression_coeffs[shared_type_offset + type_central];
	const bool use_edge_cache =
		HasTwoLayerEdgePrimitiveCache(cache_atom_index, true, true);
	if (!use_edge_cache) {
		cached_sh_values_all.resize(neighbor_count * sh_count);
		cached_rb_vals_all.resize(neighbor_count * radial_val_stride);
		cached_radial_values_all.resize(neighbor_count * radial_func_count);
		cached_radial_s_all.resize(neighbor_count * radial_func_count);
		cached_radial_ss_all.resize(neighbor_count * radial_func_count);
		if (static_cast<int>(rb_ders.size()) < radial_der_stride)
			rb_ders.resize(radial_der_stride);
	}

		for (int j = 0; j < nbh.count; ++j) {
			const double tangent = neighbor_gate_tangent[j];
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const int type_outer = nbh.types[j];
		const double* cached_sh_values = nullptr;
		const double* cached_radial_values = nullptr;
		if (use_edge_cache) {
			const size_t edge = TwoLayerEdgePrimitiveOffset(cache_atom_index, j);
			cached_sh_values =
				two_layer_edge_sh_values_cache_.data() + edge * sh_count;
			cached_radial_values =
				two_layer_edge_mu_vals_cache_.data() + edge * radial_func_count;
		} else {
			double* cached_sh_values_write =
				cached_sh_values_all.data() + static_cast<size_t>(j) * sh_count;
			double* cached_rb_vals =
				cached_rb_vals_all.data() + static_cast<size_t>(j) * radial_val_stride;
			double* cached_radial_values_write =
				cached_radial_values_all.data()
				+ static_cast<size_t>(j) * radial_func_count;
			double* cached_radial_s =
				cached_radial_s_all.data()
				+ static_cast<size_t>(j) * radial_func_count;
			double* cached_radial_ss =
				cached_radial_ss_all.data()
				+ static_cast<size_t>(j) * radial_func_count;
			cached_sh_values = cached_sh_values_write;
			cached_radial_values = cached_radial_values_write;
			EvalRealSHValuesOnly(rvec, r, sh_l_max_, cached_sh_values_write);
			for (int eval_block = 0; eval_block < eval_block_count; ++eval_block) {
				const int scaling_block = radial_eval_to_scaling_block_[eval_block];
				const int basis_k = radial_eval_to_basis_k_[eval_block];
				p_RadialBasis->RB_Calc(
					r,
					regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer],
					regression_coeffs[C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer],
					basis_k);
				for (int xi = 0; xi < R; ++xi)
					cached_rb_vals[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
				for (int xi = 0; xi < R * 5; ++xi)
					rb_ders[eval_block * R * 5 + xi] = p_RadialBasis->rb_ders[xi] * scaling;
			}
			for (int mu = 0; mu < radial_func_count; ++mu) {
				const int radial_base = mu_to_radial_eval_block_[mu] * R;
				const int deriv_base = 5 * radial_base;
				const int radial_offset = radial_coeff_base + mu * (R + C);
				double radial_value = 0.0;
				double radial_s = 0.0;
					double radial_ss = 0.0;
					for (int xi = 0; xi < R; ++xi) {
						const double coeff = regression_coeffs[radial_offset + xi];
						radial_value += coeff * cached_rb_vals[radial_base + xi];
						radial_s += coeff * rb_ders[deriv_base + xi + R];
						radial_ss += coeff * rb_ders[deriv_base + xi + 3 * R];
					}
					cached_radial_values_write[mu] = radial_value;
					cached_radial_s[mu] = radial_s;
					cached_radial_ss[mu] = radial_ss;
				}
		}
				if (tangent == 0.0)
					continue;
				const double outer_type_coeff = regression_coeffs[shared_type_offset + type_outer];
			PrepareTwoLayerGateNeighborMuBuffers(
				type_outer, center_type_coeff, outer_type_coeff, 0.0);
			const double* gate_additive_by_mu =
				two_layer_gate_additive_mu_buffer_.data();
			double* gate_scaled_radial_by_mu =
				two_layer_gate_scaled_mu_vals_buffer_.data();
			const double tangent_center = center_type_coeff * tangent;
			for (int mu = 0; mu < radial_func_count; ++mu)
				gate_scaled_radial_by_mu[mu] =
					tangent_center * gate_additive_by_mu[mu] * cached_radial_values[mu];

					for (int i = 0; i < alpha_index_basic_count; ++i) {
						const int mu = basic_mu_cache_[i];
						const int sh_index = basic_sh_index_cache_[i];
						grad_dloss_dsenders_[i] +=
							gate_scaled_radial_by_mu[mu] * cached_sh_values[sh_index];
					}
		}

	const bool use_product_rows = UseSHProductRows();
	const bool use_product_hvt_reverse = !use_product_rows && UseSHProductHVTReverse();
	if (use_product_rows) {
		AccumulateSHProductRowsForward(grad_dloss_dsenders_, grad_dloss_dsenders_);
		for (const SHProductRow& row : sh_product_rows_) {
			const double adj_target = site_energy_ders_wrt_moments_[row.target];
			if (adj_target == 0.0)
				continue;
			const int end = row.term_begin + row.term_count;
			for (int t = row.term_begin; t < end; ++t) {
				const SHProductRowTerm& term = sh_product_row_terms_[t];
				grad_dloss_dmom_[term.left] +=
					grad_dloss_dsenders_[term.right] * adj_target * term.coeff;
				grad_dloss_dmom_[term.right] +=
					grad_dloss_dsenders_[term.left] * adj_target * term.coeff;
			}
		}
		BackpropSHProductRows(grad_dloss_dmom_);
	} else {
		for (size_t p = 0; p < sh_products_.size(); ++p) {
			const SHProduct& product = sh_products_[p];
			grad_dloss_dsenders_[product.target] += product.coeff * (
				grad_dloss_dsenders_[product.left] * moment_vals[product.right]
				+ moment_vals[product.left] * grad_dloss_dsenders_[product.right]);
		}
		if (use_product_hvt_reverse) {
			for (int p = static_cast<int>(sh_products_.size()) - 1; p >= 0; --p) {
				const SHProduct& product = sh_products_[p];
				const double adj_target = site_energy_ders_wrt_moments_[product.target];
				const double hvt_target = grad_dloss_dmom_[product.target];
				if (adj_target == 0.0 && hvt_target == 0.0)
					continue;
				grad_dloss_dmom_[product.left] += product.coeff * (
					moment_vals[product.right] * hvt_target
					+ grad_dloss_dsenders_[product.right] * adj_target);
				grad_dloss_dmom_[product.right] += product.coeff * (
					moment_vals[product.left] * hvt_target
					+ grad_dloss_dsenders_[product.left] * adj_target);
			}
		} else {
			for (size_t p = 0; p < sh_products_.size(); ++p) {
				const SHProduct& product = sh_products_[p];
				const double adj_target = site_energy_ders_wrt_moments_[product.target];
				if (adj_target == 0.0)
					continue;
				grad_dloss_dmom_[product.left] +=
					grad_dloss_dsenders_[product.right] * adj_target * product.coeff;
				grad_dloss_dmom_[product.right] +=
					grad_dloss_dsenders_[product.left] * adj_target * product.coeff;
			}
			for (int p = static_cast<int>(sh_products_.size()) - 1; p >= 0; --p) {
				const SHProduct& product = sh_products_[p];
				const double adj_target = grad_dloss_dmom_[product.target];
				if (adj_target == 0.0)
					continue;
				grad_dloss_dmom_[product.left] +=
					product.coeff * moment_vals[product.right] * adj_target;
				grad_dloss_dmom_[product.right] +=
					product.coeff * moment_vals[product.left] * adj_target;
			}
		}
	}

	double center_linear_grad = 0.0;
	for (int i = 0; i < alpha_scalar_moments; ++i) {
		const int node = alpha_moment_mapping[i];
		const double moment_coeff = linear_coeffs[species_count + i] * linear_mults[i];
		center_linear_grad += moment_coeff * grad_dloss_dsenders_[node];
		out_grad_accumulator[coeff_count + species_count + i] +=
			center_linear * linear_mults[i] * grad_dloss_dsenders_[node];
	}
	out_grad_accumulator[coeff_count + type_central] += center_linear_grad;

	if (static_cast<int>(grad_radial_coeff_value_accum_.size()) != radial_func_count)
		grad_radial_coeff_value_accum_.resize(radial_func_count);
	if (static_cast<int>(grad_radial_coeff_coord_accum_.size()) != radial_func_count)
		grad_radial_coeff_coord_accum_.resize(radial_func_count);

	for (int j = 0; j < nbh.count; ++j) {
		const int type_outer = nbh.types[j];
		const double* cached_sh_values = nullptr;
		const double* cached_rb_vals = nullptr;
		const double* cached_radial_values = nullptr;
		const double* cached_radial_s = nullptr;
		const double* cached_radial_ss = nullptr;
		if (use_edge_cache) {
			const size_t edge = TwoLayerEdgePrimitiveOffset(cache_atom_index, j);
			cached_sh_values =
				two_layer_edge_sh_values_cache_.data() + edge * sh_count;
			cached_rb_vals =
				two_layer_edge_radial_vals_cache_.data() + edge * radial_val_stride;
			cached_radial_values =
				two_layer_edge_mu_vals_cache_.data() + edge * radial_func_count;
			cached_radial_s =
				two_layer_edge_mu_ders_s_cache_.data() + edge * radial_func_count;
			cached_radial_ss =
				two_layer_edge_mu_ders_ss_cache_.data() + edge * radial_func_count;
		} else {
			cached_sh_values =
				cached_sh_values_all.data() + static_cast<size_t>(j) * sh_count;
			cached_rb_vals =
				cached_rb_vals_all.data() + static_cast<size_t>(j) * radial_val_stride;
			cached_radial_values =
				cached_radial_values_all.data()
				+ static_cast<size_t>(j) * radial_func_count;
			cached_radial_s =
				cached_radial_s_all.data()
				+ static_cast<size_t>(j) * radial_func_count;
			cached_radial_ss =
				cached_radial_ss_all.data()
				+ static_cast<size_t>(j) * radial_func_count;
		}

				const double outer_type_coeff = regression_coeffs[shared_type_offset + type_outer];
				const double gate_residual = TwoLayerGateNeighborResidual(nbh, j);
				const double pair_type_scale = center_type_coeff * outer_type_coeff;
				const double tangent = neighbor_gate_tangent[j];
				PrepareTwoLayerGateNeighborMuBuffers(
					type_outer, center_type_coeff, outer_type_coeff, gate_residual);
				const double* gate_additive_by_mu =
					two_layer_gate_additive_mu_buffer_.data();
				const double* gate_type_scale_by_mu =
					two_layer_gate_type_scale_mu_buffer_.data();
				double gate_adjoint = 0.0;
		double* radial_coeff_value_accum = grad_radial_coeff_value_accum_.data();
		double* radial_coeff_direct_accum = grad_radial_coeff_coord_accum_.data();
		std::fill(radial_coeff_value_accum, radial_coeff_value_accum + radial_func_count, 0.0);
		std::fill(radial_coeff_direct_accum, radial_coeff_direct_accum + radial_func_count, 0.0);

		for (int i = 0; i < alpha_index_basic_count; ++i) {
			const int mu = basic_mu_cache_[i];
			const int sh_index = basic_sh_index_cache_[i];
			const double y = cached_sh_values[sh_index];
			const double value_weight = center_linear * grad_dloss_dmom_[i];
			const double direct_weight =
				center_linear * site_energy_ders_wrt_moments_[i] * tangent;
			if (value_weight == 0.0 && direct_weight == 0.0)
				continue;
			if (value_weight != 0.0)
				radial_coeff_value_accum[mu] += value_weight * y;
			if (direct_weight != 0.0)
				radial_coeff_direct_accum[mu] += direct_weight * y;
		}
		for (int mu = 0; mu < radial_func_count; ++mu) {
			const double value_accum = radial_coeff_value_accum[mu];
			const double direct_accum = radial_coeff_direct_accum[mu];
			if (value_accum == 0.0 && direct_accum == 0.0)
				continue;
			const int radial_base = mu_to_radial_eval_block_[mu] * R;
			const int scaling_block =
				radial_eval_to_scaling_block_[mu_to_radial_eval_block_[mu]];
			const int radial_offset = radial_coeff_base + mu * (R + C);
					const int sigma_coeff_offset =
						C + 2 * C * C * scaling_block + type_central * C + type_outer;
					const double dot_val = cached_radial_values[mu];
					const double additive_coeff = gate_additive_by_mu[mu];
					const double type_scale = gate_type_scale_by_mu[mu];
				if (value_accum != 0.0) {
					out_grad_accumulator[shared_type_offset + type_central] +=
						(outer_type_coeff + additive_coeff * gate_residual)
						* dot_val * value_accum;
					out_grad_accumulator[shared_type_offset + type_outer] +=
						center_type_coeff * dot_val * value_accum;
				}
				if (direct_accum != 0.0)
					out_grad_accumulator[shared_type_offset + type_central] +=
						additive_coeff * dot_val * direct_accum;
				if (value_accum != 0.0)
					gate_adjoint += center_type_coeff * additive_coeff * dot_val * value_accum;
				const double additive_accum =
					center_type_coeff * (gate_residual * value_accum + direct_accum);
				if (two_layer_gate_enabled_ && additive_accum != 0.0)
					out_grad_accumulator[TwoLayerGateAdditiveCoeffIndex(type_outer, mu)] +=
						dot_val * additive_accum;
				const double sigma_accum =
					type_scale * value_accum + center_type_coeff * additive_coeff * direct_accum;
				if (sigma_accum != 0.0) {
				out_grad_accumulator[sigma_coeff_offset] +=
					cached_radial_s[mu] * sigma_accum;
				out_grad_accumulator[sigma_coeff_offset + C * C] +=
					cached_radial_ss[mu] * sigma_accum;
			}
			for (int xi = 0; xi < R; ++xi) {
					const double rb_val = cached_rb_vals[radial_base + xi];
					if (value_accum != 0.0)
						out_grad_accumulator[radial_offset + xi] +=
							type_scale * rb_val * value_accum;
					if (direct_accum != 0.0)
						out_grad_accumulator[radial_offset + xi] +=
							center_type_coeff * additive_coeff * rb_val * direct_accum;
				}
		}
		AddTwoLayerGateAdjoint(nbh, j, gate_adjoint);
	}
}

void MLMTPR::AccumulateSHCombinationGrad(const Neighborhood& nbh,
                                         std::vector<double>& out_grad_accumulator,
                                         const double se_weight,
                                         const Vector3* se_ders_weights)
{
	if (SHUsesPrecomputedLmpTable(p_RadialBasis))
		ERROR("SUS2-SH coefficient gradients require direct radial basis evaluation.");

	if (TwoLayerResidualEnabled()) {
		out_grad_accumulator.resize(CoeffCount());
		const bool need_e0_grad = (two_layer_residual_eval_stage_ != 2);
		const bool need_e1_grad = (two_layer_residual_eval_stage_ != 1);
		std::vector<double> e0_grad;
		std::vector<double> e1_grad;
		const bool saved_residual = two_layer_residual_enabled_;
		const std::vector<double>* saved_gate_values = active_two_layer_gate_values_;
		std::vector<double>* saved_gate_adjoints = active_two_layer_gate_adjoints_;
		const bool saved_skip_outer_param_grad =
			two_layer_residual_skip_outer_param_grad_;
		const std::vector<double> saved_linear_coeffs = linear_coeffs;
		const int residual_linear_offset = LinearCoeffOffset();
		const int residual_e0_offset = TwoLayerResidualE0CoeffOffset();
		const int component_linear_offset =
			TwoLayerGateWeightOffset() + TwoLayerGateWeightCount();
		const int type_central = nbh.my_type;
		std::vector<double> saved_e0_coeffs(alpha_scalar_moments, 0.0);
		for (int i = 0; i < alpha_scalar_moments; ++i)
			saved_e0_coeffs[i] = TwoLayerResidualE0Coeff(i);

		if (need_e0_grad) {
			e0_grad.assign(CoeffCount(), 0.0);
			two_layer_residual_enabled_ = false;
			active_two_layer_gate_values_ = nullptr;
			active_two_layer_gate_adjoints_ = nullptr;
			for (int i = 0; i < alpha_scalar_moments; ++i)
				linear_coeffs[species_count + i] = saved_e0_coeffs[i];
			for (int i = 0; i < species_count; ++i)
				linear_coeffs[i] = 1.0;
			AccumulateSHCombinationGrad(nbh, e0_grad, se_weight, se_ders_weights);
			for (int i = 0; i < component_linear_offset; ++i)
				out_grad_accumulator[i] += e0_grad[i];
			for (int i = 0; i < species_count; ++i)
				out_grad_accumulator[residual_linear_offset + i] +=
					(i == type_central ? se_weight : 0.0);
			for (int i = 0; i < alpha_scalar_moments; ++i)
				out_grad_accumulator[residual_e0_offset + i] +=
					e0_grad[component_linear_offset + species_count + i];
		}

		if (need_e1_grad) {
			e1_grad.assign(CoeffCount(), 0.0);
			two_layer_residual_enabled_ = false;
			linear_coeffs = saved_linear_coeffs;
			for (int i = 0; i < species_count; ++i)
				linear_coeffs[i] = 1.0;
			active_two_layer_gate_values_ = saved_gate_values;
			active_two_layer_gate_adjoints_ = saved_gate_adjoints;
			two_layer_residual_skip_outer_param_grad_ =
				(two_layer_residual_eval_stage_ == 2);
			AccumulateSHCombinationGrad(nbh, e1_grad, se_weight, se_ders_weights);
			two_layer_residual_skip_outer_param_grad_ =
				saved_skip_outer_param_grad;
			if (shift_)
				e1_grad[type_central] -= se_weight;
			e1_grad[component_linear_offset + type_central] -= se_weight;
			for (int i = 0; i < component_linear_offset; ++i)
				out_grad_accumulator[i] += e1_grad[i];
			for (int i = species_count; i < LinearCoeffCount(); ++i)
				out_grad_accumulator[residual_linear_offset + i] +=
					e1_grad[component_linear_offset + i];
		}

		two_layer_residual_enabled_ = saved_residual;
		active_two_layer_gate_values_ = saved_gate_values;
		active_two_layer_gate_adjoints_ = saved_gate_adjoints;
		two_layer_residual_skip_outer_param_grad_ =
			saved_skip_outer_param_grad;
		linear_coeffs = saved_linear_coeffs;
		return;
	}

	const bool skip_site_der_output = UseSHAccumSkipSiteDers();

	{
		buff_site_energy_ = 0.0;
		buff_site_energy_0 = 0.0;
		buff_site_energy_ders_.resize(nbh.count);
		if (!skip_site_der_output)
			FillWithZero(buff_site_energy_ders_);
		out_grad_accumulator.resize(CoeffCount());
		double* grad_out = out_grad_accumulator.data();

		const int C = species_count;
		const int R = p_RadialBasis->rb_size;
		const int K = radial_func_count;
		const int coeff_count = LinearCoeffOffset();
		const int type_central = nbh.my_type;
		const int radial_coeff_base = C + 2 * C * C * K_;
		const int shared_type_offset = radial_coeff_base + R;
		const double center_linear = linear_coeffs[type_central];

		site_energy_ders_wrt_moments_.resize(alpha_moments_count);
		std::fill(site_energy_ders_wrt_moments_.begin(), site_energy_ders_wrt_moments_.end(), 0.0);
		if (static_cast<int>(grad_dloss_dsenders_.size()) != alpha_moments_count)
			grad_dloss_dsenders_.resize(alpha_moments_count);
		if (static_cast<int>(grad_dloss_dmom_.size()) != alpha_moments_count)
			grad_dloss_dmom_.resize(alpha_moments_count);
		std::fill(grad_dloss_dsenders_.begin(), grad_dloss_dsenders_.end(), 0.0);
		std::fill(grad_dloss_dmom_.begin(), grad_dloss_dmom_.end(), 0.0);
		const bool use_product_rows = UseSHProductRows();
		const bool use_product_hvt_reverse = !use_product_rows && UseSHProductHVTReverse();
		const bool profile_accum = SHAccumProfileEnabledOnRank0();
		const double profile_start = profile_accum ? SHAccumProfileNow() : 0.0;
		if (se_ders_weights != nullptr)
			CalcSHMomentValuesWithGradientCache(nbh);
		else
			CalcSHMomentValuesOnly(nbh);
		const double profile_after_moment = profile_accum ? SHAccumProfileNow() : 0.0;

			double scalar_sum = 0.0;
			for (int i = 0; i < alpha_scalar_moments; ++i) {
				const int node = alpha_moment_mapping[i];
				const double moment_coeff = linear_coeffs[species_count + i] * linear_mults[i];
				scalar_sum += moment_coeff * moment_vals[node];
				site_energy_ders_wrt_moments_[node] += moment_coeff;
			}

			if (use_product_rows) {
				BackpropSHProductRows(site_energy_ders_wrt_moments_);
			} else {
				for (int p = static_cast<int>(sh_products_.size()) - 1; p >= 0; --p) {
					const SHProduct& product = sh_products_[p];
					const double adj_target = site_energy_ders_wrt_moments_[product.target];
					if (adj_target == 0.0)
						continue;
					site_energy_ders_wrt_moments_[product.left] +=
						product.coeff * moment_vals[product.right] * adj_target;
					site_energy_ders_wrt_moments_[product.right] +=
						product.coeff * moment_vals[product.left] * adj_target;
				}
			}
		const double profile_after_energy_backprop = profile_accum ? SHAccumProfileNow() : 0.0;

			double sh_values[kMaxSHComponents];
			double sh_ders[3 * kMaxSHComponents];
		std::vector<double>& rb_vals = radial_vals_buffer_;
		std::vector<double>& rb_ders = radial_ders_buffer_;
		std::vector<double>& radial_values = grad_mu_contract_vals_;
		std::vector<double>& radial_derivatives = grad_mu_contract_ders_;
			std::vector<double>& radial_s_derivatives = grad_mu_contract_ders_s_;
			std::vector<double>& radial_ss_derivatives = grad_mu_contract_ders_ss_;
			std::vector<double>& radial_coord_s_derivatives = grad_mu_contract_coord_ders_s_;
			std::vector<double>& radial_coord_ss_derivatives = grad_mu_contract_coord_ders_ss_;
			const int eval_block_count = static_cast<int>(radial_eval_to_scaling_block_.size());
			const int sh_count = (sh_l_max_ + 1) * (sh_l_max_ + 1);
			const int radial_val_stride = eval_block_count * R;
				const int radial_der_stride = radial_val_stride * 5;
				const int mu_stride = radial_func_count;
				const bool use_edge_cache_for_gradient =
					se_ders_weights != nullptr
					&& HasTwoLayerEdgePrimitiveCache(-1, true, true);
				double profile_after_force_cache = profile_after_energy_backprop;
				double profile_after_tangent_forward = profile_after_energy_backprop;
				double profile_after_mixed_seed = profile_after_energy_backprop;

			if (se_ders_weights != nullptr) {
				for (int j = 0; j < nbh.count; ++j) {
					const Vector3& rvec = nbh.vecs[j];
					const Vector3& se_weight_vec = se_ders_weights[j];
					const double r = nbh.dists[j];
					const double inv_r = 1.0 / r;
						const double wr = (se_weight_vec[0] * rvec[0]
							+ se_weight_vec[1] * rvec[1]
							+ se_weight_vec[2] * rvec[2]) * inv_r;
						const int type_outer = nbh.types[j];
						const double* cached_sh_values = nullptr;
						const double* cached_sh_ders = nullptr;
						const double* cached_radial_values = nullptr;
						const double* cached_radial_derivatives = nullptr;
						if (use_edge_cache_for_gradient) {
							const size_t edge = TwoLayerEdgePrimitiveOffset(-1, j);
							cached_sh_values =
								two_layer_edge_sh_values_cache_.data() + edge * sh_count;
							cached_sh_ders =
								two_layer_edge_sh_ders_cache_.data() + edge * 3 * sh_count;
							cached_radial_values =
								two_layer_edge_mu_vals_cache_.data() + edge * mu_stride;
							cached_radial_derivatives =
								two_layer_edge_mu_ders_cache_.data() + edge * mu_stride;
						} else {
							cached_sh_values =
								grad_neighbor_sh_values_cache_.data()
								+ static_cast<size_t>(j) * sh_count;
							cached_sh_ders =
								grad_neighbor_sh_ders_cache_.data()
								+ static_cast<size_t>(j) * 3 * sh_count;
							cached_radial_values =
								grad_neighbor_mu_contract_vals_cache_.data()
								+ static_cast<size_t>(j) * mu_stride;
							cached_radial_derivatives =
								grad_neighbor_mu_contract_ders_cache_.data()
								+ static_cast<size_t>(j) * mu_stride;
						}

							const double center_type_coeff = regression_coeffs[shared_type_offset + type_central];
							const double outer_type_coeff = regression_coeffs[shared_type_offset + type_outer];
							const double pair_type_scale = center_type_coeff * outer_type_coeff;
							const double gate_residual = TwoLayerGateNeighborResidual(nbh, j);
							PrepareTwoLayerGateNeighborMuBuffers(
								type_outer, center_type_coeff, outer_type_coeff, gate_residual);
							const double* gate_type_scale_by_mu =
								two_layer_gate_type_scale_mu_buffer_.data();
							double* gate_scaled_radial_by_mu =
								two_layer_gate_scaled_mu_vals_buffer_.data();
							double* gate_scaled_der_by_mu =
								two_layer_gate_scaled_mu_ders_buffer_.data();
							for (int mu = 0; mu < radial_func_count; ++mu) {
								const double type_scale = gate_type_scale_by_mu[mu];
								gate_scaled_radial_by_mu[mu] =
									type_scale * cached_radial_values[mu];
								gate_scaled_der_by_mu[mu] =
									type_scale * cached_radial_derivatives[mu];
							}

							for (int i = 0; i < alpha_index_basic_count; ++i) {
								const int mu = basic_mu_cache_[i];
							const int sh_index = basic_sh_index_cache_[i];
							const int sh_der_index = basic_sh_der_index_cache_[i];
							const double radial_value = gate_scaled_radial_by_mu[mu];
							const double radial_der = gate_scaled_der_by_mu[mu];
							const double y = cached_sh_values[sh_index];
								const double wdy =
									se_weight_vec[0] * cached_sh_ders[sh_der_index]
									+ se_weight_vec[1] * cached_sh_ders[sh_der_index + 1]
									+ se_weight_vec[2] * cached_sh_ders[sh_der_index + 2];
								grad_dloss_dsenders_[i] += radial_der * wr * y + radial_value * wdy;
							}
				}
				profile_after_force_cache = profile_accum ? SHAccumProfileNow() : 0.0;

						if (use_product_rows) {
							AccumulateSHProductRowsForward(grad_dloss_dsenders_, grad_dloss_dsenders_);
				} else {
					for (size_t p = 0; p < sh_products_.size(); ++p) {
						const SHProduct& product = sh_products_[p];
						grad_dloss_dsenders_[product.target] += product.coeff * (
							grad_dloss_dsenders_[product.left] * moment_vals[product.right]
							+ moment_vals[product.left] * grad_dloss_dsenders_[product.right]);
						}
					}
					profile_after_tangent_forward = profile_accum ? SHAccumProfileNow() : 0.0;

					if (use_product_rows) {
						for (const SHProductRow& row : sh_product_rows_) {
						const double adj_target = site_energy_ders_wrt_moments_[row.target];
						if (adj_target == 0.0)
							continue;
						const int end = row.term_begin + row.term_count;
						for (int t = row.term_begin; t < end; ++t) {
							const SHProductRowTerm& term = sh_product_row_terms_[t];
							grad_dloss_dmom_[term.left] +=
								grad_dloss_dsenders_[term.right] * adj_target * term.coeff;
							grad_dloss_dmom_[term.right] +=
								grad_dloss_dsenders_[term.left] * adj_target * term.coeff;
						}
					}
				} else if (!use_product_hvt_reverse) {
					for (size_t p = 0; p < sh_products_.size(); ++p) {
						const SHProduct& product = sh_products_[p];
						const double adj_target = site_energy_ders_wrt_moments_[product.target];
						if (adj_target == 0.0)
							continue;
						grad_dloss_dmom_[product.left] +=
							grad_dloss_dsenders_[product.right] * adj_target * product.coeff;
						grad_dloss_dmom_[product.right] +=
							grad_dloss_dsenders_[product.left] * adj_target * product.coeff;
					}
				}
				profile_after_mixed_seed = profile_accum ? SHAccumProfileNow() : 0.0;
			}

		for (int i = 0; i < alpha_index_basic_count; ++i)
			grad_dloss_dmom_[i] += se_weight * site_energy_ders_wrt_moments_[i];

			if (se_ders_weights != nullptr) {
				if (use_product_rows) {
					BackpropSHProductRows(grad_dloss_dmom_);
				} else if (use_product_hvt_reverse) {
					for (int p = static_cast<int>(sh_products_.size()) - 1; p >= 0; --p) {
						const SHProduct& product = sh_products_[p];
						const double adj_target = site_energy_ders_wrt_moments_[product.target];
						const double hvt_target = grad_dloss_dmom_[product.target];
						if (adj_target == 0.0 && hvt_target == 0.0)
							continue;
						grad_dloss_dmom_[product.left] += product.coeff * (
							moment_vals[product.right] * hvt_target
							+ grad_dloss_dsenders_[product.right] * adj_target);
						grad_dloss_dmom_[product.right] += product.coeff * (
							moment_vals[product.left] * hvt_target
							+ grad_dloss_dsenders_[product.left] * adj_target);
					}
				} else {
					for (int p = static_cast<int>(sh_products_.size()) - 1; p >= 0; --p) {
						const SHProduct& product = sh_products_[p];
						const double adj_target = grad_dloss_dmom_[product.target];
						if (adj_target == 0.0)
							continue;
						grad_dloss_dmom_[product.left] +=
							product.coeff * moment_vals[product.right] * adj_target;
						grad_dloss_dmom_[product.right] +=
							product.coeff * moment_vals[product.left] * adj_target;
					}
				}
			}
		const double profile_after_force_product = profile_accum ? SHAccumProfileNow() : 0.0;

		buff_site_energy_ += regression_coeffs[type_central] + center_linear + center_linear * scalar_sum;
		buff_site_energy_0 += center_linear * scalar_sum;
		if (shift_)
			grad_out[type_central] += se_weight;

		double center_linear_force_grad = 0.0;
		if (se_ders_weights != nullptr) {
			for (int i = 0; i < alpha_scalar_moments; ++i) {
				const int node = alpha_moment_mapping[i];
				const double moment_coeff = linear_coeffs[species_count + i] * linear_mults[i];
				center_linear_force_grad += moment_coeff * grad_dloss_dsenders_[node];
			}
		}
		grad_out[coeff_count + type_central] +=
			se_weight * (1.0 + scalar_sum) + center_linear_force_grad;
		for (int i = 0; i < alpha_scalar_moments; ++i) {
			const int node = alpha_moment_mapping[i];
			const double sender = (se_ders_weights == nullptr) ? 0.0 : grad_dloss_dsenders_[node];
			grad_out[coeff_count + species_count + i] +=
				center_linear * linear_mults[i] * (se_weight * moment_vals[node] + sender);
		}
		const double profile_after_scalar_grad = profile_accum ? SHAccumProfileNow() : 0.0;

		if (static_cast<int>(grad_radial_coeff_value_accum_.size()) != radial_func_count)
			grad_radial_coeff_value_accum_.resize(radial_func_count);
		if (static_cast<int>(grad_radial_coeff_coord_accum_.size()) != radial_func_count)
			grad_radial_coeff_coord_accum_.resize(radial_func_count);

		for (int j = 0; j < nbh.count; ++j) {
			const Vector3& rvec = nbh.vecs[j];
			const double r = nbh.dists[j];
			const double inv_r = 1.0 / r;
			const int type_outer = nbh.types[j];
				const double wx = (se_ders_weights == nullptr) ? 0.0 : se_ders_weights[j][0];
				const double wy = (se_ders_weights == nullptr) ? 0.0 : se_ders_weights[j][1];
				const double wz = (se_ders_weights == nullptr) ? 0.0 : se_ders_weights[j][2];
				const double wr = (wx * rvec[0] + wy * rvec[1] + wz * rvec[2]) * inv_r;
				const double* sh_values_use = sh_values;
				const double* sh_ders_use = sh_ders;
				const double* rb_vals_use = rb_vals.data();
				const double* rb_ders_use = rb_ders.data();
				const double* radial_values_use = radial_values.data();
				const double* radial_derivatives_use = radial_derivatives.data();
					const double* radial_s_derivatives_use = radial_s_derivatives.data();
					const double* radial_ss_derivatives_use = radial_ss_derivatives.data();
					const double* radial_coord_s_derivatives_use = radial_coord_s_derivatives.data();
					const double* radial_coord_ss_derivatives_use = radial_coord_ss_derivatives.data();
					if (se_ders_weights != nullptr) {
						if (use_edge_cache_for_gradient) {
							const size_t edge = TwoLayerEdgePrimitiveOffset(-1, j);
							sh_values_use =
								two_layer_edge_sh_values_cache_.data() + edge * sh_count;
							sh_ders_use =
								two_layer_edge_sh_ders_cache_.data() + edge * 3 * sh_count;
							rb_vals_use =
								two_layer_edge_radial_vals_cache_.data()
								+ edge * radial_val_stride;
							rb_ders_use =
								two_layer_edge_radial_ders_cache_.data()
								+ edge * radial_der_stride;
							radial_values_use =
								two_layer_edge_mu_vals_cache_.data() + edge * mu_stride;
							radial_derivatives_use =
								two_layer_edge_mu_ders_cache_.data() + edge * mu_stride;
							radial_s_derivatives_use =
								two_layer_edge_mu_ders_s_cache_.data() + edge * mu_stride;
							radial_coord_s_derivatives_use =
								two_layer_edge_mu_coord_ders_s_cache_.data()
								+ edge * mu_stride;
							radial_ss_derivatives_use =
								two_layer_edge_mu_ders_ss_cache_.data() + edge * mu_stride;
							radial_coord_ss_derivatives_use =
								two_layer_edge_mu_coord_ders_ss_cache_.data()
								+ edge * mu_stride;
						} else {
							sh_values_use =
								grad_neighbor_sh_values_cache_.data()
								+ static_cast<size_t>(j) * sh_count;
							sh_ders_use =
								grad_neighbor_sh_ders_cache_.data()
								+ static_cast<size_t>(j) * 3 * sh_count;
							rb_vals_use =
								grad_neighbor_radial_vals_cache_.data()
								+ static_cast<size_t>(j) * radial_val_stride;
							rb_ders_use =
								grad_neighbor_radial_ders_cache_.data()
								+ static_cast<size_t>(j) * radial_der_stride;
							radial_values_use =
								grad_neighbor_mu_contract_vals_cache_.data()
								+ static_cast<size_t>(j) * mu_stride;
							radial_derivatives_use =
								grad_neighbor_mu_contract_ders_cache_.data()
								+ static_cast<size_t>(j) * mu_stride;
							radial_s_derivatives_use =
								grad_neighbor_mu_contract_ders_s_cache_.data()
								+ static_cast<size_t>(j) * mu_stride;
							radial_coord_s_derivatives_use =
								grad_neighbor_mu_contract_coord_ders_s_cache_.data()
								+ static_cast<size_t>(j) * mu_stride;
							radial_ss_derivatives_use =
								grad_neighbor_mu_contract_ders_ss_cache_.data()
								+ static_cast<size_t>(j) * mu_stride;
							radial_coord_ss_derivatives_use =
								grad_neighbor_mu_contract_coord_ders_ss_cache_.data()
								+ static_cast<size_t>(j) * mu_stride;
						}
					} else {
					EvalRealSH(rvec, r, sh_l_max_, sh_values, sh_ders);

				for (int eval_block = 0; eval_block < eval_block_count; ++eval_block) {
					const int scaling_block = radial_eval_to_scaling_block_[eval_block];
					const int basis_k = radial_eval_to_basis_k_[eval_block];
					p_RadialBasis->RB_Calc(
					r,
					regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer],
					regression_coeffs[C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer],
					basis_k);
				for (int xi = 0; xi < R; ++xi)
					rb_vals[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
				for (int xi = 0; xi < R * 5; ++xi)
					rb_ders[eval_block * R * 5 + xi] = p_RadialBasis->rb_ders[xi] * scaling;
			}
			for (int mu = 0; mu < radial_func_count; ++mu) {
				const int radial_base = mu_to_radial_eval_block_[mu] * R;
				const int deriv_base = 5 * radial_base;
				const int radial_offset = radial_coeff_base + mu * (R + C);
				double dot_val = 0.0;
				double dot_der = 0.0;
				double dot_s = 0.0;
				double dot_coord_s = 0.0;
				double dot_ss = 0.0;
				double dot_coord_ss = 0.0;
				for (int xi = 0; xi < R; ++xi) {
					const double coeff = regression_coeffs[radial_offset + xi];
					dot_val += coeff * rb_vals[radial_base + xi];
					dot_der += coeff * rb_ders[deriv_base + xi];
					dot_s += coeff * rb_ders[deriv_base + xi + R];
					dot_coord_s += coeff * rb_ders[deriv_base + xi + 2 * R];
					dot_ss += coeff * rb_ders[deriv_base + xi + 3 * R];
					dot_coord_ss += coeff * rb_ders[deriv_base + xi + 4 * R];
				}
				radial_values[mu] = dot_val;
				radial_derivatives[mu] = dot_der;
				radial_s_derivatives[mu] = dot_s;
				radial_coord_s_derivatives[mu] = dot_coord_s;
					radial_ss_derivatives[mu] = dot_ss;
					radial_coord_ss_derivatives[mu] = dot_coord_ss;
				}
				}

						const double center_type_coeff = regression_coeffs[shared_type_offset + type_central];
						const double outer_type_coeff = regression_coeffs[shared_type_offset + type_outer];
					const double pair_type_scale = center_type_coeff * outer_type_coeff;
					const double gate_residual = TwoLayerGateNeighborResidual(nbh, j);
					PrepareTwoLayerGateNeighborMuBuffers(
						type_outer, center_type_coeff, outer_type_coeff, gate_residual);
					const double* gate_additive_by_mu =
						two_layer_gate_additive_mu_buffer_.data();
					const double* gate_type_scale_by_mu =
						two_layer_gate_type_scale_mu_buffer_.data();
					const bool skip_outer_param_grad =
						two_layer_residual_skip_outer_param_grad_;
					double gate_adjoint = 0.0;
			double* radial_coeff_value_accum = grad_radial_coeff_value_accum_.data();
			double* radial_coeff_coord_accum = grad_radial_coeff_coord_accum_.data();
			std::fill(radial_coeff_value_accum,
			          radial_coeff_value_accum + radial_func_count,
			          0.0);
			std::fill(radial_coeff_coord_accum,
			          radial_coeff_coord_accum + radial_func_count,
			          0.0);

					for (int i = 0; i < alpha_index_basic_count; ++i) {
						const int mu = basic_mu_cache_[i];
						const int scaling_block = basic_scaling_block_cache_[i];
						const int sh_index = basic_sh_index_cache_[i];
						const int sh_der_index = basic_sh_der_index_cache_[i];
						const double y = sh_values_use[sh_index];
					double dy[3];
					dy[0] = sh_ders_use[sh_der_index];
					dy[1] = sh_ders_use[sh_der_index + 1];
					dy[2] = sh_ders_use[sh_der_index + 2];
					const double wdy = wx * dy[0] + wy * dy[1] + wz * dy[2];

					const double dot_val = radial_values_use[mu];
					const double dot_der = radial_derivatives_use[mu];
					const double dot_s = radial_s_derivatives_use[mu];
					const double dot_coord_s = radial_coord_s_derivatives_use[mu];
					const double dot_ss = radial_ss_derivatives_use[mu];
					const double dot_coord_ss = radial_coord_ss_derivatives_use[mu];

				const double value_weight = center_linear * grad_dloss_dmom_[i];
				const double coord_weight = (se_ders_weights == nullptr)
					? 0.0
					: center_linear * site_energy_ders_wrt_moments_[i];
					const double energy_der_weight = center_linear * site_energy_ders_wrt_moments_[i];

					const double base_direction = dot_der * wr * y + dot_val * wdy;
					const double final_contribution_weight =
						value_weight * dot_val * y
						+ coord_weight * base_direction;
						radial_coeff_value_accum[mu] +=
							value_weight * y + coord_weight * wdy;
							radial_coeff_coord_accum[mu] += coord_weight * y;

						if (!skip_site_der_output && energy_der_weight != 0.0) {
						const double type_scale = gate_type_scale_by_mu[mu];
							for (int a = 0; a < 3; ++a) {
							const double base_coord = dot_der * rvec[a] * inv_r * y + dot_val * dy[a];
							buff_site_energy_ders_[j][a] += energy_der_weight * type_scale * base_coord;
								}
							}
						}
					for (int mu = 0; mu < radial_func_count; ++mu) {
						const double value_accum = radial_coeff_value_accum[mu];
						const double coord_accum = radial_coeff_coord_accum[mu];
							if (value_accum == 0.0 && coord_accum == 0.0)
								continue;
								const int radial_base = mu_to_radial_eval_block_[mu] * R;
								const int deriv_base = 5 * radial_base;
								const int radial_offset = radial_coeff_base + mu * (R + C);
								const double additive_coeff = gate_additive_by_mu[mu];
								const double mu_contribution =
								radial_values_use[mu] * value_accum
								+ radial_derivatives_use[mu] * wr * coord_accum;
							gate_adjoint +=
								center_type_coeff * additive_coeff * mu_contribution;
							if (two_layer_gate_enabled_)
								grad_out[TwoLayerGateAdditiveCoeffIndex(type_outer, mu)] +=
									center_type_coeff * gate_residual * mu_contribution;
								if (skip_outer_param_grad)
									continue;
								const double type_scale = gate_type_scale_by_mu[mu];
								const double outer_plus_gate =
									outer_type_coeff + additive_coeff * gate_residual;
								const int scaling_block =
									radial_eval_to_scaling_block_[mu_to_radial_eval_block_[mu]];
								const int sigma_coeff_offset =
									C + 2 * C * C * scaling_block
									+ type_central * C + type_outer;
								grad_out[shared_type_offset + type_central] +=
									outer_plus_gate * mu_contribution;
								grad_out[shared_type_offset + type_outer] +=
									center_type_coeff * mu_contribution;
								grad_out[sigma_coeff_offset] += type_scale * (
									radial_s_derivatives_use[mu] * value_accum
									+ radial_coord_s_derivatives_use[mu] * wr * coord_accum);
								grad_out[sigma_coeff_offset + C * C] += type_scale * (
									radial_ss_derivatives_use[mu] * value_accum
									+ radial_coord_ss_derivatives_use[mu] * wr * coord_accum);
								for (int xi = 0; xi < R; ++xi) {
								grad_out[radial_offset + xi] += type_scale * (
									rb_vals_use[radial_base + xi] * value_accum
								+ rb_ders_use[deriv_base + xi] * wr * coord_accum);
						}
					}
				AddTwoLayerGateAdjoint(nbh, j, gate_adjoint);
				}
		if (profile_accum) {
			const double profile_end = SHAccumProfileNow();
			RecordSHAccumProfile(
				profile_end - profile_start,
				profile_after_moment - profile_start,
				profile_after_energy_backprop - profile_after_moment,
				profile_after_force_product - profile_after_energy_backprop,
				profile_after_force_cache - profile_after_energy_backprop,
				profile_after_tangent_forward - profile_after_force_cache,
				profile_after_mixed_seed - profile_after_tangent_forward,
				profile_after_force_product - profile_after_mixed_seed,
				profile_after_scalar_grad - profile_after_force_product,
				profile_end - profile_after_scalar_grad,
				nbh.count);
		}
		return;
	}

	CalcSHBasisFuncsDers(nbh);
	buff_site_energy_ = 0.0;
	buff_site_energy_0 = 0.0;
	buff_site_energy_ders_.resize(nbh.count);
	FillWithZero(buff_site_energy_ders_);
	out_grad_accumulator.resize(CoeffCount());
	double* grad_out = out_grad_accumulator.data();

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int K = radial_func_count;
	const int coeff_count = LinearCoeffOffset();
	const int type_central = nbh.my_type;
	const int radial_coeff_base = C + 2 * C * C * K_;
	const int shared_type_offset = radial_coeff_base + R;
	const double center_linear = linear_coeffs[type_central];

	if (static_cast<int>(sh_adj_vals_.size()) != alpha_moments_count)
		sh_adj_vals_.resize(alpha_moments_count);
	std::fill(sh_adj_vals_.begin(), sh_adj_vals_.end(), 0.0);
	if (sh_adj_ders_.size1 != alpha_moments_count || sh_adj_ders_.size2 != nbh.count)
		sh_adj_ders_.resize(alpha_moments_count, nbh.count, 3);
	sh_adj_ders_.set(0);

	double scalar_sum = 0.0;
	double center_linear_force_grad = 0.0;
	for (int i = 0; i < alpha_scalar_moments; ++i) {
		const int node = alpha_moment_mapping[i];
		const double moment_coeff = linear_coeffs[species_count + i] * linear_mults[i];
		scalar_sum += moment_coeff * moment_vals[node];
		sh_adj_vals_[node] += se_weight * center_linear * moment_coeff;
		if (se_ders_weights != nullptr) {
			for (int j = 0; j < nbh.count; ++j)
				for (int a = 0; a < 3; ++a) {
					const double weighted_der = moment_coeff * basis_ders(1 + i, j, a);
					center_linear_force_grad += se_ders_weights[j][a] * weighted_der;
					sh_adj_ders_(node, j, a) += center_linear * moment_coeff * se_ders_weights[j][a];
				}
		}
	}

	buff_site_energy_ += regression_coeffs[type_central] + center_linear + center_linear * scalar_sum;
	buff_site_energy_0 += center_linear * scalar_sum;
	for (int i = 0; i < alpha_scalar_moments; ++i) {
		const double coeff = center_linear * linear_coeffs[species_count + i] * linear_mults[i];
		const int node = alpha_moment_mapping[i];
		for (int j = 0; j < nbh.count; ++j)
			for (int a = 0; a < 3; ++a)
				buff_site_energy_ders_[j][a] += coeff * moment_ders(node, j, a);
	}

	for (int p = static_cast<int>(sh_products_.size()) - 1; p >= 0; --p) {
		const SHProduct& product = sh_products_[p];
		const double left_value = moment_vals[product.left];
		const double right_value = moment_vals[product.right];
		const double adj_target = sh_adj_vals_[product.target];
		if (adj_target != 0.0) {
			sh_adj_vals_[product.left] += product.coeff * right_value * adj_target;
			sh_adj_vals_[product.right] += product.coeff * left_value * adj_target;
		}
		for (int j = 0; j < nbh.count; ++j)
			for (int a = 0; a < 3; ++a) {
				const double adj_der = sh_adj_ders_(product.target, j, a);
				if (adj_der == 0.0)
					continue;
				sh_adj_ders_(product.left, j, a) += product.coeff * right_value * adj_der;
				sh_adj_ders_(product.right, j, a) += product.coeff * left_value * adj_der;
				sh_adj_vals_[product.left] += product.coeff * moment_ders(product.right, j, a) * adj_der;
				sh_adj_vals_[product.right] += product.coeff * moment_ders(product.left, j, a) * adj_der;
			}
	}

	if (shift_)
		grad_out[type_central] += se_weight;
	grad_out[coeff_count + type_central] += se_weight * (1.0 + scalar_sum) + center_linear_force_grad;
	for (int i = 0; i < alpha_scalar_moments; ++i) {
		const int node = alpha_moment_mapping[i];
		double grad = se_weight * center_linear * linear_mults[i] * moment_vals[node];
		if (se_ders_weights != nullptr) {
			for (int j = 0; j < nbh.count; ++j)
				for (int a = 0; a < 3; ++a)
					grad += center_linear * linear_mults[i] * se_ders_weights[j][a] * moment_ders(node, j, a);
		}
		grad_out[coeff_count + species_count + i] += grad;
	}

		double sh_values[kMaxSHComponents];
		double sh_ders[3 * kMaxSHComponents];
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& rb_ders = radial_ders_buffer_;

	for (int j = 0; j < nbh.count; ++j) {
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const double inv_r = 1.0 / r;
		const int type_outer = nbh.types[j];
			EvalRealSH(rvec, r, sh_l_max_, sh_values, sh_ders);

		for (int eval_block = 0; eval_block < static_cast<int>(radial_eval_to_scaling_block_.size()); ++eval_block) {
			const int scaling_block = radial_eval_to_scaling_block_[eval_block];
			const int basis_k = radial_eval_to_basis_k_[eval_block];
			p_RadialBasis->RB_Calc(
				r,
				regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer],
				regression_coeffs[C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer],
				basis_k);
			for (int xi = 0; xi < R; ++xi)
				rb_vals[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
			for (int xi = 0; xi < R * 5; ++xi)
				rb_ders[eval_block * R * 5 + xi] = p_RadialBasis->rb_ders[xi] * scaling;
		}

				const double center_type_coeff = regression_coeffs[shared_type_offset + type_central];
				const double outer_type_coeff = regression_coeffs[shared_type_offset + type_outer];
				const double pair_type_scale = center_type_coeff * outer_type_coeff;
				const double gate_residual = TwoLayerGateNeighborResidual(nbh, j);
				PrepareTwoLayerGateNeighborMuBuffers(
					type_outer, center_type_coeff, outer_type_coeff, gate_residual);
				const double* gate_additive_by_mu =
					two_layer_gate_additive_mu_buffer_.data();
				const double* gate_type_scale_by_mu =
					two_layer_gate_type_scale_mu_buffer_.data();

					for (int i = 0; i < alpha_index_basic_count; ++i) {
					const int mu = basic_mu_cache_[i];
					const int scaling_block = basic_scaling_block_cache_[i];
					const int radial_base = basic_radial_base_cache_[i];
					const int deriv_base = basic_radial_deriv_base_cache_[i];
					const int radial_offset = basic_radial_offset_cache_[i];
				const int sh_index = basic_sh_index_cache_[i];
				const int sh_der_index = basic_sh_der_index_cache_[i];
				const double y = sh_values[sh_index];
			double dy[3];
			for (int a = 0; a < 3; ++a)
				dy[a] = sh_ders[sh_der_index + a];

			double dot_val = 0.0;
			double dot_der = 0.0;
			double dot_s = 0.0;
			double dot_coord_s = 0.0;
			double dot_ss = 0.0;
			double dot_coord_ss = 0.0;
			for (int xi = 0; xi < R; ++xi) {
				const double coeff = regression_coeffs[radial_offset + xi];
				dot_val += coeff * rb_vals[radial_base + xi];
				dot_der += coeff * rb_ders[deriv_base + xi];
				dot_s += coeff * rb_ders[deriv_base + xi + R];
				dot_coord_s += coeff * rb_ders[deriv_base + xi + 2 * R];
				dot_ss += coeff * rb_ders[deriv_base + xi + 3 * R];
				dot_coord_ss += coeff * rb_ders[deriv_base + xi + 4 * R];
			}

					const double adj_value = sh_adj_vals_[i];
					double adj_der[3] = {sh_adj_ders_(i, j, 0), sh_adj_ders_(i, j, 1), sh_adj_ders_(i, j, 2)};
					const double additive_coeff = gate_additive_by_mu[mu];
					const double type_scale = gate_type_scale_by_mu[mu];
				for (int xi = 0; xi < R; ++xi) {
					const double rb_val = rb_vals[radial_base + xi];
				const double rb_der = rb_ders[deriv_base + xi];
				double grad = adj_value * type_scale * rb_val * y;
				for (int a = 0; a < 3; ++a)
					grad += adj_der[a] * type_scale * (rb_der * rvec[a] * inv_r * y + rb_val * dy[a]);
				grad_out[radial_offset + xi] += grad;
			}

				double center_grad =
					adj_value * (outer_type_coeff + additive_coeff * gate_residual)
					* dot_val * y;
				double outer_grad = adj_value * center_type_coeff * dot_val * y;
				double sigma_grad = adj_value * type_scale * dot_s * y;
				double shift_grad = adj_value * type_scale * dot_ss * y;
				double additive_grad =
					adj_value * center_type_coeff * gate_residual * dot_val * y;
				for (int a = 0; a < 3; ++a) {
					const double base_coord = dot_der * rvec[a] * inv_r * y + dot_val * dy[a];
					const double sigma_coord = dot_coord_s * rvec[a] * inv_r * y + dot_s * dy[a];
					const double shift_coord = dot_coord_ss * rvec[a] * inv_r * y + dot_ss * dy[a];
					center_grad +=
						adj_der[a] * (outer_type_coeff + additive_coeff * gate_residual)
						* base_coord;
					outer_grad += adj_der[a] * center_type_coeff * base_coord;
					sigma_grad += adj_der[a] * type_scale * sigma_coord;
					shift_grad += adj_der[a] * type_scale * shift_coord;
					additive_grad +=
						adj_der[a] * center_type_coeff * gate_residual * base_coord;
				}

				const int sigma_coeff_offset = C + 2 * C * C * scaling_block + type_central * C + type_outer;
				grad_out[shared_type_offset + type_central] += center_grad;
				grad_out[shared_type_offset + type_outer] += outer_grad;
				grad_out[sigma_coeff_offset] += sigma_grad;
				grad_out[sigma_coeff_offset + C * C] += shift_grad;
				if (two_layer_gate_enabled_)
					grad_out[TwoLayerGateAdditiveCoeffIndex(type_outer, mu)] +=
						additive_grad;
			}
		}
	}
