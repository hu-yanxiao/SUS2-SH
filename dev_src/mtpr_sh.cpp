/* SUS2-SH real spherical-harmonic evaluation and analytic coefficient gradients. */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <vector>

#include "mtpr.h"

using namespace std;

namespace {

const double kPi = std::acos(-1.0);
const double kSqrt2 = std::sqrt(2.0);
const int kMaxSHComponents = 25; // lmax <= 4

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

void MLMTPR::CalcSHBasisFuncs(const Neighborhood& nbh, double* bf_vals)
{
	CalcSHMomentValuesOnly(nbh);
	for (int i = 0; i < alpha_count; ++i)
		bf_vals[i] = basis_vals[i];
}

void MLMTPR::CalcSHMomentValuesOnly(const Neighborhood& nbh)
{
	if (SHUsesPrecomputedLmpTable(p_RadialBasis))
		ERROR("SUS2-SH does not use precomputed LAMMPS radial tables in training.");
	if (sh_l_max_ < 0 || sh_l_max_ > 4)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,4].");

	const int C = species_count;
	const int R = p_RadialBasis->rb_size;
	const int type_central = nbh.my_type;
	if (type_central >= species_count)
		throw MlipException("Too few species count in the MTP potential!");

	std::fill(moment_vals, moment_vals + alpha_moments_count, 0.0);

	double sh_values[kMaxSHComponents];
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& radial_values = grad_mu_contract_vals_;

	const int radial_coeff_base = C + 2 * C * C * K_;
	const int shared_type_offset = radial_coeff_base + R;

	for (int j = 0; j < nbh.count; ++j) {
		const Vector3& rvec = nbh.vecs[j];
		const double r = nbh.dists[j];
		const int type_outer = nbh.types[j];
		EvalRealSHValuesOnly(rvec, r, sh_l_max_, sh_values);

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
		}
		for (int mu = 0; mu < radial_func_count; ++mu) {
			const int radial_base = mu_to_radial_eval_block_[mu] * R;
			const int radial_offset = radial_coeff_base + mu * (R + C);
			double radial_value = 0.0;
			for (int xi = 0; xi < R; ++xi)
				radial_value += regression_coeffs[radial_offset + xi] * rb_vals[radial_base + xi];
			radial_values[mu] = radial_value;
		}

		const double center_type_coeff = regression_coeffs[shared_type_offset + type_central];
		const double outer_type_coeff = regression_coeffs[shared_type_offset + type_outer];
		const double type_scale = center_type_coeff * outer_type_coeff;

			for (int i = 0; i < alpha_index_basic_count; ++i) {
				const int mu = alpha_index_basic_.comp0[i];
				const int l = alpha_index_basic_.comp1[i];
				const int m = alpha_index_basic_.comp2[i];
				moment_vals[i] += type_scale * radial_values[mu] * RealSHValue(sh_values, l, m);
			}
	}

	for (size_t p = 0; p < sh_products_.size(); ++p) {
		const SHProduct& product = sh_products_[p];
		moment_vals[product.target] +=
			product.coeff * moment_vals[product.left] * moment_vals[product.right];
	}

	basis_vals[0] = 1.0;
	for (int i = 0; i < alpha_scalar_moments; ++i)
		basis_vals[1 + i] = moment_vals[alpha_moment_mapping[i]];
}

void MLMTPR::CalcSHBasisFuncsDers(const Neighborhood& nbh)
{
	if (SHUsesPrecomputedLmpTable(p_RadialBasis))
		ERROR("SUS2-SH does not use precomputed LAMMPS radial tables in training.");
	if (sh_l_max_ < 0 || sh_l_max_ > 4)
		ERROR("SUS2-SH evaluator currently supports sh_l_max in [0,4].");

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

		const double center_type_coeff = regression_coeffs[shared_type_offset + type_central];
		const double outer_type_coeff = regression_coeffs[shared_type_offset + type_outer];
		const double type_scale = center_type_coeff * outer_type_coeff;

			for (int i = 0; i < alpha_index_basic_count; ++i) {
				const int mu = alpha_index_basic_.comp0[i];
				const int l = alpha_index_basic_.comp1[i];
				const int m = alpha_index_basic_.comp2[i];
				const double radial_value = radial_values[mu];
			const double radial_der = radial_derivatives[mu];

			const double y = RealSHValue(sh_values, l, m);
			moment_vals[i] += type_scale * radial_value * y;
			for (int a = 0; a < 3; ++a) {
				const double dy = RealSHDer(sh_ders, l, m, a);
				moment_ders(i, j, a) += type_scale * (radial_der * rvec[a] * inv_r * y + radial_value * dy);
			}
		}
	}

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

	basis_vals[0] = 1.0;
	if (basis_ders.size1 != alpha_count || basis_ders.size2 != nbh.count)
		basis_ders.resize(alpha_count, nbh.count, 3);
	basis_ders.set(0);
	for (int i = 0; i < alpha_scalar_moments; ++i) {
		const int node = alpha_moment_mapping[i];
		basis_vals[1 + i] = moment_vals[node];
		for (int j = 0; j < nbh.count; ++j)
			for (int a = 0; a < 3; ++a)
				basis_ders(1 + i, j, a) = moment_ders(node, j, a);
	}
}

void MLMTPR::CalcSHSiteEnergyDers(const Neighborhood& nbh)
{
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

	double sh_values[kMaxSHComponents];
	double sh_ders[3 * kMaxSHComponents];
	std::vector<double>& rb_vals = radial_vals_buffer_;
	std::vector<double>& rb_ders = basis_radial_ders_buffer_;
	std::vector<double>& radial_values = grad_mu_contract_vals_;
	std::vector<double>& radial_derivatives = grad_mu_contract_ders_;

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

		const double center_type_coeff = regression_coeffs[shared_type_offset + type_central];
		const double outer_type_coeff = regression_coeffs[shared_type_offset + type_outer];
		const double type_scale = center_type_coeff * outer_type_coeff;

		for (int i = 0; i < alpha_index_basic_count; ++i) {
			const double adj = center_linear * site_energy_ders_wrt_moments_[i];
			if (adj == 0.0)
				continue;
			const int mu = alpha_index_basic_.comp0[i];
			const int l = alpha_index_basic_.comp1[i];
			const int m = alpha_index_basic_.comp2[i];
			const double radial_value = radial_values[mu];
			const double radial_der = radial_derivatives[mu];
			const double y = RealSHValue(sh_values, l, m);
			for (int a = 0; a < 3; ++a) {
				const double dy = RealSHDer(sh_ders, l, m, a);
				buff_site_energy_ders_[j][a] +=
					adj * type_scale * (radial_der * rvec[a] * inv_r * y + radial_value * dy);
			}
		}
	}
}

void MLMTPR::AccumulateSHCombinationGrad(const Neighborhood& nbh,
                                         std::vector<double>& out_grad_accumulator,
                                         const double se_weight,
                                         const Vector3* se_ders_weights)
{
	if (SHUsesPrecomputedLmpTable(p_RadialBasis))
		ERROR("SUS2-SH coefficient gradients require direct radial basis evaluation.");

	{
		CalcSHMomentValuesOnly(nbh);
		buff_site_energy_ = 0.0;
		buff_site_energy_0 = 0.0;
		buff_site_energy_ders_.resize(nbh.count);
		FillWithZero(buff_site_energy_ders_);
		out_grad_accumulator.resize(CoeffCount());
		double* grad_out = out_grad_accumulator.data();

		const int C = species_count;
		const int R = p_RadialBasis->rb_size;
		const int K = radial_func_count;
		const int coeff_count = C + 2 * C * C * K_ + (R + C) * K;
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

		double scalar_sum = 0.0;
		for (int i = 0; i < alpha_scalar_moments; ++i) {
			const int node = alpha_moment_mapping[i];
			const double moment_coeff = linear_coeffs[species_count + i] * linear_mults[i];
			scalar_sum += moment_coeff * moment_vals[node];
			site_energy_ders_wrt_moments_[node] += moment_coeff;
		}

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

			if (se_ders_weights != nullptr) {
				const size_t neighbor_count = static_cast<size_t>(nbh.count);
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
				for (int j = 0; j < nbh.count; ++j) {
					const Vector3& rvec = nbh.vecs[j];
					const Vector3& se_weight_vec = se_ders_weights[j];
					const double r = nbh.dists[j];
					const double inv_r = 1.0 / r;
				const double wr = (se_weight_vec[0] * rvec[0]
						+ se_weight_vec[1] * rvec[1]
						+ se_weight_vec[2] * rvec[2]) * inv_r;
					const int type_outer = nbh.types[j];
					double* cached_sh_values = grad_neighbor_sh_values_cache_.data() + static_cast<size_t>(j) * sh_count;
					double* cached_sh_ders = grad_neighbor_sh_ders_cache_.data() + static_cast<size_t>(j) * 3 * sh_count;
					double* cached_rb_vals = grad_neighbor_radial_vals_cache_.data() + static_cast<size_t>(j) * radial_val_stride;
					double* cached_rb_ders = grad_neighbor_radial_ders_cache_.data() + static_cast<size_t>(j) * radial_der_stride;
					double* cached_radial_values = grad_neighbor_mu_contract_vals_cache_.data() + static_cast<size_t>(j) * mu_stride;
					double* cached_radial_derivatives = grad_neighbor_mu_contract_ders_cache_.data() + static_cast<size_t>(j) * mu_stride;
					double* cached_radial_s_derivatives = grad_neighbor_mu_contract_ders_s_cache_.data() + static_cast<size_t>(j) * mu_stride;
					double* cached_radial_ss_derivatives = grad_neighbor_mu_contract_ders_ss_cache_.data() + static_cast<size_t>(j) * mu_stride;
					double* cached_radial_coord_s_derivatives = grad_neighbor_mu_contract_coord_ders_s_cache_.data() + static_cast<size_t>(j) * mu_stride;
					double* cached_radial_coord_ss_derivatives = grad_neighbor_mu_contract_coord_ders_ss_cache_.data() + static_cast<size_t>(j) * mu_stride;
					EvalRealSH(rvec, r, sh_l_max_, cached_sh_values, cached_sh_ders);

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
						cached_radial_values[mu] = dot_val;
						cached_radial_derivatives[mu] = dot_der;
						cached_radial_s_derivatives[mu] = dot_s;
						cached_radial_coord_s_derivatives[mu] = dot_coord_s;
						cached_radial_ss_derivatives[mu] = dot_ss;
						cached_radial_coord_ss_derivatives[mu] = dot_coord_ss;
					}

				const double center_type_coeff = regression_coeffs[shared_type_offset + type_central];
				const double outer_type_coeff = regression_coeffs[shared_type_offset + type_outer];
				const double type_scale = center_type_coeff * outer_type_coeff;

				for (int i = 0; i < alpha_index_basic_count; ++i) {
						const int mu = alpha_index_basic_.comp0[i];
						const int l = alpha_index_basic_.comp1[i];
						const int m = alpha_index_basic_.comp2[i];
						const double radial_value = cached_radial_values[mu];
						const double radial_der = cached_radial_derivatives[mu];
						const double y = RealSHValue(cached_sh_values, l, m);
						double wdy = 0.0;
						for (int a = 0; a < 3; ++a)
							wdy += se_weight_vec[a] * RealSHDer(cached_sh_ders, l, m, a);
						grad_dloss_dsenders_[i] += type_scale * (radial_der * wr * y + radial_value * wdy);
					}
			}

			for (size_t p = 0; p < sh_products_.size(); ++p) {
				const SHProduct& product = sh_products_[p];
				grad_dloss_dsenders_[product.target] += product.coeff * (
					grad_dloss_dsenders_[product.left] * moment_vals[product.right]
					+ moment_vals[product.left] * grad_dloss_dsenders_[product.right]);
			}

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

		for (int i = 0; i < alpha_index_basic_count; ++i)
			grad_dloss_dmom_[i] += se_weight * site_energy_ders_wrt_moments_[i];

		if (se_ders_weights != nullptr) {
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
					sh_values_use = grad_neighbor_sh_values_cache_.data() + static_cast<size_t>(j) * sh_count;
					sh_ders_use = grad_neighbor_sh_ders_cache_.data() + static_cast<size_t>(j) * 3 * sh_count;
					rb_vals_use = grad_neighbor_radial_vals_cache_.data() + static_cast<size_t>(j) * radial_val_stride;
					rb_ders_use = grad_neighbor_radial_ders_cache_.data() + static_cast<size_t>(j) * radial_der_stride;
					radial_values_use = grad_neighbor_mu_contract_vals_cache_.data() + static_cast<size_t>(j) * mu_stride;
					radial_derivatives_use = grad_neighbor_mu_contract_ders_cache_.data() + static_cast<size_t>(j) * mu_stride;
					radial_s_derivatives_use = grad_neighbor_mu_contract_ders_s_cache_.data() + static_cast<size_t>(j) * mu_stride;
					radial_coord_s_derivatives_use = grad_neighbor_mu_contract_coord_ders_s_cache_.data() + static_cast<size_t>(j) * mu_stride;
					radial_ss_derivatives_use = grad_neighbor_mu_contract_ders_ss_cache_.data() + static_cast<size_t>(j) * mu_stride;
					radial_coord_ss_derivatives_use = grad_neighbor_mu_contract_coord_ders_ss_cache_.data() + static_cast<size_t>(j) * mu_stride;
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
			const double type_scale = center_type_coeff * outer_type_coeff;

					for (int i = 0; i < alpha_index_basic_count; ++i) {
						const int mu = alpha_index_basic_.comp0[i];
						const int l = alpha_index_basic_.comp1[i];
						const int m = alpha_index_basic_.comp2[i];
						const int scaling_block = mu_to_K[mu];
							const int eval_block = mu_to_radial_eval_block_[mu];
							const int radial_base = eval_block * R;
							const int deriv_base = 5 * radial_base;
							const int radial_offset = radial_coeff_base + mu * (R + C);
						const double y = RealSHValue(sh_values_use, l, m);
					double dy[3];
					double wdy = 0.0;
					for (int a = 0; a < 3; ++a) {
						dy[a] = RealSHDer(sh_ders_use, l, m, a);
					}
					wdy = wx * dy[0] + wy * dy[1] + wz * dy[2];

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

					for (int xi = 0; xi < R; ++xi) {
						const double rb_val = rb_vals_use[radial_base + xi];
						const double rb_der = rb_ders_use[deriv_base + xi];
					const double radial_direction = rb_der * wr * y + rb_val * wdy;
					grad_out[radial_offset + xi] += type_scale * (
						value_weight * rb_val * y + coord_weight * radial_direction);
				}

				const double base_direction = dot_der * wr * y + dot_val * wdy;
				const double sigma_direction = dot_coord_s * wr * y + dot_s * wdy;
				const double shift_direction = dot_coord_ss * wr * y + dot_ss * wdy;
				const int sigma_coeff_offset = C + 2 * C * C * scaling_block + type_central * C + type_outer;
				grad_out[shared_type_offset + type_central] +=
					value_weight * outer_type_coeff * dot_val * y
					+ coord_weight * outer_type_coeff * base_direction;
				grad_out[shared_type_offset + type_outer] +=
					value_weight * center_type_coeff * dot_val * y
					+ coord_weight * center_type_coeff * base_direction;
				grad_out[sigma_coeff_offset] +=
					value_weight * type_scale * dot_s * y
					+ coord_weight * type_scale * sigma_direction;
				grad_out[sigma_coeff_offset + C * C] +=
					value_weight * type_scale * dot_ss * y
					+ coord_weight * type_scale * shift_direction;

				if (energy_der_weight != 0.0) {
					for (int a = 0; a < 3; ++a) {
						const double base_coord = dot_der * rvec[a] * inv_r * y + dot_val * dy[a];
						buff_site_energy_ders_[j][a] += energy_der_weight * type_scale * base_coord;
					}
				}
			}
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
	const int coeff_count = C + 2 * C * C * K_ + (R + C) * K;
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
		const double type_scale = center_type_coeff * outer_type_coeff;

			for (int i = 0; i < alpha_index_basic_count; ++i) {
				const int mu = alpha_index_basic_.comp0[i];
				const int l = alpha_index_basic_.comp1[i];
				const int m = alpha_index_basic_.comp2[i];
				const int eval_block = mu_to_radial_eval_block_[mu];
				const int scaling_block = mu_to_K[mu];
				const int radial_base = eval_block * R;
				const int deriv_base = 5 * radial_base;
				const int radial_offset = radial_coeff_base + mu * (R + C);
				const double y = RealSHValue(sh_values, l, m);
			double dy[3];
			for (int a = 0; a < 3; ++a)
				dy[a] = RealSHDer(sh_ders, l, m, a);

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
			for (int xi = 0; xi < R; ++xi) {
				const double rb_val = rb_vals[radial_base + xi];
				const double rb_der = rb_ders[deriv_base + xi];
				double grad = adj_value * type_scale * rb_val * y;
				for (int a = 0; a < 3; ++a)
					grad += adj_der[a] * type_scale * (rb_der * rvec[a] * inv_r * y + rb_val * dy[a]);
				grad_out[radial_offset + xi] += grad;
			}

			double center_grad = adj_value * outer_type_coeff * dot_val * y;
			double outer_grad = adj_value * center_type_coeff * dot_val * y;
			double sigma_grad = adj_value * type_scale * dot_s * y;
			double shift_grad = adj_value * type_scale * dot_ss * y;
			for (int a = 0; a < 3; ++a) {
				const double base_coord = dot_der * rvec[a] * inv_r * y + dot_val * dy[a];
				const double sigma_coord = dot_coord_s * rvec[a] * inv_r * y + dot_s * dy[a];
				const double shift_coord = dot_coord_ss * rvec[a] * inv_r * y + dot_ss * dy[a];
				center_grad += adj_der[a] * outer_type_coeff * base_coord;
				outer_grad += adj_der[a] * center_type_coeff * base_coord;
				sigma_grad += adj_der[a] * type_scale * sigma_coord;
				shift_grad += adj_der[a] * type_scale * shift_coord;
			}

			const int sigma_coeff_offset = C + 2 * C * C * scaling_block + type_central * C + type_outer;
			grad_out[shared_type_offset + type_central] += center_grad;
			grad_out[shared_type_offset + type_outer] += outer_grad;
			grad_out[sigma_coeff_offset] += sigma_grad;
			grad_out[sigma_coeff_offset + C * C] += shift_grad;
		}
	}
}
