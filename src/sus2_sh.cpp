#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sus2sh {

constexpr double kPi = 3.141592653589793238462643383279502884;

struct Vec3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

Vec3 operator+(const Vec3& a, const Vec3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 operator-(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 operator*(double s, const Vec3& a) { return {s * a.x, s * a.y, s * a.z}; }
double Dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
double Norm(const Vec3& a) { return std::sqrt(Dot(a, a)); }

struct Atom {
  int type = 0;
  Vec3 pos;
  Vec3 force;
};

struct Config {
  std::array<Vec3, 3> lattice{};
  std::vector<Atom> atoms;
  double energy = 0.0;
  bool has_stress = false;
  std::array<std::array<double, 3>, 3> stress{};
};

struct Options {
  std::string command;
  std::string train_cfg;
  std::string model_out = "sus2-sh.model.json";
  int l_max = 3;
  int k_max = 3;
  int body_order = 5;
  int radial_basis_size = 10;
  double cutoff = 7.5;
  std::string radial_basis_type = "RBChebyshev_sss";
  int max_configs = 256;
  int max_iter = 1000;
  double ridge = 1.0e-10;
  double fd_step = 1.0e-5;
  double energy_weight = 1.0;
  double force_weight = 1.0;
  double stress_weight = 0.0;
};

std::vector<std::string> Split(const std::string& s) {
  std::istringstream iss(s);
  std::vector<std::string> out;
  std::string tok;
  while (iss >> tok) out.push_back(tok);
  return out;
}

bool StartsWith(const std::string& s, const std::string& prefix) {
  return s.rfind(prefix, 0) == 0;
}

std::vector<Config> ReadCfgs(const std::string& path, int max_configs) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open cfg file: " + path);

  std::vector<Config> cfgs;
  std::string line;
  while (std::getline(in, line)) {
    if (line.find("BEGIN_CFG") == std::string::npos) continue;
    Config cfg;
    int size = -1;
    while (std::getline(in, line)) {
      if (line.find("END_CFG") != std::string::npos) break;
      auto toks = Split(line);
      if (toks.empty()) continue;
      if (toks[0] == "Size") {
        std::getline(in, line);
        size = std::stoi(Split(line).at(0));
        cfg.atoms.resize(size);
      } else if (toks[0] == "Supercell") {
        for (int i = 0; i < 3; ++i) {
          std::getline(in, line);
          auto v = Split(line);
          cfg.lattice[i] = {std::stod(v.at(0)), std::stod(v.at(1)), std::stod(v.at(2))};
        }
      } else if (StartsWith(toks[0], "AtomData")) {
        if (size < 0) throw std::runtime_error("AtomData before Size in cfg");
        for (int i = 0; i < size; ++i) {
          std::getline(in, line);
          auto v = Split(line);
          if (v.size() < 8) throw std::runtime_error("unsupported AtomData row: " + line);
          cfg.atoms[i].type = std::stoi(v[1]);
          cfg.atoms[i].pos = {std::stod(v[2]), std::stod(v[3]), std::stod(v[4])};
          cfg.atoms[i].force = {std::stod(v[5]), std::stod(v[6]), std::stod(v[7])};
        }
      } else if (toks[0] == "Energy") {
        std::getline(in, line);
        cfg.energy = std::stod(Split(line).at(0));
      } else if (toks[0] == "PlusStress:" || toks[0] == "Virial:" || toks[0] == "Stress:") {
        auto keys = toks;
        keys.erase(keys.begin());
        std::getline(in, line);
        auto vals = Split(line);
        if (vals.size() < keys.size()) throw std::runtime_error("broken stress row");
        cfg.has_stress = true;
        for (std::size_t i = 0; i < keys.size(); ++i) {
          const double value = std::stod(vals[i]);
          const std::string& key = keys[i];
          if (key == "xx") cfg.stress[0][0] = value;
          else if (key == "yy") cfg.stress[1][1] = value;
          else if (key == "zz") cfg.stress[2][2] = value;
          else if (key == "xy") cfg.stress[0][1] = cfg.stress[1][0] = value;
          else if (key == "xz") cfg.stress[0][2] = cfg.stress[2][0] = value;
          else if (key == "yz") cfg.stress[1][2] = cfg.stress[2][1] = value;
        }
        if (toks[0] == "Stress:") {
          for (auto& row : cfg.stress)
            for (double& v : row) v *= -1.0;
        }
      }
    }
    if (!cfg.atoms.empty()) cfgs.push_back(std::move(cfg));
    if (max_configs > 0 && static_cast<int>(cfgs.size()) >= max_configs) break;
  }
  return cfgs;
}

double Fact(int n) {
  if (n < 0) return 0.0;
  return std::tgamma(static_cast<double>(n) + 1.0);
}

bool Triangle(int a, int b, int c) {
  return std::abs(a - b) <= c && c <= a + b;
}

double Wigner3j(int j1, int j2, int j3, int m1, int m2, int m3) {
  if (m1 + m2 + m3 != 0) return 0.0;
  if (!Triangle(j1, j2, j3)) return 0.0;
  if (std::abs(m1) > j1 || std::abs(m2) > j2 || std::abs(m3) > j3) return 0.0;

  const double delta = static_cast<double>(Fact(j1 + j2 - j3)) *
                       Fact(j1 - j2 + j3) * Fact(-j1 + j2 + j3) /
                       Fact(j1 + j2 + j3 + 1);
  double pref = ((j1 - j2 - m3) % 2 == 0 ? 1.0 : -1.0) * std::sqrt(delta);
  pref *= std::sqrt(static_cast<double>(Fact(j1 + m1)) * Fact(j1 - m1) *
                    Fact(j2 + m2) * Fact(j2 - m2) * Fact(j3 + m3) *
                    Fact(j3 - m3));

  double sum = 0.0;
  for (int z = 0; z <= j1 + j2 + j3 + 1; ++z) {
    const int a = j1 + j2 - j3 - z;
    const int b = j1 - m1 - z;
    const int c = j2 + m2 - z;
    const int d = j3 - j2 + m1 + z;
    const int e = j3 - j1 - m2 + z;
    if (a < 0 || b < 0 || c < 0 || d < 0 || e < 0) continue;
    const double denom = static_cast<double>(Fact(z)) * Fact(a) * Fact(b) *
                         Fact(c) * Fact(d) * Fact(e);
    sum += (z % 2 == 0 ? 1.0 : -1.0) / denom;
  }
  return pref * sum;
}

double ClebschGordan(int j1, int m1, int j2, int m2, int j, int m) {
  if (m1 + m2 != m) return 0.0;
  const double phase = ((j1 - j2 + m) % 2 == 0) ? 1.0 : -1.0;
  return phase * std::sqrt(2.0 * j + 1.0) * Wigner3j(j1, j2, j, m1, m2, -m);
}

double AssocLegendre(int l, int m, double x) {
  const double s = std::sqrt(std::max(0.0, 1.0 - x * x));
  if (l == 0 && m == 0) return 1.0;
  if (l == 1 && m == 0) return x;
  if (l == 1 && m == 1) return -s;
  if (l == 2 && m == 0) return 0.5 * (3.0 * x * x - 1.0);
  if (l == 2 && m == 1) return -3.0 * x * s;
  if (l == 2 && m == 2) return 3.0 * s * s;
  if (l == 3 && m == 0) return 0.5 * (5.0 * x * x * x - 3.0 * x);
  if (l == 3 && m == 1) return -1.5 * (5.0 * x * x - 1.0) * s;
  if (l == 3 && m == 2) return 15.0 * x * s * s;
  if (l == 3 && m == 3) return -15.0 * s * s * s;
  throw std::runtime_error("AssocLegendre only implemented for l<=3");
}

std::complex<double> ComplexYlm(int l, int m, const Vec3& rvec) {
  const double r = Norm(rvec);
  if (r <= 0.0) return {0.0, 0.0};
  if (m < 0) {
    const auto yp = ComplexYlm(l, -m, rvec);
    return ((-m) % 2 == 0 ? 1.0 : -1.0) * std::conj(yp);
  }
  const double ct = std::max(-1.0, std::min(1.0, rvec.z / r));
  const double phi = std::atan2(rvec.y, rvec.x);
  const double norm = std::sqrt((2.0 * l + 1.0) / (4.0 * kPi) *
                                static_cast<double>(Fact(l - m)) / Fact(l + m));
  const double p = AssocLegendre(l, m, ct);
  return norm * p * std::complex<double>(std::cos(m * phi), std::sin(m * phi));
}

using Cplx = std::complex<double>;

struct Vec3C {
  Cplx x = {0.0, 0.0};
  Cplx y = {0.0, 0.0};
  Cplx z = {0.0, 0.0};
};

struct YlmGrad {
  Cplx value = {0.0, 0.0};
  Vec3C grad;
};

struct PolyTerm {
  Cplx c;
  int ax = 0;
  int ay = 0;
  int az = 0;
};

double PowInt(double x, int n) {
  double out = 1.0;
  for (int i = 0; i < n; ++i) out *= x;
  return out;
}

std::vector<PolyTerm> PositiveYlmPolynomial(int l, int m) {
  const double inv_sqrt_pi = 1.0 / std::sqrt(kPi);
  if (l == 0 && m == 0) return {{{0.5 * inv_sqrt_pi, 0.0}, 0, 0, 0}};

  if (l == 1 && m == 0) return {{{std::sqrt(3.0 / (4.0 * kPi)), 0.0}, 0, 0, 1}};
  if (l == 1 && m == 1) {
    const double c = -std::sqrt(3.0 / (8.0 * kPi));
    return {{{c, 0.0}, 1, 0, 0}, {{0.0, c}, 0, 1, 0}};
  }

  if (l == 2 && m == 0) {
    const double c = std::sqrt(5.0 / (16.0 * kPi));
    return {{{-c, 0.0}, 2, 0, 0}, {{-c, 0.0}, 0, 2, 0}, {{2.0 * c, 0.0}, 0, 0, 2}};
  }
  if (l == 2 && m == 1) {
    const double c = -std::sqrt(15.0 / (8.0 * kPi));
    return {{{c, 0.0}, 1, 0, 1}, {{0.0, c}, 0, 1, 1}};
  }
  if (l == 2 && m == 2) {
    const double c = std::sqrt(15.0 / (32.0 * kPi));
    return {{{c, 0.0}, 2, 0, 0}, {{-c, 0.0}, 0, 2, 0}, {{0.0, 2.0 * c}, 1, 1, 0}};
  }

  if (l == 3 && m == 0) {
    const double c = std::sqrt(7.0 / (16.0 * kPi));
    return {{{-3.0 * c, 0.0}, 2, 0, 1},
            {{-3.0 * c, 0.0}, 0, 2, 1},
            {{2.0 * c, 0.0}, 0, 0, 3}};
  }
  if (l == 3 && m == 1) {
    const double c = -std::sqrt(21.0 / (64.0 * kPi));
    return {{{-c, 0.0}, 3, 0, 0},
            {{-c, 0.0}, 1, 2, 0},
            {{4.0 * c, 0.0}, 1, 0, 2},
            {{0.0, -c}, 2, 1, 0},
            {{0.0, -c}, 0, 3, 0},
            {{0.0, 4.0 * c}, 0, 1, 2}};
  }
  if (l == 3 && m == 2) {
    const double c = std::sqrt(105.0 / (32.0 * kPi));
    return {{{c, 0.0}, 2, 0, 1}, {{-c, 0.0}, 0, 2, 1}, {{0.0, 2.0 * c}, 1, 1, 1}};
  }
  if (l == 3 && m == 3) {
    const double c = -std::sqrt(35.0 / (64.0 * kPi));
    return {{{c, 0.0}, 3, 0, 0},
            {{0.0, 3.0 * c}, 2, 1, 0},
            {{-3.0 * c, 0.0}, 1, 2, 0},
            {{0.0, -c}, 0, 3, 0}};
  }
  throw std::runtime_error("Y_lm polynomial only implemented for l<=3");
}

YlmGrad ComplexYlmWithGrad(int l, int m, const Vec3& rvec) {
  if (m < 0) {
    auto yp = ComplexYlmWithGrad(l, -m, rvec);
    const double phase = ((-m) % 2 == 0) ? 1.0 : -1.0;
    yp.value = phase * std::conj(yp.value);
    yp.grad.x = phase * std::conj(yp.grad.x);
    yp.grad.y = phase * std::conj(yp.grad.y);
    yp.grad.z = phase * std::conj(yp.grad.z);
    return yp;
  }

  const double r = Norm(rvec);
  if (r <= 1.0e-14) return {};
  const auto terms = PositiveYlmPolynomial(l, m);
  Cplx p = {0.0, 0.0};
  Cplx px = {0.0, 0.0};
  Cplx py = {0.0, 0.0};
  Cplx pz = {0.0, 0.0};
  for (const auto& t : terms) {
    const double mon = PowInt(rvec.x, t.ax) * PowInt(rvec.y, t.ay) * PowInt(rvec.z, t.az);
    p += t.c * mon;
    if (t.ax > 0) {
      px += t.c * static_cast<double>(t.ax) * PowInt(rvec.x, t.ax - 1) *
            PowInt(rvec.y, t.ay) * PowInt(rvec.z, t.az);
    }
    if (t.ay > 0) {
      py += t.c * static_cast<double>(t.ay) * PowInt(rvec.x, t.ax) *
            PowInt(rvec.y, t.ay - 1) * PowInt(rvec.z, t.az);
    }
    if (t.az > 0) {
      pz += t.c * static_cast<double>(t.az) * PowInt(rvec.x, t.ax) *
            PowInt(rvec.y, t.ay) * PowInt(rvec.z, t.az - 1);
    }
  }

  const double inv_rl = (l == 0) ? 1.0 : 1.0 / std::pow(r, l);
  const double inv_r2 = 1.0 / (r * r);
  YlmGrad out;
  out.value = p * inv_rl;
  if (l == 0) {
    out.grad = {px, py, pz};
  } else {
    out.grad.x = px * inv_rl - static_cast<double>(l) * p * inv_rl * rvec.x * inv_r2;
    out.grad.y = py * inv_rl - static_cast<double>(l) * p * inv_rl * rvec.y * inv_r2;
    out.grad.z = pz * inv_rl - static_cast<double>(l) * p * inv_rl * rvec.z * inv_r2;
  }
  return out;
}

struct RadialValues {
  std::vector<double> values;
  std::vector<double> derivs;
};

RadialValues RBChebyshevSss(double r, int size, double cutoff, double scal, double shift) {
  RadialValues out;
  out.values.assign(size, 0.0);
  out.derivs.assign(size, 0.0);
  if (r >= cutoff || size <= 0) return out;

  const double x = 0.5 * scal * (r - shift);
  const double denom = x * x + 1.0;
  const double sq = std::sqrt(denom);
  const double ksi = x / sq;
  const double dksi = 0.5 * scal / (denom * sq);
  const double dr = r - cutoff;
  const double f = dr * dr;
  const double df = 2.0 * dr;

  out.values[0] = f;
  out.derivs[0] = df;
  if (size == 1) return out;
  out.values[1] = ksi * f;
  out.derivs[1] = dksi * f + ksi * df;
  for (int i = 2; i < size; ++i) {
    out.values[i] = 2.0 * ksi * out.values[i - 1] - out.values[i - 2];
    out.derivs[i] =
        2.0 * (dksi * out.values[i - 1] + ksi * out.derivs[i - 1]) - out.derivs[i - 2];
  }
  return out;
}

struct Channel {
  int l = 0;
  int k = 0;
};

bool ChannelGreaterEq(const Channel& a, const Channel& b) {
  if (a.l != b.l) return a.l > b.l;
  return a.k >= b.k;
}

struct Coupling {
  int order = 0;
  std::array<int, 4> q{};
  int intermediate_l = 0;
};

struct Topology {
  std::vector<Channel> channels;
  std::vector<Coupling> couplings;
};

std::vector<int> AllowedL(int l1, int l2) {
  std::vector<int> out;
  for (int l = std::abs(l1 - l2); l <= l1 + l2; ++l) out.push_back(l);
  return out;
}

Topology BuildTopology(int l_max, int k_max, int body_order) {
  if (l_max > 3) throw std::runtime_error("this prototype supports l_max<=3");
  Topology top;
  for (int l = l_max; l >= 0; --l) {
    for (int k = k_max; k >= 1; --k) top.channels.push_back({l, k});
  }
  const int n = static_cast<int>(top.channels.size());
  const int max_factors = std::max(1, body_order - 1);

  if (max_factors >= 1) {
    for (int i = 0; i < n; ++i) {
      if (top.channels[i].l == 0) top.couplings.push_back({1, {i, -1, -1, -1}, 0});
    }
  }
  if (max_factors >= 2) {
    for (int i = 0; i < n; ++i) {
      for (int j = i; j < n; ++j) {
        const auto& a = top.channels[i];
        const auto& b = top.channels[j];
        if (!ChannelGreaterEq(a, b)) continue;
        if ((a.l + b.l) % 2 != 0) continue;
        if (a.l == b.l) top.couplings.push_back({2, {i, j, -1, -1}, 0});
      }
    }
  }
  if (max_factors >= 3) {
    for (int i = 0; i < n; ++i) {
      for (int j = i; j < n; ++j) {
        for (int k = j; k < n; ++k) {
          const auto& a = top.channels[i];
          const auto& b = top.channels[j];
          const auto& c = top.channels[k];
          if ((a.l + b.l + c.l) % 2 != 0) continue;
          if (Triangle(a.l, b.l, c.l)) {
            top.couplings.push_back({3, {i, j, k, -1}, c.l});
          }
        }
      }
    }
  }
  if (max_factors >= 4) {
    for (int i = 0; i < n; ++i) {
      for (int j = i; j < n; ++j) {
        for (int k = j; k < n; ++k) {
          for (int h = k; h < n; ++h) {
            const auto& a = top.channels[i];
            const auto& b = top.channels[j];
            const auto& c = top.channels[k];
            const auto& d = top.channels[h];
            if ((a.l + b.l + c.l + d.l) % 2 != 0) continue;
            const auto left = AllowedL(a.l, b.l);
            const auto right = AllowedL(c.l, d.l);
            for (int L : left) {
              if (std::find(right.begin(), right.end(), L) != right.end()) {
                top.couplings.push_back({4, {i, j, k, h}, L});
              }
            }
          }
        }
      }
    }
  }
  return top;
}

struct PairCGKey {
  int l1 = 0;
  int l2 = 0;
  int l = 0;
  bool operator<(const PairCGKey& other) const {
    if (l1 != other.l1) return l1 < other.l1;
    if (l2 != other.l2) return l2 < other.l2;
    return l < other.l;
  }
};

struct CGTerm {
  int m1 = 0;
  int m2 = 0;
  int m = 0;
  double c = 0.0;
};

class Evaluator {
 public:
  Evaluator(Options opts, Topology top)
      : opts_(std::move(opts)), top_(std::move(top)), feature_count_(top_.couplings.size() + 1) {
    BuildCgCache();
  }

  int FeatureCount() const { return feature_count_; }
  const Topology& topology() const { return top_; }

  std::vector<double> Features(const Config& cfg) const {
    std::vector<double> total(feature_count_, 0.0);
    total[0] = 1.0;
    for (std::size_t center = 0; center < cfg.atoms.size(); ++center) {
      const auto basic = BasicMoments(cfg, static_cast<int>(center));
      std::vector<double> site(feature_count_, 0.0);
      site[0] = 1.0;
      for (std::size_t i = 0; i < top_.couplings.size(); ++i) {
        site[i + 1] = EvalCoupling(top_.couplings[i], basic).real();
      }
      for (int p = 1; p < feature_count_; ++p) total[p] += site[p];
    }
    const double inv_n = 1.0 / std::max<std::size_t>(1, cfg.atoms.size());
    for (double& v : total) v *= inv_n;
    return total;
  }

  double EnergyForcesStress(const Config& cfg, const std::vector<double>& coeffs,
                            std::vector<Vec3>* forces,
                            std::array<std::array<double, 3>, 3>* stress) const {
    if (static_cast<int>(coeffs.size()) != feature_count_) {
      throw std::runtime_error("coefficient count does not match topology");
    }
    if (forces != nullptr) forces->assign(cfg.atoms.size(), Vec3{});
    if (stress != nullptr) {
      for (auto& row : *stress)
        for (double& v : row) v = 0.0;
    }
    double total_energy = 0.0;

    for (std::size_t center = 0; center < cfg.atoms.size(); ++center) {
      const auto basic = BasicMoments(cfg, static_cast<int>(center));
      std::vector<VecC> adj;
      const double site_e = SiteEnergyAndBasicAdjoints(basic, coeffs, &adj);
      total_energy += site_e;
      if (forces != nullptr || stress != nullptr) {
        AccumulateEfsFromBasicAdjoints(cfg, static_cast<int>(center), adj, forces, stress);
      }
    }
    return total_energy;
  }

  double EnergyAndForces(const Config& cfg, const std::vector<double>& coeffs,
                         std::vector<Vec3>* forces) const {
    return EnergyForcesStress(cfg, coeffs, forces, nullptr);
  }

  std::vector<std::vector<double>> ForceDesignRows(const Config& cfg) const {
    const int rows = static_cast<int>(cfg.atoms.size()) * 3;
    std::vector<std::vector<double>> out(rows, std::vector<double>(feature_count_, 0.0));
    std::vector<double> coeffs(feature_count_, 0.0);
    std::vector<Vec3> forces;
    for (int p = 1; p < feature_count_; ++p) {
      coeffs[p] = 1.0;
      EnergyForcesStress(cfg, coeffs, &forces, nullptr);
      for (std::size_t atom = 0; atom < cfg.atoms.size(); ++atom) {
        out[3 * atom + 0][p] = forces[atom].x;
        out[3 * atom + 1][p] = forces[atom].y;
        out[3 * atom + 2][p] = forces[atom].z;
      }
      coeffs[p] = 0.0;
    }
    return out;
  }

  std::vector<std::vector<double>> StressDesignRows(const Config& cfg) const {
    std::vector<std::vector<double>> out(6, std::vector<double>(feature_count_, 0.0));
    std::vector<double> coeffs(feature_count_, 0.0);
    std::array<std::array<double, 3>, 3> stress{};
    for (int p = 1; p < feature_count_; ++p) {
      coeffs[p] = 1.0;
      EnergyForcesStress(cfg, coeffs, nullptr, &stress);
      out[0][p] = stress[0][0];
      out[1][p] = stress[1][1];
      out[2][p] = stress[2][2];
      out[3][p] = stress[1][2];
      out[4][p] = stress[0][2];
      out[5][p] = stress[0][1];
      coeffs[p] = 0.0;
    }
    return out;
  }

  const std::vector<std::pair<double, double>>& RadialScaling() const { return radial_scaling_; }

 private:
  using VecC = std::vector<Cplx>;

  std::vector<VecC> BasicMoments(const Config& cfg, int center) const {
    std::vector<VecC> out(top_.channels.size());
    for (std::size_t q = 0; q < top_.channels.size(); ++q) {
      const int l = top_.channels[q].l;
      out[q].assign(2 * l + 1, Cplx{0.0, 0.0});
    }

    double min_len = 1.0e100;
    for (const auto& a : cfg.lattice) min_len = std::min(min_len, Norm(a));
    const int image_range = std::max(1, static_cast<int>(std::ceil(opts_.cutoff / min_len)) + 1);
    const Vec3 pos_i = cfg.atoms[center].pos;

    for (std::size_t j = 0; j < cfg.atoms.size(); ++j) {
      for (int nx = -image_range; nx <= image_range; ++nx) {
        for (int ny = -image_range; ny <= image_range; ++ny) {
          for (int nz = -image_range; nz <= image_range; ++nz) {
            if (static_cast<int>(j) == center && nx == 0 && ny == 0 && nz == 0) continue;
            Vec3 disp = cfg.atoms[j].pos - pos_i;
            disp = disp + static_cast<double>(nx) * cfg.lattice[0] +
                   static_cast<double>(ny) * cfg.lattice[1] +
                   static_cast<double>(nz) * cfg.lattice[2];
            const double r = Norm(disp);
            if (r <= 1.0e-12 || r >= opts_.cutoff) continue;
            for (std::size_t q = 0; q < top_.channels.size(); ++q) {
              const int l = top_.channels[q].l;
              const int k = top_.channels[q].k;
              const double scal = radial_scaling_.at(k - 1).first;
              const double shift = radial_scaling_.at(k - 1).second;
              const auto rb = RBChebyshevSss(r, opts_.radial_basis_size, opts_.cutoff, scal, shift);
              const double radial = rb.values[std::min(k - 1, opts_.radial_basis_size - 1)];
              for (int m = -l; m <= l; ++m) {
                out[q][m + l] += radial * ComplexYlmWithGrad(l, m, disp).value;
              }
            }
          }
        }
      }
    }
    return out;
  }

  void AccumulateEfsFromBasicAdjoints(const Config& cfg, int center,
                                      const std::vector<VecC>& adj,
                                      std::vector<Vec3>* forces,
                                      std::array<std::array<double, 3>, 3>* stress) const {
    double min_len = 1.0e100;
    for (const auto& a : cfg.lattice) min_len = std::min(min_len, Norm(a));
    const int image_range = std::max(1, static_cast<int>(std::ceil(opts_.cutoff / min_len)) + 1);
    const Vec3 pos_i = cfg.atoms[center].pos;

    for (std::size_t j = 0; j < cfg.atoms.size(); ++j) {
      for (int nx = -image_range; nx <= image_range; ++nx) {
        for (int ny = -image_range; ny <= image_range; ++ny) {
          for (int nz = -image_range; nz <= image_range; ++nz) {
            if (static_cast<int>(j) == center && nx == 0 && ny == 0 && nz == 0) continue;
            Vec3 disp = cfg.atoms[j].pos - pos_i;
            disp = disp + static_cast<double>(nx) * cfg.lattice[0] +
                   static_cast<double>(ny) * cfg.lattice[1] +
                   static_cast<double>(nz) * cfg.lattice[2];
            const double r = Norm(disp);
            if (r <= 1.0e-12 || r >= opts_.cutoff) continue;
            if (static_cast<int>(j) == center) continue;

            Vec3 grad_e_disp{};
            const Vec3 nvec = (1.0 / r) * disp;
            for (std::size_t q = 0; q < top_.channels.size(); ++q) {
              const int l = top_.channels[q].l;
              const int k = top_.channels[q].k;
              const double scal = radial_scaling_.at(k - 1).first;
              const double shift = radial_scaling_.at(k - 1).second;
              const auto rb = RBChebyshevSss(r, opts_.radial_basis_size, opts_.cutoff, scal, shift);
              const int rb_index = std::min(k - 1, opts_.radial_basis_size - 1);
              const double radial = rb.values[rb_index];
              const double radial_der = rb.derivs[rb_index];
              for (int m = -l; m <= l; ++m) {
                const auto yg = ComplexYlmWithGrad(l, m, disp);
                const Cplx a = adj[q][m + l];
                const Cplx dx = radial_der * nvec.x * yg.value + radial * yg.grad.x;
                const Cplx dy = radial_der * nvec.y * yg.value + radial * yg.grad.y;
                const Cplx dz = radial_der * nvec.z * yg.value + radial * yg.grad.z;
                grad_e_disp.x += (a * dx).real();
                grad_e_disp.y += (a * dy).real();
                grad_e_disp.z += (a * dz).real();
              }
            }
            if (forces != nullptr) {
              (*forces)[j].x -= grad_e_disp.x;
              (*forces)[j].y -= grad_e_disp.y;
              (*forces)[j].z -= grad_e_disp.z;
              (*forces)[center].x += grad_e_disp.x;
              (*forces)[center].y += grad_e_disp.y;
              (*forces)[center].z += grad_e_disp.z;
            }
            if (stress != nullptr) {
              const double g[3] = {grad_e_disp.x, grad_e_disp.y, grad_e_disp.z};
              const double rv[3] = {disp.x, disp.y, disp.z};
              for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                  (*stress)[a][b] -= g[a] * rv[b];
                }
              }
            }
          }
        }
      }
    }
  }

  VecC CoupleVectors(const VecC& a, int l1, const VecC& b, int l2, int l) const {
    VecC out(2 * l + 1, Cplx{0.0, 0.0});
    const auto& terms = cg_cache_.at({l1, l2, l});
    for (const auto& t : terms) {
      out[t.m + l] += t.c * a[t.m1 + l1] * b[t.m2 + l2];
    }
    return out;
  }

  Cplx EvalCoupling(const Coupling& c, const std::vector<VecC>& basic) const {
    if (c.order == 1) {
      const auto& ch = top_.channels[c.q[0]];
      return basic[c.q[0]][ch.l];
    }
    if (c.order == 2) {
      const auto& a = top_.channels[c.q[0]];
      const auto& b = top_.channels[c.q[1]];
      return CoupleVectors(basic[c.q[0]], a.l, basic[c.q[1]], b.l, 0)[0];
    }
    if (c.order == 3) {
      const auto& a = top_.channels[c.q[0]];
      const auto& b = top_.channels[c.q[1]];
      const auto& d = top_.channels[c.q[2]];
      const auto ab = CoupleVectors(basic[c.q[0]], a.l, basic[c.q[1]], b.l, c.intermediate_l);
      return CoupleVectors(ab, c.intermediate_l, basic[c.q[2]], d.l, 0)[0];
    }
    if (c.order == 4) {
      const auto& a = top_.channels[c.q[0]];
      const auto& b = top_.channels[c.q[1]];
      const auto& d = top_.channels[c.q[2]];
      const auto& e = top_.channels[c.q[3]];
      const auto ab = CoupleVectors(basic[c.q[0]], a.l, basic[c.q[1]], b.l, c.intermediate_l);
      const auto de = CoupleVectors(basic[c.q[2]], d.l, basic[c.q[3]], e.l, c.intermediate_l);
      return CoupleVectors(ab, c.intermediate_l, de, c.intermediate_l, 0)[0];
    }
    return {0.0, 0.0};
  }

  void CoupleBackward(const VecC& a, int l1, const VecC& b, int l2, int l,
                      const VecC& adj_out, VecC* adj_a, VecC* adj_b) const {
    const auto& terms = cg_cache_.at({l1, l2, l});
    for (const auto& t : terms) {
      const Cplx seed = adj_out[t.m + l] * t.c;
      (*adj_a)[t.m1 + l1] += seed * b[t.m2 + l2];
      (*adj_b)[t.m2 + l2] += seed * a[t.m1 + l1];
    }
  }

  double SiteEnergyAndBasicAdjoints(const std::vector<VecC>& basic,
                                    const std::vector<double>& coeffs,
                                    std::vector<VecC>* adj_out) const {
    double energy = coeffs[0];
    adj_out->clear();
    adj_out->resize(top_.channels.size());
    for (std::size_t q = 0; q < top_.channels.size(); ++q) {
      const int l = top_.channels[q].l;
      (*adj_out)[q].assign(2 * l + 1, Cplx{0.0, 0.0});
    }

    for (std::size_t p = 0; p < top_.couplings.size(); ++p) {
      const auto& c = top_.couplings[p];
      const double w = coeffs[p + 1];
      if (w == 0.0) continue;
      const Cplx value = EvalCoupling(c, basic);
      energy += w * value.real();

      if (c.order == 1) {
        const auto& ch = top_.channels[c.q[0]];
        (*adj_out)[c.q[0]][ch.l] += w;
      } else if (c.order == 2) {
        const auto& a = top_.channels[c.q[0]];
        const auto& b = top_.channels[c.q[1]];
        VecC adj_scalar(1, Cplx{w, 0.0});
        CoupleBackward(basic[c.q[0]], a.l, basic[c.q[1]], b.l, 0, adj_scalar,
                       &(*adj_out)[c.q[0]], &(*adj_out)[c.q[1]]);
      } else if (c.order == 3) {
        const auto& a = top_.channels[c.q[0]];
        const auto& b = top_.channels[c.q[1]];
        const auto& d = top_.channels[c.q[2]];
        const auto ab = CoupleVectors(basic[c.q[0]], a.l, basic[c.q[1]], b.l, c.intermediate_l);
        VecC adj_ab(2 * c.intermediate_l + 1, Cplx{0.0, 0.0});
        VecC adj_d(2 * d.l + 1, Cplx{0.0, 0.0});
        VecC adj_scalar(1, Cplx{w, 0.0});
        CoupleBackward(ab, c.intermediate_l, basic[c.q[2]], d.l, 0, adj_scalar, &adj_ab, &adj_d);
        CoupleBackward(basic[c.q[0]], a.l, basic[c.q[1]], b.l, c.intermediate_l, adj_ab,
                       &(*adj_out)[c.q[0]], &(*adj_out)[c.q[1]]);
        for (std::size_t i = 0; i < adj_d.size(); ++i) (*adj_out)[c.q[2]][i] += adj_d[i];
      } else if (c.order == 4) {
        const auto& a = top_.channels[c.q[0]];
        const auto& b = top_.channels[c.q[1]];
        const auto& d = top_.channels[c.q[2]];
        const auto& e = top_.channels[c.q[3]];
        const auto ab = CoupleVectors(basic[c.q[0]], a.l, basic[c.q[1]], b.l, c.intermediate_l);
        const auto de = CoupleVectors(basic[c.q[2]], d.l, basic[c.q[3]], e.l, c.intermediate_l);
        VecC adj_ab(2 * c.intermediate_l + 1, Cplx{0.0, 0.0});
        VecC adj_de(2 * c.intermediate_l + 1, Cplx{0.0, 0.0});
        VecC adj_scalar(1, Cplx{w, 0.0});
        CoupleBackward(ab, c.intermediate_l, de, c.intermediate_l, 0, adj_scalar, &adj_ab, &adj_de);
        CoupleBackward(basic[c.q[0]], a.l, basic[c.q[1]], b.l, c.intermediate_l, adj_ab,
                       &(*adj_out)[c.q[0]], &(*adj_out)[c.q[1]]);
        CoupleBackward(basic[c.q[2]], d.l, basic[c.q[3]], e.l, c.intermediate_l, adj_de,
                       &(*adj_out)[c.q[2]], &(*adj_out)[c.q[3]]);
      }
    }
    return energy;
  }

  void BuildCgCache() {
    for (int l1 = 0; l1 <= 6; ++l1) {
      for (int l2 = 0; l2 <= 6; ++l2) {
        for (int l = std::abs(l1 - l2); l <= l1 + l2; ++l) {
          std::vector<CGTerm> terms;
          for (int m1 = -l1; m1 <= l1; ++m1) {
            for (int m2 = -l2; m2 <= l2; ++m2) {
              const int m = m1 + m2;
              if (std::abs(m) > l) continue;
              const double c = ClebschGordan(l1, m1, l2, m2, l, m);
              if (std::abs(c) > 1.0e-14) terms.push_back({m1, m2, m, c});
            }
          }
          cg_cache_[{l1, l2, l}] = std::move(terms);
        }
      }
    }
    radial_scaling_.clear();
    for (int k = 0; k < opts_.k_max; ++k) {
      const double scal = 2.0 + 0.25 * k;
      const double shift = 0.5 * opts_.cutoff;
      radial_scaling_.push_back({scal, shift});
    }
  }

  Options opts_;
  Topology top_;
  int feature_count_ = 0;
  std::map<PairCGKey, std::vector<CGTerm>> cg_cache_;
  std::vector<std::pair<double, double>> radial_scaling_;
};

double DotVec(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return s;
}

std::vector<double> SolveLinearSystem(std::vector<std::vector<double>> a, std::vector<double> b) {
  const int n = static_cast<int>(b.size());
  for (int col = 0; col < n; ++col) {
    int pivot = col;
    double best = std::abs(a[col][col]);
    for (int row = col + 1; row < n; ++row) {
      const double v = std::abs(a[row][col]);
      if (v > best) {
        best = v;
        pivot = row;
      }
    }
    if (best < 1.0e-20) {
      throw std::runtime_error("linear solve failed: singular normal matrix");
    }
    if (pivot != col) {
      std::swap(a[pivot], a[col]);
      std::swap(b[pivot], b[col]);
    }
    const double diag = a[col][col];
    for (int j = col; j < n; ++j) a[col][j] /= diag;
    b[col] /= diag;
    for (int row = 0; row < n; ++row) {
      if (row == col) continue;
      const double f = a[row][col];
      if (f == 0.0) continue;
      for (int j = col; j < n; ++j) a[row][j] -= f * a[col][j];
      b[row] -= f * b[col];
    }
  }
  return b;
}

void SaveModel(const std::string& path, const Options& opts, const Topology& top,
               const Evaluator& eval, const std::vector<double>& coeffs,
               double energy_mae, double energy_rmse, double force_mae, double force_rmse,
               double stress_mae, double stress_rmse, int stress_rows,
               const std::vector<int>& species) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write model: " + path);
  out << std::setprecision(17);
  out << "{\n";
  out << "  \"format\": \"SUS2-SH-prototype\",\n";
  out << "  \"version\": 1,\n";
  out << "  \"l_max\": " << opts.l_max << ",\n";
  out << "  \"k_max\": " << opts.k_max << ",\n";
  out << "  \"body_order\": " << opts.body_order << ",\n";
  out << "  \"max_factors\": " << opts.body_order - 1 << ",\n";
  out << "  \"parity\": \"even_O3\",\n";
  out << "  \"radial_basis_type\": \"" << opts.radial_basis_type << "\",\n";
  out << "  \"radial_basis_size\": " << opts.radial_basis_size << ",\n";
  out << "  \"cutoff\": " << opts.cutoff << ",\n";
  out << "  \"energy_weight\": " << opts.energy_weight << ",\n";
  out << "  \"force_weight\": " << opts.force_weight << ",\n";
  out << "  \"stress_weight\": " << opts.stress_weight << ",\n";
  out << "  \"max_configs\": " << opts.max_configs << ",\n";
  out << "  \"scalar_count_without_bias\": " << top.couplings.size() << ",\n";
  out << "  \"energy_mae_eV_per_atom\": " << energy_mae << ",\n";
  out << "  \"energy_rmse_eV_per_atom\": " << energy_rmse << ",\n";
  out << "  \"force_mae_eV_A\": " << force_mae << ",\n";
  out << "  \"force_rmse_eV_A\": " << force_rmse << ",\n";
  out << "  \"stress_rows\": " << stress_rows << ",\n";
  out << "  \"stress_mae\": " << stress_mae << ",\n";
  out << "  \"stress_rmse\": " << stress_rmse << ",\n";
  out << "  \"species\": [";
  for (std::size_t i = 0; i < species.size(); ++i) {
    if (i) out << ", ";
    out << species[i];
  }
  out << "],\n";
  out << "  \"radial_scaling_by_k\": [";
  const auto& scaling = eval.RadialScaling();
  for (std::size_t i = 0; i < scaling.size(); ++i) {
    if (i) out << ", ";
    out << "{\"k\": " << i + 1 << ", \"scale\": " << scaling[i].first
        << ", \"shift\": " << scaling[i].second << "}";
  }
  out << "],\n";
  out << "  \"channels\": [";
  for (std::size_t i = 0; i < top.channels.size(); ++i) {
    if (i) out << ", ";
    out << "{\"id\": " << i << ", \"l\": " << top.channels[i].l
        << ", \"k\": " << top.channels[i].k << "}";
  }
  out << "],\n";
  out << "  \"couplings\": [";
  for (std::size_t i = 0; i < top.couplings.size(); ++i) {
    const auto& c = top.couplings[i];
    if (i) out << ", ";
    out << "{\"feature\": " << i + 1 << ", \"order\": " << c.order
        << ", \"q\": [" << c.q[0] << ", " << c.q[1] << ", " << c.q[2]
        << ", " << c.q[3] << "], \"intermediate_l\": " << c.intermediate_l << "}";
  }
  out << "],\n";
  out << "  \"coefficients\": [";
  for (std::size_t i = 0; i < coeffs.size(); ++i) {
    if (i) out << ", ";
    out << coeffs[i];
  }
  out << "]\n";
  out << "}\n";
}

void AddNormalRow(std::vector<std::vector<double>>& normal, std::vector<double>& rhs,
                  const std::vector<double>& row, double target, double weight) {
  if (weight == 0.0) return;
  const double s = std::sqrt(weight);
  const double y = s * target;
  const int n = static_cast<int>(row.size());
  for (int i = 0; i < n; ++i) {
    const double xi = s * row[i];
    rhs[i] += xi * y;
    for (int j = 0; j <= i; ++j) normal[i][j] += xi * (s * row[j]);
  }
}

void PrintTopology(const Options& opts) {
  const auto top = BuildTopology(opts.l_max, opts.k_max, opts.body_order);
  std::map<int, int> by_order;
  for (const auto& c : top.couplings) by_order[c.order]++;
  std::cout << "SUS2-SH topology\n";
  std::cout << "  l_max=" << opts.l_max << " k_max=" << opts.k_max
            << " body_order=" << opts.body_order << " max_factors=" << opts.body_order - 1
            << "\n";
  std::cout << "  parity=even_O3\n";
  std::cout << "  channels=" << top.channels.size() << "\n";
  std::cout << "  scalar_count_without_bias=" << top.couplings.size() << "\n";
  std::cout << "  scalar_count_with_bias=" << top.couplings.size() + 1 << "\n";
  for (const auto& kv : by_order) {
    std::cout << "  order_" << kv.first << "_factor_scalars=" << kv.second << "\n";
  }
}

void Train(const Options& opts) {
  const auto top = BuildTopology(opts.l_max, opts.k_max, opts.body_order);
  Evaluator eval(opts, top);
  auto cfgs = ReadCfgs(opts.train_cfg, opts.max_configs);
  if (cfgs.empty()) throw std::runtime_error("no configurations loaded");
  std::vector<double> w(eval.FeatureCount(), 0.0);

  std::cout << "loaded_cfgs=" << cfgs.size() << "\n";
  std::cout << "feature_count=" << eval.FeatureCount() << "\n";
  std::cout << "radial_basis_type=" << opts.radial_basis_type
            << " radial_basis_size=" << opts.radial_basis_size << " cutoff=" << opts.cutoff
            << "\n";

  std::vector<std::vector<double>> x_cache;
  std::vector<double> y_cache;
  std::vector<std::vector<std::vector<double>>> force_row_cache;
  std::vector<std::vector<std::vector<double>>> stress_row_cache;
  x_cache.reserve(cfgs.size());
  y_cache.reserve(cfgs.size());
  force_row_cache.reserve(cfgs.size());
  stress_row_cache.reserve(cfgs.size());
  std::vector<int> species;
  for (const auto& cfg : cfgs) {
    x_cache.push_back(eval.Features(cfg));
    y_cache.push_back(cfg.energy / std::max<std::size_t>(1, cfg.atoms.size()));
    if (opts.force_weight != 0.0) force_row_cache.push_back(eval.ForceDesignRows(cfg));
    if (opts.stress_weight != 0.0 && cfg.has_stress) {
      stress_row_cache.push_back(eval.StressDesignRows(cfg));
    } else {
      stress_row_cache.push_back({});
    }
    for (const auto& atom : cfg.atoms) {
      if (std::find(species.begin(), species.end(), atom.type) == species.end()) {
        species.push_back(atom.type);
      }
    }
  }
  std::sort(species.begin(), species.end());

  const int p_count = eval.FeatureCount();
  std::vector<std::vector<double>> normal(p_count, std::vector<double>(p_count, 0.0));
  std::vector<double> rhs(p_count, 0.0);
  for (std::size_t row = 0; row < x_cache.size(); ++row) {
    AddNormalRow(normal, rhs, x_cache[row], y_cache[row], opts.energy_weight);
    if (opts.force_weight != 0.0) {
      int comp = 0;
      for (const auto& atom : cfgs[row].atoms) {
        const double targets[3] = {atom.force.x, atom.force.y, atom.force.z};
        for (int a = 0; a < 3; ++a, ++comp) {
          AddNormalRow(normal, rhs, force_row_cache[row][comp], targets[a], opts.force_weight);
        }
      }
    }
    if (opts.stress_weight != 0.0 && cfgs[row].has_stress) {
      const double targets[6] = {cfgs[row].stress[0][0], cfgs[row].stress[1][1],
                                 cfgs[row].stress[2][2], cfgs[row].stress[1][2],
                                 cfgs[row].stress[0][2], cfgs[row].stress[0][1]};
      for (int a = 0; a < 6; ++a) {
        AddNormalRow(normal, rhs, stress_row_cache[row][a], targets[a], opts.stress_weight);
      }
    }
  }
  for (int i = 0; i < p_count; ++i) {
    for (int j = 0; j < i; ++j) normal[j][i] = normal[i][j];
    normal[i][i] += opts.ridge;
  }
  w = SolveLinearSystem(std::move(normal), std::move(rhs));

  double mae = 0.0;
  double mse = 0.0;
  double force_mae = 0.0;
  double force_mse = 0.0;
  int force_count = 0;
  double stress_mae = 0.0;
  double stress_mse = 0.0;
  int stress_count = 0;
  for (std::size_t i = 0; i < x_cache.size(); ++i) {
    const double e_err = DotVec(w, x_cache[i]) - y_cache[i];
    mae += std::abs(e_err);
    mse += e_err * e_err;
    std::vector<Vec3> pred_forces;
    std::array<std::array<double, 3>, 3> pred_stress{};
    eval.EnergyForcesStress(cfgs[i], w, &pred_forces, &pred_stress);
    for (std::size_t atom = 0; atom < cfgs[i].atoms.size(); ++atom) {
      const double diffs[3] = {
          pred_forces[atom].x - cfgs[i].atoms[atom].force.x,
          pred_forces[atom].y - cfgs[i].atoms[atom].force.y,
          pred_forces[atom].z - cfgs[i].atoms[atom].force.z};
      for (double d : diffs) {
        force_mae += std::abs(d);
        force_mse += d * d;
        ++force_count;
      }
    }
    if (cfgs[i].has_stress) {
      const double pred[6] = {pred_stress[0][0], pred_stress[1][1], pred_stress[2][2],
                              pred_stress[1][2], pred_stress[0][2], pred_stress[0][1]};
      const double target[6] = {cfgs[i].stress[0][0], cfgs[i].stress[1][1], cfgs[i].stress[2][2],
                                cfgs[i].stress[1][2], cfgs[i].stress[0][2], cfgs[i].stress[0][1]};
      for (int a = 0; a < 6; ++a) {
        const double d = pred[a] - target[a];
        stress_mae += std::abs(d);
        stress_mse += d * d;
        ++stress_count;
      }
    }
  }
  mae /= x_cache.size();
  const double rmse = std::sqrt(mse / x_cache.size());
  force_mae /= std::max(1, force_count);
  const double force_rmse = std::sqrt(force_mse / std::max(1, force_count));
  if (stress_count > 0) stress_mae /= stress_count;
  const double stress_rmse = stress_count > 0 ? std::sqrt(stress_mse / stress_count) : 0.0;
  SaveModel(opts.model_out, opts, top, eval, w, mae, rmse, force_mae, force_rmse,
            stress_mae, stress_rmse, stress_count, species);
  std::cout << "final_train_energy_mae_eV_per_atom=" << mae << "\n";
  std::cout << "final_train_energy_rmse_eV_per_atom=" << rmse << "\n";
  std::cout << "final_train_force_mae_eV_A=" << force_mae << "\n";
  std::cout << "final_train_force_rmse_eV_A=" << force_rmse << "\n";
  std::cout << "final_train_stress_rows=" << stress_count << "\n";
  std::cout << "final_train_stress_mae=" << stress_mae << "\n";
  std::cout << "final_train_stress_rmse=" << stress_rmse << "\n";
  std::cout << "wrote_model=" << opts.model_out << "\n";
}

void CheckGradient(const Options& opts) {
  auto cfgs = ReadCfgs(opts.train_cfg, 1);
  if (cfgs.empty()) throw std::runtime_error("no configurations loaded");
  auto cfg = cfgs.front();
  const auto top = BuildTopology(opts.l_max, opts.k_max, opts.body_order);
  Evaluator eval(opts, top);
  std::vector<double> coeffs(eval.FeatureCount(), 0.0);
  for (std::size_t i = 0; i < coeffs.size(); ++i) {
    coeffs[i] = 1.0e-4 * std::sin(0.37 * static_cast<double>(i + 1));
  }
  coeffs[0] = -1.0;
  std::vector<Vec3> forces;
  const double e0 = eval.EnergyAndForces(cfg, coeffs, &forces);
  double max_abs = 0.0;
  double rms = 0.0;
  int count = 0;
  for (std::size_t atom = 0; atom < cfg.atoms.size(); ++atom) {
    for (int a = 0; a < 3; ++a) {
      Config plus = cfg;
      Config minus = cfg;
      double* pp = a == 0 ? &plus.atoms[atom].pos.x : (a == 1 ? &plus.atoms[atom].pos.y : &plus.atoms[atom].pos.z);
      double* pm = a == 0 ? &minus.atoms[atom].pos.x : (a == 1 ? &minus.atoms[atom].pos.y : &minus.atoms[atom].pos.z);
      *pp += opts.fd_step;
      *pm -= opts.fd_step;
      const double ep = eval.EnergyAndForces(plus, coeffs, nullptr);
      const double em = eval.EnergyAndForces(minus, coeffs, nullptr);
      const double fd_force = -(ep - em) / (2.0 * opts.fd_step);
      const double analytic = a == 0 ? forces[atom].x : (a == 1 ? forces[atom].y : forces[atom].z);
      const double diff = analytic - fd_force;
      max_abs = std::max(max_abs, std::abs(diff));
      rms += diff * diff;
      ++count;
    }
  }
  rms = std::sqrt(rms / std::max(1, count));
  std::cout << std::setprecision(17);
  std::cout << "energy=" << e0 << "\n";
  std::cout << "force_fd_check_components=" << count << "\n";
  std::cout << "force_fd_rms_abs_error=" << rms << "\n";
  std::cout << "force_fd_max_abs_error=" << max_abs << "\n";
}

Options ParseOptions(int argc, char** argv) {
  if (argc < 2) throw std::runtime_error("usage: sus2-sh <inspect|train> [options]");
  Options opts;
  opts.command = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string key = argv[i];
    auto need_value = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + name);
      return argv[++i];
    };
    if (key == "--train-cfg") opts.train_cfg = need_value(key);
    else if (key == "--model-out") opts.model_out = need_value(key);
    else if (key == "--l-max") opts.l_max = std::stoi(need_value(key));
    else if (key == "--k-max") opts.k_max = std::stoi(need_value(key));
    else if (key == "--body-order") opts.body_order = std::stoi(need_value(key));
    else if (key == "--radial-basis-size") opts.radial_basis_size = std::stoi(need_value(key));
    else if (key == "--cutoff") opts.cutoff = std::stod(need_value(key));
    else if (key == "--radial-basis-type") opts.radial_basis_type = need_value(key);
    else if (key == "--max-configs") opts.max_configs = std::stoi(need_value(key));
    else if (key == "--max-iter") opts.max_iter = std::stoi(need_value(key));
    else if (key == "--fd-step") opts.fd_step = std::stod(need_value(key));
    else if (key == "--energy-weight") opts.energy_weight = std::stod(need_value(key));
    else if (key == "--force-weight") opts.force_weight = std::stod(need_value(key));
    else if (key == "--stress-weight") opts.stress_weight = std::stod(need_value(key));
    else throw std::runtime_error("unknown option: " + key);
  }
  if ((opts.command == "train" || opts.command == "check-grad") && opts.train_cfg.empty()) {
    throw std::runtime_error(opts.command + " requires --train-cfg");
  }
  if (opts.body_order < 2 || opts.body_order > 5) {
    throw std::runtime_error("prototype supports body_order in [2,5]");
  }
  if (opts.radial_basis_type != "RBChebyshev_sss") {
    throw std::runtime_error("this prototype currently supports --radial-basis-type RBChebyshev_sss");
  }
  return opts;
}

}  // namespace sus2sh

int main(int argc, char** argv) {
  try {
    const auto opts = sus2sh::ParseOptions(argc, argv);
    if (opts.command == "inspect") {
      sus2sh::PrintTopology(opts);
    } else if (opts.command == "train") {
      sus2sh::Train(opts);
    } else if (opts.command == "check-grad") {
      sus2sh::CheckGradient(opts);
    } else {
      throw std::runtime_error("unknown command: " + opts.command);
    }
  } catch (const std::exception& e) {
    std::cerr << "sus2-sh: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
