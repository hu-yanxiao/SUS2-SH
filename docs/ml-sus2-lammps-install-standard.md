# ML-SUS2 LAMMPS Install Standard

## Scope

This note records the current standard for maintaining the `ML-SUS2` LAMMPS interface on the phy-weigw server.

Use `ML-SUS2` naming as the active convention. Do not treat `ML-SUS2-MTP` as the active package name in new work, except when cleaning up historical build residue.

## Canonical Remote Paths

- Shared source tree:

```bash
/work/phy-weigw/apps/lammps-10Dec2025
```

- CPU install location:

```bash
/work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi
```

- GPU install location:

```bash
/work/phy-weigw/20260321_Test/lammps-sus2kk-v45-all-double-centroidstress/lmp.v45_all_double_centroidstress_tabstep_double_compute1
```

- Interface tarball reference:

```bash
/work/phy-weigw/20260321_Test/SUS2-MLIP-release-work-codex/sus2-interface-20260410.tar.gz
```

## CPU Standard

CPU LAMMPS uses the old `make` workflow.

Requirements:

1. Use the Intel compiler toolchain.
2. Build CPU-specific installed binaries for the main CPU instruction-set families instead of relying on `-xHost`.
3. Keep the canonical installed executable path as an auto-dispatch wrapper:

```bash
/work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi
```

Current CPU installed binaries:

```bash
/work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi_avx2
/work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi_avx512_skx
```

Current CPU `make` targets:

```bash
make -j8 ml_sus2_avx2
make -j8 ml_sus2_avx512_skx
```

The wrapper accepts an override for A/B testing:

```bash
SUS2_LAMMPS_CPU_TARGET=auto
SUS2_LAMMPS_CPU_TARGET=avx2
SUS2_LAMMPS_CPU_TARGET=avx512_skx
```

CPU target guidance:

- Use `avx2` for Intel E5, AMD EPYC 7352, and unknown nodes.
- Use `avx2` for Xeon Gold 6133/6138 and Platinum 8175M-class Skylake-SP nodes; SUS2-SL benchmarked faster with AVX2 than AVX512 on Gold 6133 because AVX512 downclock outweighed vector-width gains.
- Use `avx512_skx` only for newer AVX512 nodes after direct validation, such as Ice Lake 8375C-class nodes.
- Consider a future Sapphire Rapids target only after direct SUS2 benchmarking on 8488C, 8457C, or 8581C nodes shows a benefit.

Operational rule:

- CPU `make` modifies the shared source tree directly. Assume it leaves residue that must be cleaned before any later GPU CMake rebuild.

## GPU Standard

GPU LAMMPS uses CMake from the same shared source tree.

Use the cluster CMake module for GPU CMake work:

```bash
module load cmake/3.25.2
```

Requirements:

1. Use the same source tree:

```bash
/work/phy-weigw/apps/lammps-10Dec2025
```

2. Before configuring GPU CMake, first clear the residue left by the CPU `make` route:

```bash
make -C /work/phy-weigw/apps/lammps-10Dec2025/src no-all purge
```

3. Also remove stale root-level SUS2 interface files from `src/` if they exist:

```bash
pair_sus2_mtp.cpp
pair_sus2_mtp.h
sus2_mtp_*.cpp
sus2_mtp_*.h
```

4. Use a clean CMake build directory instead of trying to reuse a polluted one.
5. Submit the GPU build to a GPU node through LSF instead of relying on the login node for the full rebuild.
6. GPU build scripts should load `cmake/3.25.2` explicitly before running `cmake --build` or configuring a build directory.

Installed result:

```bash
/work/phy-weigw/20260321_Test/lammps-sus2kk-v45-all-double-centroidstress/lmp.v45_all_double_centroidstress_tabstep_double_compute1
```

## Recommended Rebuild Order

1. Sync the interface source updates into the shared LAMMPS source tree.
2. Rebuild the CPU binary with `make` if the CPU interface changed.
3. Before any GPU rebuild, purge the CPU `make` residue from the shared tree.
4. Reconfigure GPU CMake in a clean build directory.
5. Submit the GPU compile on a GPU queue.
6. Replace the canonical installed GPU binary in place.
7. Run smoke tests from:

```bash
/work/phy-weigw/hyx/ma/l3k3/jacobi/lmp
```

## Operational Intent

The purpose of this standard is to keep future interface edits smooth:

- one shared source tree
- one canonical CPU install path
- one canonical GPU install path
- explicit purge between `make` and CMake
- GPU rebuilds performed on GPU nodes

Avoid ad hoc binary copies and mixed build states unless the user explicitly asks for a temporary experiment.
