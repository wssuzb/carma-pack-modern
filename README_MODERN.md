# carma_pack_modern 0.1.1 — CAR1 initialization repair

Upstream: https://github.com/brandonckelly/carma_pack
Pinned commit: `26616d99e3b3b38a2c9dc76b6fc7ff563af8f2f1` (MIT).
Phase 1 checkpoint: `9508ecfdaec59b0b9a64197b12dbb379a9ccac61`.

Distribution name: `carma_pack_modern`; Python import: `carmcmc`. Do not
co-install the legacy carmcmc distribution. Use the **root** pyproject.toml;
the historical setup files under src/ are not the modern build entry point.

This version fixes CAR1 automatic starts outside the existing prior support.
The initial tau draw is the original median(dt)*Uniform(1,50), conditioned on
being no longer than the data span. A returned start must be legal and have
finite posterior density. Attempts are limited to 128. Invalid explicit CAR1
initial points raise ValueError rather than silently falling back to random
initialization. Constant/too-short/invalid data and unrepresentable original
proposal scales are rejected before sampling.

**Two core files now differ from upstream:** src/carpack.cpp (initialization)
and src/carmcmc.cpp (CAR1 sampler entry validation). Full patch and exact hashes
are provided. The target density/prior, Kalman recursion, proposal/adaptation,
MCMC transition and acceptance code are unchanged. All other original C++ and
header hashes remain identical. See REPORT_PHASE2.md and
`audit/phase2/core_changes.json`. `audit/REPORT.md` describes the older Phase 1,
not the current version.

## Build wheel (cloud Linux recipe)

A C++14 compiler, Armadillo and Boost.Filesystem/Timer development packages,
and the corresponding BLAS/LAPACK runtime dependencies must be available.
Recorded cloud versions are in audit/cloud_cpp_dependencies.json. For a normal
Ubuntu environment, libarmadillo-dev, libboost-filesystem-dev and
libboost-timer-dev supply the principal native dependencies.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install build cmake ninja
# With native libraries in standard paths:
CXX=g++ python -m build --wheel
python -m pip install dist/carma_pack_modern-0.1.1-*.whl
```

A private prefix requires CMAKE_PREFIX_PATH during builds and an appropriate
runtime search path (LD_LIBRARY_PATH in the cloud run). The delivered wheel is
CPython 3.12 / Linux x86_64, uses external shared libraries, and is **not** a
self-contained manylinux or macOS wheel. See audit/phase2/auditwheel.txt.
No build, installation or command was executed on the user's Mac.

## Reproduce the full suite, including an immutable original C++ oracle

`modern_tests/upstream_core` contains byte-identical pinned upstream snapshots,
verified against PROVENANCE.json and the Phase 1 source hashes. It is compiled
as a separate standalone CMake project and never linked into the wheel.

```bash
python -m pip install pytest numpy scipy matplotlib pybind11 cmake ninja
cmake -S modern_tests/upstream_core -B build/upstream-oracle -G Ninja \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/upstream-oracle -j2
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build --target initialization_probe -j2
export CARMA_REFERENCE_DRIVER="$PWD/build/upstream-oracle/upstream_reference_driver"
export CARMA_UPSTREAM_PROBE="$PWD/build/upstream-oracle/upstream_initialization_probe"
export CARMA_CURRENT_PROBE="$PWD/build/initialization_probe"
python modern_tests/run_cloud_validation.py
```

Add `-DCMAKE_PREFIX_PATH=...` to both CMake configurations when using a private
native dependency prefix, and set its runtime library search path. Test the
installed wheel from a noneditable Python environment. The root build's legacy
reference_driver target links the patched core; **use the independent upstream
executable above** for comparisons with unmodified C++.

The historical failing case can be run offline without Python:

```bash
build/upstream-oracle/upstream_initialization_probe inspect \
  < audit/phase2/reproducer_input.txt
build/initialization_probe inspect < audit/phase2/reproducer_input.txt
```

Expected evidence is preserved in upstream_root_cause.txt / fixed_root_cause.txt.
`initialization_probe exhaust` uses a test-only extremely restrictive amplitude
prior to verify the 128-attempt failure path; it does not change the library.

## API notes

Vectors remain vecD/vecC/vecvecD, predictions retain pairD.first/second.
`set_seed(seed)` sets both Boost and Armadillo RNGs. All core calls retain the
GIL because RNG/state is shared; no thread-local or concurrent sampling claim.
The constructor temperature and sampler default arguments remain as in Phase 1.

Direct low-level inputs must be finite and equal length >=3, with strictly
increasing times and positive errors. Use CAR1/KalmanFilter1 for OU; the generic
CARp/KalmanFilterp require p>=2. Data-only Kalman constructors raise ValueError;
provide process parameters. These Phase 1 boundary changes remain intentional.

A CAR1 explicit initial vector is `[stationary_stdev, error_variance_scale,
mean, log(omega)]`; the first element is not the driving-noise amplitude.
The existing prior bounds, including error scale [0.5,2], are unchanged.
Log values from GetLogLikes are the stored unnormalized **posterior** (prior
included, n log(2pi) omitted), despite the historical name.

acor remains an optional diagnostic, loaded only when requested; no substitute
estimator was added and its own build compatibility is unverified. SciPy and
Matplotlib remain Python API dependencies. Higher-order initialization policies
and ZCAR were not repaired in this CAR1-specific patch.

Passing finite-start and movement tests does not establish convergence,
posterior coverage or mixing. Review diagnostics for each scientific dataset.
The current code is ready for a separate, isolated macOS build experiment;
macOS portability and dependency packaging remain untested. Do not install the
Linux wheel or modify the user's Mac as part of this delivery.
