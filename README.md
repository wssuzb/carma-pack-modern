# carma-pack-modern

A modern Python build of Brandon C. Kelly's CARMA time-series package. It preserves the original numerical methods and uses pybind11, CMake and scikit-build-core. Based on [carma_pack](https://github.com/brandonckelly/carma_pack/tree/26616d99e3b3b38a2c9dc76b6fc7ff563af8f2f1) under the [MIT license](LICENSE.txt).

## Install from source

Use a fresh Python 3.12 environment for the tested configuration. The distribution is named `carma-pack-modern` and its import is `carmcmc`. The original distribution uses the same import, so give each version its own environment.

On macOS, install Xcode Command Line Tools and the native build dependencies:

```bash
xcode-select --install
brew install boost armadillo cmake ninja
```

On Debian or Ubuntu, the corresponding dependencies are:

```bash
sudo apt-get install build-essential cmake ninja-build libarmadillo-dev libboost-filesystem-dev libboost-timer-dev
```

From the repository root:

```bash
git clone https://github.com/wssuzb/carma-pack-modern.git
cd carma-pack-modern
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install .
python -c 'import carmcmc; print(carmcmc.__file__)'
```

For native libraries installed under a custom prefix, set `CMAKE_PREFIX_PATH` before building. Intel macOS with CPython 3.12 passed 125 tests with the local toolchain. A Linux x86_64 build was tested in a cloud environment. The Homebrew recipe, Apple Silicon, Windows and portable Linux wheels still need release validation.

## Fit the OGLE example

```bash
python examples/fit_ogle.py
```

The script saves `ogle_car1_fit.png` in the working directory. The solid line is the conditional mean at the maximum-posterior parameter draw. The lighter lines are conditional simulations at the same parameters. In seasonal gaps, the conditional mean tends toward the long-term mean while the simulated paths continue to fluctuate.

For the original CAR(6,0) workflow, see [Kelly's OGLE notebook](examples/carma_pack_guide.ipynb). The notebook uses older plotting and diagnostic APIs.

## Compatibility and limitations

The original and modernized C++ cores returned identical samples and log posteriors for the full 20,000-sample OGLE CAR(6,0) run with the same Python binding, input, seed and toolchain. CAR(1) automatic initialization was corrected to reject invalid starts. CARMA(2,1) retains the upstream MA prior, which can leave an unbounded posterior tail and cause parameter drift. See the [release notes](RELEASE_NOTES.md) for details.
