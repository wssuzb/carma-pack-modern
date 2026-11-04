"""Fit the bundled OGLE light curve and save mean and conditional draws."""
from pathlib import Path

import carmcmc as cm
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    data = Path(__file__).with_name("OGLE-LMC-LPV-00007.dat")
    time, magnitude, error = np.loadtxt(data, unpack=True)
    time -= time.min()

    cm.set_seed(42)
    posterior = cm.CarmaModel(time, magnitude, error, p=1, q=0).run_mcmc(
        4000, nburnin=2000
    )
    grid = np.arange(0.0, time.max() + 101.0)
    mean, variance = posterior.predict(grid, bestfit="map")
    sigma = np.sqrt(variance)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.errorbar(time, magnitude, yerr=error, fmt=".", color="black", label="OGLE")
    ax.fill_between(grid, mean - sigma, mean + sigma, alpha=0.15, color="tab:blue")
    for i in range(3):
        draw = posterior.simulate(grid.copy(), bestfit="map")
        ax.plot(grid, draw, color="tab:blue", alpha=0.3,
                label="Conditional draws" if i == 0 else None)
    ax.plot(grid, mean, color="tab:orange", label="Conditional mean")
    ax.set(xlabel="Days since first observation", ylabel="Magnitude")
    ax.invert_yaxis()
    ax.legend()
    fig.tight_layout()
    fig.savefig("ogle_car1_fit.png", dpi=150)


if __name__ == "__main__":
    main()
