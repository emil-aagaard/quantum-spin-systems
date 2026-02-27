import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_eigenvalues(
    eigenvalues: np.ndarray[np.float64],
    eigenvectors: np.ndarray[np.ndarray[np.float64]],
    path: Path,
    tolerance: float=1e-12
) -> None:
    previous_eigenvalue = None
    eigenspace: list[list[np.float64 | int]] = []
    for eigenvalue, _ in zip(eigenvalues, eigenvectors.T):
        if (
            previous_eigenvalue is not None
            and abs(eigenvalue - previous_eigenvalue) < tolerance
        ):
            eigenspace[-1][1] += 1
        else:
            previous_eigenvalue = eigenvalue
            eigenspace.append([eigenvalue, 1])
    xs = []
    ys = []
    for eigenvalue, count in eigenspace:
        for index in range(count):
            xs.append(eigenvalue)
            ys.append(index - (count - 1) / 2)
    fig, ax = plt.subplots()
    ax.scatter(xs, ys)
    fig.savefig(path)
