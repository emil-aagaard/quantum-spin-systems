from quantum_spin_systems import (
    QuantumSpinSystem,
    ChainLattice,
    SquareLattice,
    CubicLattice,
    HeisenbergHamiltonian,
    H0Hamiltonian,
    SZTotal,
    XYHamiltonian,
)
from quantum_spin_systems.plot import plot_eigenvalues
from quantum_spin_systems.helpers import get_eigenvalue
from pathlib import Path

if __name__ == "__main__":
    dimensions = [
        2,
    ] * 8
    quantum_spin_system = QuantumSpinSystem(dimensions=dimensions)
    lattice = ChainLattice(size=quantum_spin_system.n)
    hamiltonian = HeisenbergHamiltonian(
        quantum_spin_system=quantum_spin_system, lattice=lattice, J=1.0
    )
    eigenvalues, eigenvectors = hamiltonian.diagonalize(k=10, which="SA")
    plot_eigenvalues(
        eigenvalues=eigenvalues, eigenvectors=eigenvectors, path=Path("fig")
    )
