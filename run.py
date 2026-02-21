from quantum_spin_systems import (
    QuantumSpinSystem,
    ChainLattice,
    SquareLattice,
    CubicLattice,
    HeisenbergHamiltonian,
    XYHamiltonian,
)

if __name__ == "__main__":
    dimensions = [
        3,
    ] * 6
    quantum_spin_system = QuantumSpinSystem(dimensions=dimensions)

    # Heisenberg model on a chain lattice
    lattice = ChainLattice(size=len(dimensions))
    hamiltonian = HeisenbergHamiltonian(
        quantum_spin_system=quantum_spin_system, lattice=lattice, J=1.0
    )
    eigenvalues, eigenvectors = hamiltonian.diagonalize(k=4)
    print(
        f"Heisenberg model S={(dimensions[0]-1)/2}, N={len(dimensions)} on a chain lattice:"
    )
    print("Eigenvalues:", eigenvalues)

    # XY model on a chain lattice
    lattice = ChainLattice(size=len(dimensions))
    hamiltonian = XYHamiltonian(
        quantum_spin_system=quantum_spin_system, lattice=lattice, J=1.0
    )
    eigenvalues, eigenvectors = hamiltonian.diagonalize(k=4)
    print(
        f"\nXY model S={(dimensions[0]-1)/2}, N={len(dimensions)} on a chain lattice:"
    )
    print("Eigenvalues:", eigenvalues)

    dimensions = [
        3,
    ] * 9
    quantum_spin_system = QuantumSpinSystem(dimensions=dimensions)

    # Heisenberg model on a square lattice
    lattice = SquareLattice(size=len(dimensions))
    hamiltonian = HeisenbergHamiltonian(
        quantum_spin_system=quantum_spin_system, lattice=lattice, J=1.0
    )
    eigenvalues, eigenvectors = hamiltonian.diagonalize(k=4)
    print(
        f"\nHeisenberg model S={(dimensions[0]-1)/2}, N={len(dimensions)} on a square lattice:"
    )
    print("Eigenvalues:", eigenvalues)

    # XY model on a square lattice
    lattice = SquareLattice(size=len(dimensions))
    hamiltonian = XYHamiltonian(
        quantum_spin_system=quantum_spin_system, lattice=lattice, J=1.0
    )
    eigenvalues, eigenvectors = hamiltonian.diagonalize(k=4)
    print(
        f"\nXY model S={(dimensions[0]-1)/2}, N={len(dimensions)} on a square lattice:"
    )
    print("Eigenvalues:", eigenvalues)

    dimensions = [
        3,
    ] * 8
    quantum_spin_system = QuantumSpinSystem(dimensions=dimensions)

    # Heisenberg model on a cubic lattice
    lattice = CubicLattice(size=len(dimensions))
    hamiltonian = HeisenbergHamiltonian(
        quantum_spin_system=quantum_spin_system, lattice=lattice, J=1.0
    )
    eigenvalues, eigenvectors = hamiltonian.diagonalize(k=4)
    print(
        f"\nHeisenberg model S={(dimensions[0]-1)/2}, N={len(dimensions)} on a cubic lattice:"
    )
    print("Eigenvalues:", eigenvalues)

    # XY model on a cubic lattice
    lattice = CubicLattice(size=len(dimensions))
    hamiltonian = XYHamiltonian(
        quantum_spin_system=quantum_spin_system, lattice=lattice, J=1.0
    )
    eigenvalues, eigenvectors = hamiltonian.diagonalize(k=4)
    print(
        f"\nXY model S={(dimensions[0]-1)/2}, N={len(dimensions)} on a cubic lattice:"
    )
    print("Eigenvalues:", eigenvalues)
