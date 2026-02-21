from scipy.sparse import lil_matrix, identity, kron
from scipy.sparse.linalg import eigsh


class QuantumSpinSystem:
    def __init__(self, dimensions: list[int]) -> None:
        self.dimensions = dimensions
        self.total_dimension = 1
        for dimension in dimensions:
            self.total_dimension *= dimension

    def embed_matrix(self, matrix: lil_matrix, i: int) -> lil_matrix:
        result = identity(1, format="lil")
        for j in range(len(self.dimensions)):
            if j == i:
                result = kron(result, matrix, format="lil")
            else:
                identity_matrix = identity(self.dimensions[j], format="lil")
                result = kron(result, identity_matrix, format="lil")
        return result

    def get_s_minus_i(self, i: int) -> lil_matrix:
        dimension = self.dimensions[i]
        s_minus_i = lil_matrix((dimension, dimension))
        for m in range(1, dimension):
            s_minus_i[m, m - 1] = ((dimension - m) * m) ** 0.5
        s_minus_i = self.embed_matrix(s_minus_i, i)
        return s_minus_i

    def get_s_plus_i(self, i: int) -> lil_matrix:
        dimension = self.dimensions[i]
        s_plus_i = lil_matrix((dimension, dimension))
        for m in range(dimension - 1):
            s_plus_i[m, m + 1] = ((dimension - m - 1) * (m + 1)) ** 0.5
        s_plus_i = self.embed_matrix(s_plus_i, i)
        return s_plus_i

    def get_s_z_i(self, i: int) -> lil_matrix:
        dimension = self.dimensions[i]
        s_z_i = lil_matrix((dimension, dimension))
        for m in range(dimension):
            s_z_i[m, m] = m - (dimension - 1) / 2
        s_z_i = self.embed_matrix(s_z_i, i)
        return s_z_i


class Lattice:
    def __init__(self, size: int) -> None:
        self.size = size

    def get_neighbors(self, _: int) -> list[int]:
        raise NotImplementedError("This method should be implemented by subclasses.")


class ChainLattice(Lattice):
    def get_neighbors(self, i: int) -> list[int]:
        return [(i + 1) % self.size]


class SquareLattice(Lattice):
    def get_neighbors(self, i: int) -> list[int]:
        side_length = int(self.size**0.5)
        row = i // side_length
        column = i % side_length
        neighbors = [
            (row * side_length + (column + 1) % side_length),
            ((row + 1) % side_length * side_length + column),
        ]
        return neighbors


class CubicLattice(Lattice):
    def get_neighbors(self, i: int) -> list[int]:
        side_length = int(self.size ** (1 / 3))
        layer_size = side_length**2
        layer = i // layer_size
        row = (i % layer_size) // side_length
        column = i % side_length
        neighbors = [
            (layer * layer_size + row * side_length + (column + 1) % side_length),
            (layer * layer_size + ((row + 1) % side_length) * side_length + column),
            (((layer + 1) % side_length) * layer_size + row * side_length + column),
        ]
        return neighbors


class Hamiltonian:
    def __init__(
        self, quantum_spin_system: QuantumSpinSystem, lattice: Lattice
    ) -> None:
        self.quantum_spin_system = quantum_spin_system
        self.lattice = lattice
        self.matrix = lil_matrix(
            (quantum_spin_system.total_dimension, quantum_spin_system.total_dimension)
        )

    def construct_hamiltonian(self) -> None:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def diagonalize(self, k: int = 6) -> tuple:
        eigenvalues, eigenvectors = eigsh(self.matrix, k=k, which="SA")
        return eigenvalues, eigenvectors


class HeisenbergHamiltonian(Hamiltonian):
    def __init__(
        self, quantum_spin_system: QuantumSpinSystem, lattice: Lattice, J: float
    ) -> None:
        super().__init__(quantum_spin_system, lattice)
        self.J = J
        self.construct_hamiltonian()

    def construct_hamiltonian(self) -> None:
        for i in range(len(self.quantum_spin_system.dimensions)):
            neighbors = self.lattice.get_neighbors(i)
            for j in neighbors:
                s_minus_i = self.quantum_spin_system.get_s_minus_i(i)
                s_plus_i = self.quantum_spin_system.get_s_plus_i(i)
                s_z_i = self.quantum_spin_system.get_s_z_i(i)

                s_minus_j = self.quantum_spin_system.get_s_minus_i(j)
                s_plus_j = self.quantum_spin_system.get_s_plus_i(j)
                s_z_j = self.quantum_spin_system.get_s_z_i(j)

                interaction_term = (
                    0.5 * s_minus_i @ s_plus_j
                    + 0.5 * s_plus_i @ s_minus_j
                    + s_z_i @ s_z_j
                )
                self.matrix += self.J * interaction_term


class XYHamiltonian(Hamiltonian):
    def __init__(
        self, quantum_spin_system: QuantumSpinSystem, lattice: Lattice, J: float
    ) -> None:
        super().__init__(quantum_spin_system, lattice)
        self.J = J
        self.construct_hamiltonian()

    def construct_hamiltonian(self) -> None:
        for i in range(len(self.quantum_spin_system.dimensions)):
            neighbors = self.lattice.get_neighbors(i)
            for j in neighbors:
                s_minus_i = self.quantum_spin_system.get_s_minus_i(i)
                s_plus_i = self.quantum_spin_system.get_s_plus_i(i)

                s_minus_j = self.quantum_spin_system.get_s_minus_i(j)
                s_plus_j = self.quantum_spin_system.get_s_plus_i(j)

                interaction_term = (
                    0.5 * s_minus_i @ s_plus_j + 0.5 * s_plus_i @ s_minus_j
                )
                self.matrix += self.J * interaction_term
