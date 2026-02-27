from scipy.sparse import lil_matrix
from numpy import ndarray, float64


def get_eigenvalue(
    matrix: lil_matrix, vector: ndarray[float64], tolerance: float=1e-12
) -> None | float64:
    vector_at_vector = vector.T @ vector
    if vector_at_vector < tolerance:
        return None
    else:
        matrix_at_vector = matrix @ vector
        alpha = vector.T @ matrix_at_vector / vector_at_vector
        adjusted_vector = matrix_at_vector - alpha * vector
        adjusted_vector_at_adjusted_vector = adjusted_vector.T @ adjusted_vector
        if adjusted_vector_at_adjusted_vector < tolerance:
            return alpha
        else:
            return None
