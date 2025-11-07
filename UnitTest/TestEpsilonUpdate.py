import unittest
import torch

from UnitTest.epsilon_logic import svd_epsilon_threshold


def generate_matrix_with_rank(m: int, n: int, r: int) -> torch.Tensor:
    """Helper function to generate a matrix with specified rank"""
    torch.manual_seed(42)
    U = torch.randn(m, r)
    V = torch.randn(n, r)
    return U @ V.T

class TestSVDEpsilonThreshold(unittest.TestCase):
    def setUp(self):
        self.matrix_small = torch.tensor([[3.0, 1.0], [1.0, 3.0]])
        torch.manual_seed(42)
        self.matrix_large = torch.randn(10, 10)
        self.epsilon = 0.1

    def test_all_singular_values_greater_than_epsilon(self):
        X = self.matrix_small
        epsilon = 0.1
        print(f"\nTesting with small matrix and epsilon={epsilon}")
        U, s, V, q = svd_epsilon_threshold(X, epsilon)
        print(f"Singular values: {s}")
        print(f"Rank estimate q: {q}")
        # If no values <= epsilon, should return full rank
        self.assertEqual(s.numel(), min(X.size()))
        self.assertEqual(q, min(X.size()))

    def test_one_singular_value_less_than_epsilon(self):
        X = self.matrix_small
        epsilon = 2.0  # Set epsilon between singular values
        print(f"\nTesting with small matrix and epsilon={epsilon}")
        U, s, V, q = svd_epsilon_threshold(X, epsilon)
        print(f"Singular values: {s}")
        print(f"Rank estimate q: {q}")
        # Should find exactly one value <= epsilon
        self.assertEqual(torch.sum(s <= epsilon).item(), 1)
        # All other values should be > epsilon
        self.assertTrue(torch.sum(s > epsilon).item() == q - 1)

    def test_full_rank_case(self):
        X = self.matrix_large
        epsilon = 1e-10  # Very small epsilon
        print(f"\nTesting with large matrix and epsilon={epsilon}")
        U, s, V, q = svd_epsilon_threshold(X, epsilon)
        print(f"Matrix shape: {X.shape}")
        print(f"Number of singular values: {s.numel()}")
        print(f"Rank estimate q: {q}")
        # Should return full rank since no values <= epsilon
        self.assertEqual(s.numel(), min(X.size()))
        self.assertEqual(q, min(X.size()))

    def test_rank_estimation(self):
        m, n = 10, 8
        # Create matrix with known rank
        X = generate_matrix_with_rank(m, n, 5)
        epsilon = 0.5
        print(f"\nTesting rank estimation with epsilon={epsilon}")
        U, s, V, q = svd_epsilon_threshold(X, epsilon)
        print(f"Singular values: {s}")
        print(f"Rank estimate q: {q}")
        # Verify one value <= epsilon
        self.assertEqual(torch.sum(s <= epsilon).item(), 1)
        # Verify remaining values > epsilon
        self.assertEqual(torch.sum(s > epsilon).item(), q - 1)

    def test_truncation_property(self):
        X = self.matrix_large
        epsilon = 0.5
        print(f"\nTesting truncation property with epsilon={epsilon}")
        U, s, V, q = svd_epsilon_threshold(X, epsilon)
        print(f"Singular values: {s}")
        print(f"Rank estimate q: {q}")
        if torch.any(s <= epsilon):
            # If we found a truncated form
            self.assertEqual(torch.sum(s <= epsilon).item(), 1)
            self.assertEqual(torch.sum(s > epsilon).item(), q - 1)
        else:
            # If no truncation possible
            self.assertEqual(q, min(X.size()))
            self.assertEqual(s.numel(), min(X.size()))

if __name__ == '__main__':
    unittest.main(verbosity=2)