import numpy as np
from scipy.special import softmax
import unittest
from time import time
from typing import Tuple, Union, List
from dataclasses import dataclass


@dataclass
class CTCState:
    """Class to store CTC computation state"""
    alpha: np.ndarray
    beta: np.ndarray = None
    loss: float = None
    gradients: np.ndarray = None


class CTCLoss:
    """CTC Loss implementation with batch processing support"""

    def __init__(self, blank_id: int = 0, reduction: str = 'mean'):
        """
        Initialize CTC Loss calculator
        """
        self.blank_id = blank_id
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction: {reduction}. Must be 'mean', 'sum', or 'none'")
        self.reduction = reduction
        self.NINF = -float('inf')
        self.EPS = 1e-7

    def extend_targets(self, targets: np.ndarray) -> np.ndarray:
        """Add blanks between, before, and after targets"""
        L = len(targets)
        extended = np.zeros(2 * L + 1, dtype=np.int32)
        extended[1::2] = targets
        return extended

    def check_inputs(self, log_probs: np.ndarray, targets: np.ndarray, input_lengths: np.ndarray, target_lengths: np.ndarray) -> None:
        """Validate input shapes and values"""
        if log_probs.ndim not in [2, 3]:
            raise ValueError(f"log_probs must be 2D or 3D, got shape {log_probs.shape}")

        if not np.all(np.isfinite(log_probs)):
            raise ValueError("log_probs contains inf or nan values")

        if np.any(input_lengths < 1) or np.any(target_lengths < 1):
            raise ValueError("All sequence lengths must be positive")

        # For single sequence
        if log_probs.ndim == 2:
            T = log_probs.shape[0]
            if T < 2 * len(targets) + 1:
                raise ValueError(
                    f"Input sequence length ({T}) must be at least 2 * target_length + 1 "
                    f"({2 * len(targets) + 1})"
                )
        # For batched input
        else:
            batch_size = log_probs.shape[0]
            for b in range(batch_size):
                T = input_lengths[b]
                L = target_lengths[b]
                if T < 2 * L + 1:
                    raise ValueError(
                        f"Sequence {b}: Input length ({T}) must be at least 2 * target_length + 1 "
                        f"({2 * L + 1})"
                    )

    def compute_forward_vars(self, log_probs: np.ndarray, extended_targets: np.ndarray, T: int) -> np.ndarray:
        """Compute forward variables efficiently"""
        alpha = np.full((T, len(extended_targets)), self.NINF)

        # Initialize first timestep
        alpha[0, 0] = log_probs[0, self.blank_id]
        if len(extended_targets) > 1:
            alpha[0, 1] = log_probs[0, extended_targets[1]]

        # Forward pass with vectorized operations where possible
        for t in range(1, T):
            for s in range(len(extended_targets)):
                current_label = extended_targets[s]
                paths = [alpha[t - 1, s]]

                if s >= 1:
                    paths.append(alpha[t - 1, s - 1])
                if s >= 2 and extended_targets[s] != extended_targets[s - 2]:
                    paths.append(alpha[t - 1, s - 2])

                alpha[t, s] = np.logaddexp.reduce(paths) + log_probs[t, current_label]

        return alpha

    def compute_backward_vars(self, log_probs: np.ndarray, extended_targets: np.ndarray, T: int) -> np.ndarray:
        """Compute backward variables efficiently"""
        beta = np.full((T, len(extended_targets)), self.NINF)
        beta[-1, -1] = 0
        beta[-1, -2] = 0

        for t in range(T - 2, -1, -1):
            for s in range(len(extended_targets)):
                paths = [beta[t + 1, s] + log_probs[t + 1, extended_targets[s]]]

                if s < len(extended_targets) - 1:
                    paths.append(beta[t + 1, s + 1] + log_probs[t + 1, extended_targets[s + 1]])
                if s < len(extended_targets) - 2 and extended_targets[s] != extended_targets[s + 2]:
                    paths.append(beta[t + 1, s + 2] + log_probs[t + 1, extended_targets[s + 2]])

                beta[t, s] = np.logaddexp.reduce(paths)

        return beta

    def forward(self, log_probs: np.ndarray, targets: np.ndarray, input_lengths: np.ndarray, target_lengths: np.ndarray) -> CTCState:
        """
        Compute CTC forward pass and loss
        """
        self.check_inputs(log_probs, targets, input_lengths, target_lengths)

        # Handle batched input
        if log_probs.ndim == 3:
            batch_size = log_probs.shape[0]
            states = []
            losses = []

            for b in range(batch_size):
                state = self.forward(
                    log_probs[b, :input_lengths[b]],
                    targets[b, :target_lengths[b]],
                    input_lengths[b:b + 1],
                    target_lengths[b:b + 1]
                )
                states.append(state)
                losses.append(state.loss)

            # Apply reduction
            if self.reduction == 'mean':
                total_loss = np.mean(losses)
            elif self.reduction == 'sum':
                total_loss = np.sum(losses)
            else:
                total_loss = np.array(losses)

            return CTCState(
                alpha=np.stack([s.alpha for s in states]),
                loss=total_loss
            )

        # Single sequence processing
        T = log_probs.shape[0]
        extended_targets = self.extend_targets(targets)
        alpha = self.compute_forward_vars(log_probs, extended_targets, T)
        loss = -np.logaddexp(alpha[-1, -1], alpha[-1, -2])

        return CTCState(alpha=alpha, loss=loss)

    def backward(self, state: CTCState, log_probs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Compute CTC backward pass and gradients
        """
        T = log_probs.shape[0]
        extended_targets = self.extend_targets(targets)

        # Compute backward variables if not already computed
        if state.beta is None:
            state.beta = self.compute_backward_vars(log_probs, extended_targets, T)

        # Compute full sequence probability
        full_prob = np.logaddexp(state.alpha[-1, -1], state.alpha[-1, -2])

        # Compute gradients efficiently
        grad = np.zeros_like(log_probs)
        for t in range(T):
            label_probs = {}
            for s, label in enumerate(extended_targets):
                if np.isfinite(state.alpha[t, s]) and np.isfinite(state.beta[t, s]):
                    prob = state.alpha[t, s] + state.beta[t, s] - full_prob
                    label_probs[label] = np.logaddexp(
                        label_probs.get(label, self.NINF),
                        prob
                    )

            for label, prob in label_probs.items():
                grad[t, label] = -np.exp(prob)

            grad[t] += np.exp(log_probs[t])

        return grad


class TestCTC(unittest.TestCase):
    def setUp(self):
        self.ctc = CTCLoss(blank_id=0)
        self.T, self.C = 5, 4
        self.targets = np.array([1, 2])

        # Create controlled probabilities
        probs = np.ones((self.T, self.C)) * 0.1
        probs[0, 0] = probs[2, 0] = probs[4, 0] = 0.9  # blanks
        probs[1, 1] = 0.9  # first target
        probs[3, 2] = 0.9  # second target
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        self.log_probs = np.log(probs)

        self.input_lengths = np.array([self.T])
        self.target_lengths = np.array([len(self.targets)])

    def test_basic_properties(self):
        state = self.ctc.forward(self.log_probs, self.targets,
                                 self.input_lengths, self.target_lengths)
        self.assertTrue(np.isfinite(state.loss))
        self.assertTrue(state.loss >= 0)

        grad = self.ctc.backward(state, self.log_probs, self.targets)
        self.assertTrue(np.allclose(np.sum(grad, axis=1), 0, atol=1e-6))

    def test_batch_processing(self):
        batch_size = 2
        batched_log_probs = np.stack([self.log_probs] * batch_size)
        batched_targets = np.stack([self.targets] * batch_size)
        batched_input_lengths = np.array([self.T] * batch_size)
        batched_target_lengths = np.array([len(self.targets)] * batch_size)

        state = self.ctc.forward(batched_log_probs, batched_targets,
                                 batched_input_lengths, batched_target_lengths)
        self.assertEqual(state.alpha.shape[0], batch_size)

    def test_error_handling(self):
        """Test various error conditions"""
        # Test input sequence too short
        short_log_probs = self.log_probs[:2]  # Only 2 timesteps
        with self.assertRaises(ValueError) as context:
            self.ctc.forward(
                short_log_probs,
                self.targets,  # length 2
                np.array([2]),  # input length 2
                np.array([2])  # target length 2
            )
        self.assertTrue("Input sequence length" in str(context.exception))

        # Test invalid input dimensions
        invalid_log_probs = np.random.rand(2, 3, 4, 5)  # 4D array
        with self.assertRaises(ValueError) as context:
            self.ctc.forward(
                invalid_log_probs,
                self.targets,
                self.input_lengths,
                self.target_lengths
            )
        self.assertTrue("must be 2D or 3D" in str(context.exception))

        # Test negative sequence lengths
        with self.assertRaises(ValueError) as context:
            self.ctc.forward(
                self.log_probs,
                self.targets,
                np.array([-1]),  # negative input length
                self.target_lengths
            )
        self.assertTrue("must be positive" in str(context.exception))


if __name__ == "__main__":
    # Example usage
    print("Running example with improved CTC implementation...")

    # Create sample data
    T, C = 10, 5
    batch_size = 2
    log_probs = np.log(softmax(np.random.rand(batch_size, T, C), axis=2))
    targets = np.array([[1, 2], [2, 3]])
    input_lengths = np.array([T, T])
    target_lengths = np.array([2, 2])

    # Initialize CTC
    ctc = CTCLoss(blank_id=0, reduction='mean')

    # Forward pass
    start_time = time()
    state = ctc.forward(log_probs, targets, input_lengths, target_lengths)

    # Backward pass (for first sequence)
    grad = ctc.backward(CTCState(alpha=state.alpha[0]), log_probs[0], targets[0])

    print(f"Batch CTC Loss: {state.loss:.4f}")
    print(f"Gradient shape: {grad.shape}")
    print(f"Gradient stats:")
    print(f"Mean: {np.mean(grad):.4f}")
    print(f"Std: {np.std(grad):.4f}")
    print(f"Min: {np.min(grad):.4f}")
    print(f"Max: {np.max(grad):.4f}")
    print(f"Execution Time: {time() - start_time:.4f} seconds")

    print("\nRunning unit tests...")
    unittest.main(argv=[''], exit=False)