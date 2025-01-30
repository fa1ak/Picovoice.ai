import numpy as np
from scipy.special import softmax

def ctc_forward(log_probs, targets, input_lengths, target_lengths):
    """
    Compute the CTC loss using the forward algorithm.

    Args:
        log_probs: (T, C) matrix of log probabilities (T = time steps, C = classes including blank)
        targets: Target sequence (L,)
        input_lengths: (T,)
        target_lengths: (L,)

    Returns:
        loss: CTC loss
        alpha: Forward probabilities (for backward pass)
    """
    T, C = log_probs.shape
    L = len(targets)

    extended_targets = np.zeros(2 * L + 1, dtype=np.int32)
    extended_targets[1::2] = targets
    blank = 0  # Assuming 0 is the blank label

    # Initializing alpha - forward probabilities
    alpha = np.full((T, len(extended_targets)), -np.inf)
    alpha[0, 0] = log_probs[0, blank]
    alpha[0, 1] = log_probs[0, extended_targets[1]]

    for t in range(1, T):
        prev_alpha = alpha[t - 1]
        alpha[t] = log_probs[t, extended_targets] + np.logaddexp.reduce(
            [prev_alpha, np.roll(prev_alpha, 1), np.roll(prev_alpha, 2)], axis=0)

        # Compute loss
    loss = -np.logaddexp(alpha[-1, -1], alpha[-1, -2])

    return loss, alpha

def ctc_backward(log_probs, targets, input_lengths, target_lengths, alpha):
    """
    Compute the CTC gradient using the backward algorithm.

    Args:
        log_probs: (T, C) matrix of log probabilities.
        targets: Target sequence (L,)
        alpha: Forward probabilities from the forward pass.

    Returns:
        grad: Gradients of the loss with respect to log_probs.
    """
    T, C = log_probs.shape
    L = len(targets)

    extended_targets = np.zeros(2 * L + 1, dtype=np.int32)
    extended_targets[1::2] = targets
    blank = 0

    # Initializing beta - backward probabilities
    beta = np.full((T, len(extended_targets)), -np.inf)
    beta[T - 1, -1] = log_probs[T - 1, blank]
    beta[T - 1, -2] = log_probs[T - 1, extended_targets[-2]]

    for t in range(T - 2, -1, -1):
        next_beta = beta[t + 1]
        beta[t] = log_probs[t, extended_targets] + np.logaddexp.reduce(
            [next_beta, np.roll(next_beta, -1), np.roll(next_beta, -2)], axis=0)

        # Compute gradients
    posterior_probs = np.exp(alpha + beta - np.logaddexp(alpha[-1, -1], alpha[-1, -2]))
    grad = np.zeros_like(log_probs)

    for t in range(T):
        for i in range(len(extended_targets)):
            grad[t, extended_targets[i]] -= posterior_probs[t, i]

    return grad

# --------------------------
# TEST CASE
# --------------------------
if __name__ == "__main__":
    from time import time

    T, C = 10, 5
    log_probs = np.log(softmax(np.random.rand(T, C), axis=1))  # Log probabilities
    targets = np.array([1, 2])  # Target sequence
    input_lengths = np.array([T])
    target_lengths = np.array([len(targets)])

    # Measuring execution time
    start_time = time()

    loss, alpha = ctc_forward(log_probs, targets, input_lengths, target_lengths)
    print(f"CTC Loss: {loss:.4f}")

    # Compute gradients
    grad = ctc_backward(log_probs, targets, input_lengths, target_lengths, alpha)
    print("Gradients:\n", grad)

    print(f"Execution Time: {time() - start_time:.4f} seconds")