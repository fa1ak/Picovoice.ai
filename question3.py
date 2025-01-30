import numpy as np


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
    blank = 0

    alpha = np.full((T, len(extended_targets)), -np.inf)
    alpha[0, 0] = log_probs[0, blank]
    alpha[0, 1] = log_probs[0, extended_targets[1]]

    for t in range(1, T):
        for i in range(len(extended_targets)):
            alpha[t, i] = log_probs[t, extended_targets[i]] + np.logaddexp(
                alpha[t - 1, i],
                alpha[t - 1, i - 1] if i > 0 else -np.inf
            )
            if i > 1 and extended_targets[i] != blank and extended_targets[i] != extended_targets[i - 2]:
                alpha[t, i] = np.logaddexp(alpha[t, i], alpha[t - 1, i - 2])

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

    beta = np.full((T, len(extended_targets)), -np.inf)
    beta[T - 1, -1] = log_probs[T - 1, blank]
    beta[T - 1, -2] = log_probs[T - 1, extended_targets[-2]]

    for t in range(T - 2, -1, -1):
        for i in range(len(extended_targets)):
            beta[t, i] = log_probs[t, extended_targets[i]] + np.logaddexp(
                beta[t + 1, i],
                beta[t + 1, i + 1] if i < len(extended_targets) - 1 else -np.inf
            )
            if i < len(extended_targets) - 2 and extended_targets[i] != blank and extended_targets[i] != \
                    extended_targets[i + 2]:
                beta[t, i] = np.logaddexp(beta[t, i], beta[t + 1, i + 2])

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
    T, C = 5, 4  # Example with 5 time steps, 4 classes
    log_probs = np.log(np.random.rand(T, C))  # Log probabilities
    targets = np.array([1, 2])  # Target sequence
    input_lengths = np.array([T])
    target_lengths = np.array([len(targets)])

    # Compute forward loss
    loss, alpha = ctc_forward(log_probs, targets, input_lengths, target_lengths)
    print(f"CTC Loss: {loss:.4f}")

    # Compute gradients
    grad = ctc_backward(log_probs, targets, input_lengths, target_lengths, alpha)
    print("Gradients:\n", grad)