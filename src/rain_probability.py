from typing import Sequence
from scipy.stats import norm

"""
Question:
The probability of rain on a given calendar day in Vancouver is p[i], where i is the day's index. 
For example, p[0] is the probability of rain on January 1st, and p[10] is the probability of precipitation on January 11th. 
Assume the year has 365 days (i.e., p has 365 elements). What is the chance it rains more than n (e.g., 100) days in Vancouver?
Write a function that accepts p (probabilities of rain on a given calendar day) and n as input arguments 
and returns the possibility of raining at least n days.
"""


def prob_rain_more_than_n(p: Sequence[float], n: int) -> float:
    """
    Computes the probability of raining at least 'n' days in a year given daily rain probabilities.
    """
    # Compute expected rainy days i.e. mean.
    mu = sum(p)

    # Compute variance.
    sigma_sq = sum(pi * (1 - pi) for pi in p)
    sigma = sigma_sq ** 0.5  # Standard deviation

    # Use normal approximation to get probability P(X ≥ n).
    probability = 1 - norm.cdf(n, loc=mu, scale=sigma)

    return probability

# ----------------------------------
# TEST CASES
# ----------------------------------

if __name__ == "__main__":
    test_p1 = [0.3] * 365  # Each day has a 30% probability of rain.
    test_n1 = 100
    print(f"Test Case 1: {prob_rain_more_than_n(test_p1, test_n1):.4f}")

    test_p2 = [0.5] * 365  # Each day has a 50% probability of rain.
    test_n2 = 150
    print(f"Test Case 2: {prob_rain_more_than_n(test_p2, test_n2):.4f}")

    test_p3 = [0.2] * 365  # Each day has a 20% probability of rain.
    test_n3 = 80
    print(f"Test Case 3: {prob_rain_more_than_n(test_p3, test_n3):.4f}")