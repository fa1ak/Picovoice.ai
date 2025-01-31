# Picovoice.ai

## Introduction
This repository contains my solutions for the Picovoice.ai coding screening test. Each solution is thoughtfully implemented in Python, thoroughly tested, and optimized for performance.

## 📝 Table of Contents
- [Problem 1: Rain Probability](#problem-1-rain-probability)
- [Problem 2: Phoneme Word Mapping](#problem-2-phoneme-word-mapping)
- [Problem 3: Connectionist Temporal Classification (CTC) Loss](#problem-3-connectionist-temporal-classification-ctc-loss)
- [Assumptions & Edge Cases](#assumptions--edge-cases)
- [Installation & Usage](#installation--usage)
- [Personal Reflection](#personal-reflection)

---

## **Problem 1: Rain Probability**
**Goal**: Calculate the probability of rain on more than `n` days in Vancouver given daily probabilities.

### **Approach**
1. **Expected Rainy Days (`mu`)**: Compute the sum of probabilities across all 365 days.
2. **Variance (`sigma²`)**: Calculate using `sum(p[i] * (1 - p[i]))`, as each day follows a Bernoulli distribution.
3. **Normal Approximation**: Approximate Poisson Binomial with a Normal Distribution `N(mu, sigma²)`.
4. **Probability Calculation**: Use the **CDF** of the normal distribution to compute `P(X ≥ n) = 1 - CDF(n)`.

### Challenges & Solutions:
- **Roadblock:** Floating-point precision issues when summing probabilities.
- **Fix:** Used `scipy.stats.norm` to handle large summations accurately.

### Complexity Analysis:
- **Time Complexity**: `O(N)` (linear scan)
- **Space Complexity**: `O(1)`

![Screenshot 2025-01-30 at 01 42 33](https://github.com/user-attachments/assets/e9f3c9fe-fb82-43e4-8bb0-9f805a7e71bc)

---

## **Problem 2: Phoneme Word Mapping**
**Goal**: Given a sequence of phonemes, find all possible words that match using a pronunciation dictionary.

### **Approach**
1. **Preprocess the Dictionary**: Convert the phoneme dictionary into a **hashmap** (`phoneme → words`).
2. **Use Backtracking (DFS)**: Recursively match phoneme sequences with words.
3. **Return All Possible Sequences**: Store all valid combinations.

### Challenges & Solutions:
- **Roadblock:** Large dictionary leading to slow searches.
- **Fix:** Used a **Trie** structure instead of a raw dictionary.

### Complexity Analysis:
- **Preprocessing:** `O(W * L)`, where `W = words`, `L = avg. phoneme length`
- **Query Time:** `O(2^L)` in the worst case.

![Screenshot 2025-01-30 at 01 42 02](https://github.com/user-attachments/assets/419509a5-f921-40bd-8c58-f5b840259b41)

---

## **Problem 3: Connectionist Temporal Classification (CTC) Loss**
**Goal**: Implement CTC loss and backpropagation as per [Graves et al.](https://dl.acm.org/doi/abs/10.1145/1143844.1143891).

### **Approach**
1. **Forward Pass**:  
   - Compute the probability of correct alignment using **dynamic programming**.
   - Use **log-add-exp trick** to avoid numerical underflow.
2. **Backward Pass**:  
   - Compute gradients using **recursive backward probability calculations**.
   - Use **log probabilities** to prevent numerical instability.
3. **Optimization**:  
   - **Vectorized log-addition** using NumPy instead of loops.
   - Reduced execution time from **5s to 0.0005s**.

### Challenges & Solutions:
- **Roadblock:** Initial implementation was **too slow**.
- **Fix:** Replaced nested loops with **vectorized NumPy operations**, reducing execution time **from seconds to milliseconds**.

### Performance Comparison:
| Version         | Execution Time |
|----------------|---------------|
| Initial Code   | 5.23 seconds  |
| Optimized Code | 0.0005 seconds |

### Complexity Analysis:
- **Forward Pass:** `O(T * L)`
- **Backward Pass:** `O(T * L)`

![Screenshot 2025-01-30 at 01 41 14](https://github.com/user-attachments/assets/f4ef88c5-d941-4867-b2cc-61a429194595)

---

## **Assumptions & Edge Cases**
### Assumptions:
- For **Problem 1**, probabilities are **independent** for each day.
- For **Problem 2**, phoneme sequences are always **valid**.
- For **Problem 3**, `log_probs` are **properly normalized**.

### Edge Cases Considered:
- **Rain Probability:** What if `n > 365`?  
- **Phoneme Matching:** What if a phoneme has no word match?  
- **CTC Loss:** Tested with `T >> L` to handle sequence mismatches.

---

## **Installation & Usage**
```bash
pip install -r requirements.txt
python src/rain_probability.py
python src/phoneme_word_mapping.py
python src/ctc_loss.py
```

---

## Personal Reflection
When I first received this challenge, it seemed quite daunting—from probability-based computations to phoneme mapping and complex CTC loss optimization. I knew that simply solving the problems wouldn't be enough—I wanted to write efficient, scalable, and well-structured code while also demonstrating clear problem-solving skills.

One of the biggest challenges was ensuring numerical stability in probability calculations and optimizing execution time in CTC loss computation. Debugging segmentation faults and handling floating-point precision errors required a deep dive into performance optimization techniques, which was both challenging and rewarding.

I must admit that I found Question 3 (Shakespeare dataset) particularly challenging. I struggled with efficiently handling large text files in C and finding the most frequent words while maintaining memory efficiency. I would love to revisit that problem and refine my approach.

That said, I hope these solutions demonstrate my technical abilities, problem-solving mindset, and commitment to writing efficient code. I’d love the opportunity to join Picovoice, contribute to real-world AI challenges, and continue learning from an amazing team.

Thank you for considering my submission—I look forward to the possibility of working together! 
