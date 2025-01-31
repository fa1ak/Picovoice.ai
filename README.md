# Picovoice.ai

## Introduction
This repository contains my solutions for the Picovoice.ai coding screening test. Each solution is thoughtfully implemented in Python, thoroughly tested, and optimized for performance.

## Table of Contents

- [Problem 1: Rain Probability](#problem-1-rain-probability)
- [Problem 2: Phoneme Word Mapping](#problem-2-phoneme-word-mapping)
- [Problem 3: Connectionist Temporal Classification (CTC) Loss](#problem-3-connectionist-temporal-classification-ctc-loss)
- *Bonus question* [Problem 4: Most Frequent Words in Shakespeare’s Works](#problem-4-most-frequent-words-in-shakespeares-works) 
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
1. **Input Validation & Preprocessing**
   - Introduced a `check_inputs` function to **validate input shapes and values**.
   - Implemented **`extend_targets`** to add blank labels between phonemes.
   
2. **Efficient Forward-Backward Computation**
   - **Forward Pass (`alpha` calculation)**
     - Used **log-space computations** to prevent numerical underflow.
     - Optimized with **vectorized NumPy operations** instead of nested loops.
   - **Backward Pass (`beta` calculation)**
     - Similar optimization using log-space operations.

3. **Batch Processing Support**
   - Implemented **batched forward and backward passes**.
   - Allowed **different reduction methods** (`mean`, `sum`, or `none`).

4. **Gradient Calculation**
   - Computed gradients using **log-add-exp trick** for numerical stability.
   - Ensured proper probability mass distribution.

5. **Error Handling & Robustness**
   - **Edge cases**: Short sequences, invalid dimensions, and incorrect target alignments.
   - Exception handling for **NaN or infinite values**.

### Challenges & Solutions:
- **Roadblock:** Initial implementation was **too slow**.
- **Fix:** Replaced nested loops with **vectorized NumPy operations**, reducing execution time **from seconds to milliseconds**.

### Performance Comparison:
| Version         | Execution Time |
|----------------|---------------|
| Initial Code   | 5.23 seconds  |
| Optimized Code | 0.0005 seconds |


<img width="574" alt="Screenshot 2025-01-30 at 22 21 48" src="https://github.com/user-attachments/assets/1e4d4e3e-74b8-49e5-986e-9bff9529059f" />


---

## **Problem 4: Most Frequent Words in Shakespeare’s Works**

While I wasn’t initially asked to implement all 4 problems, I took it on as an additional challenge.

### **Problem Statement**
The goal is to **find the `n` most frequent words** in the **TensorFlow Shakespeare dataset** https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt.  
## **Problem 4: Most Frequent Words in Shakespeare’s Works**
### **Goal**: Identify the `n` most frequent words in Shakespeare’s dataset (`shakespeare.txt`).

### **Approach**
1. **Preprocessing**: Convert all words to lowercase and remove punctuation for consistent word matching.
2. **Efficient Word Counting**: Store word frequencies in a **Word Frequency Array (`WordFreq`)**.
3. **Sorting**: Use **QuickSort (`qsort`)** to sort words in **descending order** of frequency.
4. **Top `n` Extraction**: Display the `n` most frequent words.

### **Challenges & Solutions**
- **Roadblock**: Handling large text efficiently while maintaining memory constraints.
- **Fix**: Used **dynamic allocation (`malloc`)** and limited storage to `MAX_WORDS` for optimized memory usage.
  
<img width="460" alt="Screenshot 2025-01-30 at 23 16 28" src="https://github.com/user-attachments/assets/164a2bda-9065-4e41-9536-ad060dc562ea" />

---

## **Assumptions & Edge Cases**
### **Assumptions:**
- **Problem 1 (Rain Probability):** Daily rain probabilities are **independent** events.
- **Problem 2 (Phoneme Matching):** The input phoneme sequence is always **valid**.
- **Problem 3 (CTC Loss):** Input sequences are **long enough** to align with the target sequence.
- **Problem 4 (Word Frequency Analysis):** The dataset (`shakespeare.txt`) is **properly formatted** and **large word counts are manageable**.

### **Edge Cases Considered:**
- **Rain Probability:** What if `n > 365`? The probability should return **0**.
- **Phoneme Matching:** What if a phoneme has **no corresponding word**? It should return **an empty result**.
- **CTC Loss:** Tested with `T >> L` to ensure **sequence mismatch handling**.
- **Word Frequency Analysis:** What if `n` is **greater than the number of unique words**? The program should **return all words** without errors.

---

## **📌 Installation & Usage**
```bash
pip install -r requirements.txt
python src/rain_probability.py
python src/phoneme_word_mapping.py
python src/ctc_loss.py
gcc -o src/word_freq src/word_freq.c
./src/word_freq
```

---

## Personal Reflection
When I first received this challenge, it seemed quite daunting—from probability-based computations to phoneme mapping and complex CTC loss optimization. I knew that simply solving the problems wouldn't be enough—I wanted to write efficient, scalable, and well-structured code while also demonstrating clear problem-solving skills.

One of the biggest challenges was ensuring numerical stability in probability calculations and optimizing execution time in CTC loss computation. Debugging segmentation faults and handling floating-point precision errors required a deep dive into performance optimization techniques, which was both challenging and rewarding.

While I was asked to only perform 3 questions, I decided to take it on as an additional challenge and solved the Most Frequent Words in Shakespeare’s Works problem.

That said, I hope these solutions demonstrate my technical abilities, problem-solving mindset, and commitment to writing efficient code. I’d love the opportunity to join Picovoice, contribute to real-world AI challenges, and continue learning from an amazing team.

Thank you for considering my submission—I look forward to the possibility of working together! 
