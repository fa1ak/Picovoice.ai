from typing import Sequence, Dict, List, Tuple
from collections import defaultdict

"""
Question:
A phoneme is a sound unit (similar to a character for text). We have an extensive pronunciation dictionary.
Given a sequence of phonemes as input (e.g. ["DH", "EH", "R", "DH", "EH", "R"]),
find all the combinations of the words that can produce this sequence.
Example Output:
    [["THEIR", "THEIR"], ["THEIR", "THERE"], ["THERE", "THEIR"], ["THERE", "THERE"]]
You can preprocess the dictionary into a different data structure if needed.
"""

class PhonemeWordMapper:
    def __init__(self, pronunciation_dict: Dict[str, List[str]]):
        """
        Initializes the phoneme-word mapper with a given pronunciation dictionary.
        :param pronunciation_dict: Dictionary where keys are words, values are phoneme sequences.
        """
        self.phoneme_to_words = self._preprocess_dict(pronunciation_dict)

    def _preprocess_dict(self, pronunciation_dict: Dict[str, List[str]]) -> Dict[Tuple[str, ...], List[str]]:
        """
        Converts a word-to-phoneme dictionary into a phoneme-to-word mapping.
        :param pronunciation_dict: Original dictionary mapping words to phoneme sequences.
        :return: A dictionary mapping phoneme sequences to words.
        """
        phoneme_dict = defaultdict(list)
        for word, phonemes in pronunciation_dict.items():
            phoneme_dict[tuple(phonemes)].append(word)
        return phoneme_dict

    def find_word_combos_with_pronunciation(self, phonemes: Sequence[str]) -> Sequence[Sequence[str]]:
        """
        Finds all word combinations that can produce the given phoneme sequence.
        :param phonemes: A list of phonemes that need to be matched with words.
        :return: A list of possible word sequences that match the phoneme sequence.
        """
        results = []
        self._dfs(phonemes, 0, [], results)
        return results

    def _dfs(self, phonemes: Sequence[str], index: int, path: List[str], results: List[List[str]]):
        """
        Uses backtracking (DFS) to find all valid word sequences that match the phoneme sequence.
        :param phonemes: The full sequence of phonemes.
        :param index: The current starting index for matching words.
        :param path: The current sequence of words being formed.
        :param results: Stores all valid word sequences.
        """
        if index == len(phonemes):  # If all phonemes are matched, save the result
            results.append(path[:])
            return

        for length in range(1, len(phonemes) - index + 1):
            sub_phonemes = tuple(phonemes[index:index + length])
            if sub_phonemes in self.phoneme_to_words:
                for word in self.phoneme_to_words[sub_phonemes]:
                    path.append(word)
                    self._dfs(phonemes, index + length, path, results)
                    path.pop()  # Backtrack

# ----------------------------------
# TEST CASES
# ----------------------------------

if __name__ == "__main__":
    # Sample pronunciation dictionary
    pronunciation_dict = {
        "ABACUS": ["AE", "B", "AH", "K", "AH", "S"],
        "BOOK": ["B", "UH", "K"],
        "THEIR": ["DH", "EH", "R"],
        "THERE": ["DH", "EH", "R"],
        "TOMATO": ["T", "AH", "M", "AA", "T", "OW"],
        "TOMATO": ["T", "AH", "M", "EY", "T", "OW"]
    }
    data = [{
        "ABACUS": []
    },
        {

        }]

    # Initialize the mapper
    mapper = PhonemeWordMapper(pronunciation_dict)

    # Example input sequence of phonemes
    input_phonemes = ["T", "AH", "M", "EY", "T", "OW"]

    # Find valid word sequences
    results = mapper.find_word_combos_with_pronunciation(input_phonemes)

    # Print results
    for i, result in enumerate(results):
        print(f"Word Combination {i+1}: {result}")