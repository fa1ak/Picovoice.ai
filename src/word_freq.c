#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_WORDS 100000  // Maximum words to track
#define MAX_WORD_LENGTH 100
#define FILE_NAME "shakespeare.txt"

typedef struct {
    char word[MAX_WORD_LENGTH];
    int frequency;
} WordFreq;

void to_lowercase(char *str) {
    while (*str) {
        *str = tolower((unsigned char)*str);
        str++;
    }
}

void remove_punctuation(char *str) {
    char *src = str, *dst = str;
    while (*src) {
        if (isalpha((unsigned char)*src) || isspace((unsigned char)*src)) {
            *dst++ = *src;
        }
        src++;
    }
    *dst = '\0';
}

int compare(const void *a, const void *b) {
    return ((WordFreq *)b)->frequency - ((WordFreq *)a)->frequency;
}

// finding the 'n' most frequent words in a file
WordFreq *find_frequent_words(const char *path, int32_t n, int *actual_count) {
    FILE *file = fopen(path, "r");
    if (!file) {
        printf("Error: Could not open file: %s\n", path);
        return NULL;
    }

    WordFreq *words = (WordFreq *)malloc(MAX_WORDS * sizeof(WordFreq));
    if (!words) {
        printf("Memory allocation failed!\n");
        fclose(file);
        return NULL;
    }

    int word_count = 0;
    char buffer[MAX_WORD_LENGTH];

    while (fscanf(file, "%99s", buffer) == 1) {
        remove_punctuation(buffer);
        to_lowercase(buffer);

        if (strlen(buffer) == 0) continue;

        // Checking if word already exists in the list
        int found = 0;
        for (int i = 0; i < word_count; i++) {
            if (strcmp(words[i].word, buffer) == 0) {
                words[i].frequency++;
                found = 1;
                break;
            }
        }

        // If new word, add to list
        if (!found) {
            if (word_count >= MAX_WORDS) {
                printf("Warning: Exceeded maximum word storage capacity!\n");
                break;
            }
            strncpy(words[word_count].word, buffer, MAX_WORD_LENGTH - 1);
            words[word_count].word[MAX_WORD_LENGTH - 1] = '\0'; // Null-terminate
            words[word_count].frequency = 1;
            word_count++;
        }
    }

    fclose(file);

    qsort(words, word_count, sizeof(WordFreq), compare);

    *actual_count = (n < word_count) ? n : word_count;

    return words;
}

int main() {
    int n;

    printf("Enter the number of most frequent words to display: ");
    if (scanf("%d", &n) != 1 || n <= 0) {
        printf("Invalid input! Please enter a positive integer.\n");
        return 1;
    }

    int actual_count;
    WordFreq *frequent_words = find_frequent_words(FILE_NAME, n, &actual_count);

    if (frequent_words) {
        printf("\nTop %d most frequent words:\n", actual_count);
        for (int i = 0; i < actual_count; i++) {
            printf("%s - %d times\n", frequent_words[i].word, frequent_words[i].frequency);
        }
        free(frequent_words);
    }

    return 0;
}
