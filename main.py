import re
from collections import defaultdict, Counter


def load_text(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()


def tokenize(text):
    text = text.lower()

    words = re.findall(r"\b[a-zA-Z]+\b", text)

    return words


def generate_bigrams(words):
    bigrams = []

    for i in range(len(words) - 1):
        bigram = (words[i], words[i + 1])
        bigrams.append(bigram)

    return bigrams


def build_bigram_model(bigrams):
    bigram_counts = defaultdict(Counter)

    for first_word, second_word in bigrams:
        bigram_counts[first_word][second_word] += 1

    return bigram_counts


def build_word_counts(bigrams):
    word_counts = Counter()

    for first_word, second_word in bigrams:
        word_counts[first_word] += 1

    return word_counts


def calculate_probabilities(bigram_counts, word_counts):
    probabilities = defaultdict(dict)

    for first_word in bigram_counts:
        total_count = word_counts[first_word]

        for second_word, count in bigram_counts[first_word].items():
            probabilities[first_word][second_word] = (
                count / total_count
            )

    return probabilities


def suggest_words(word, probabilities, top_k=5):
    word = word.lower()

    if word not in probabilities:
        return []

    suggestions = list(probabilities[word].items())

    suggestions.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return suggestions[:top_k]


def main():
    print("=" * 50)
    print("       BIGRAM WORD SUGGESTION TOOL")
    print("=" * 50)

    filename = "data/training.txt"

    print("\nLoading training data...")

    text = load_text(filename)

    words = tokenize(text)

    print(f"Total words: {len(words)}")
    print(f"Vocabulary size: {len(set(words))}")

    print("\nGenerating bigrams...")

    bigrams = generate_bigrams(words)

    print(f"Total bigrams: {len(bigrams)}")

    print("\nBuilding model...")

    bigram_counts = build_bigram_model(bigrams)

    word_counts = build_word_counts(bigrams)

    probabilities = calculate_probabilities(
        bigram_counts,
        word_counts
    )

    print("Model trained successfully!")

    print("\nType 'exit' to quit.")

    while True:
        sentence = input("\nEnter a sentence: ")

        if sentence.lower() == "exit":
            print("Goodbye!")
            break

        input_words = tokenize(sentence)

        if not input_words:
            print("Please enter some text.")
            continue

        last_word = input_words[-1]

        suggestions = suggest_words(
            last_word,
            probabilities,
            top_k=5
        )

        if not suggestions:
            print(
                f"No suggestions found for '{last_word}'."
            )
            continue

        print(f"\nSuggestions after '{last_word}':")

        for i, (word, probability) in enumerate(
            suggestions,
            start=1
        ):
            print(
                f"{i}. {word:<15}"
                f"{probability * 100:.2f}%"
            )


if __name__ == "__main__":
    main()
