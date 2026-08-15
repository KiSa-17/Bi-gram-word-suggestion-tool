import re
from collections import defaultdict, Counter


TRAINING_FILE = "data/training.txt"
USER_TRAINING_FILE = "data/user_training.txt"


def load_text(filename):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return ""


def save_user_text(text, filename):
    with open(filename, "a", encoding="utf-8") as file:
        file.write(text.strip() + "\n")


def tokenize(text):
    text = text.lower()

    words = re.findall(
        r"\b[a-zA-Z]+\b",
        text
    )

    return words


def build_model(words):
    bigram_counts = defaultdict(Counter)
    word_counts = Counter()

    for i in range(len(words) - 1):
        first_word = words[i]
        second_word = words[i + 1]

        bigram_counts[first_word][second_word] += 1
        word_counts[first_word] += 1

    return bigram_counts, word_counts


def learn_from_text(text, bigram_counts, word_counts):
    words = tokenize(text)

    for i in range(len(words) - 1):
        first_word = words[i]
        second_word = words[i + 1]

        bigram_counts[first_word][second_word] += 1
        word_counts[first_word] += 1


def suggest_words(
    word,
    bigram_counts,
    word_counts,
    top_k=5
):
    word = word.lower()

    if word not in bigram_counts:
        return []

    total_count = word_counts[word]

    suggestions = []

    for next_word, count in bigram_counts[word].items():
        probability = count / total_count

        suggestions.append(
            (next_word, probability, count)
        )

    suggestions.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return suggestions[:top_k]


def show_suggestions(
    word,
    bigram_counts,
    word_counts,
    top_k=5
):
    suggestions = suggest_words(
        word,
        bigram_counts,
        word_counts,
        top_k
    )

    if not suggestions:
        print(
            f"\nNo suggestions found after "
            f"'{word}'."
        )
        return

    print(
        f"\nSuggestions after "
        f"'{word}':"
    )

    print("-" * 45)

    for i, (next_word, probability, count) in enumerate(
        suggestions,
        start=1
    ):
        print(
            f"{i}. {next_word:<20}"
            f"{probability * 100:>7.2f}%"
            f"  Count: {count}"
        )

    print("-" * 45)


def show_model_statistics(
    words,
    bigram_counts,
    word_counts
):
    vocabulary_size = len(set(words))

    total_bigrams = sum(
        sum(counter.values())
        for counter in bigram_counts.values()
    )

    print("\nMODEL STATISTICS")
    print("=" * 45)
    print(f"Total words:       {len(words)}")
    print(f"Vocabulary size:   {vocabulary_size}")
    print(f"Total bigrams:     {total_bigrams}")
    print("=" * 45)


def show_bigrams(
    word,
    bigram_counts
):
    word = word.lower()

    if word not in bigram_counts:
        print(
            f"\nNo bigrams found for "
            f"'{word}'."
        )
        return

    print(
        f"\nWords occurring after "
        f"'{word}':"
    )

    print("-" * 35)

    results = bigram_counts[word].most_common()

    for next_word, count in results:
        print(
            f"{next_word:<20} {count}"
        )

    print("-" * 35)


def get_all_training_words():
    initial_text = load_text(
        TRAINING_FILE
    )

    user_text = load_text(
        USER_TRAINING_FILE
    )

    combined_text = (
        initial_text + "\n" + user_text
    )

    return tokenize(combined_text)


def main():

    print("=" * 60)
    print("             BIGRAM WORD SUGGESTION TOOL")
    print("=" * 60)

    print("\nLoading training data...")

    initial_text = load_text(
        TRAINING_FILE
    )

    user_text = load_text(
        USER_TRAINING_FILE
    )

    combined_text = (
        initial_text + "\n" + user_text
    )

    if not combined_text.strip():
        print(
            "\nNo training data found."
        )
        print(
            f"Please add text to:"
            f"\n{TRAINING_FILE}"
        )
        return

    words = tokenize(
        combined_text
    )

    print(
        f"Total training words: "
        f"{len(words)}"
    )

    print(
        f"Vocabulary size: "
        f"{len(set(words))}"
    )

    print("\nBuilding Bigram Model...")

    bigram_counts, word_counts = build_model(
        words
    )

    print("Model trained successfully!")

    print("\n" + "=" * 60)

    print("COMMANDS")
    print("=" * 60)
    print("/stats              Show model statistics")
    print("/model <word>       Show learned bigrams")
    print("/suggest <word>     Show suggestions")
    print("/exit               Exit program")
    print("=" * 60)

    while True:

        user_input = input(
            "\nYou: "
        ).strip()

        if not user_input:
            print(
                "Please enter some text."
            )
            continue

        # Exit command
        if user_input.lower() == "/exit":
            print(
                "\nModel session ended."
            )
            break

        # Statistics command
        if user_input.lower() == "/stats":
            all_words = get_all_training_words()

            show_model_statistics(
                all_words,
                bigram_counts,
                word_counts
            )

            continue

        # Model inspection command
        if user_input.lower().startswith("/model "):

            parts = user_input.split(
                maxsplit=1
            )

            if len(parts) < 2:
                print(
                    "Usage: /model <word>"
                )
                continue

            word = parts[1].strip()

            show_bigrams(
                word,
                bigram_counts
            )

            continue

        # Suggestion command
        if user_input.lower().startswith("/suggest "):

            parts = user_input.split(
                maxsplit=1
            )

            if len(parts) < 2:
                print(
                    "Usage: /suggest <word>"
                )
                continue

            word = parts[1].strip()

            show_suggestions(
                word,
                bigram_counts,
                word_counts
            )

            continue

        # Tokenize user input
        input_words = tokenize(
            user_input
        )

        if not input_words:
            print(
                "No valid words found."
            )
            continue

        # Save user input permanently
        save_user_text(
            user_input,
            USER_TRAINING_FILE
        )

        # Update model immediately
        learn_from_text(
            user_input,
            bigram_counts,
            word_counts
        )

        print(
            "✓ Model updated with your input."
        )

        # Get the last word
        last_word = input_words[-1]

        # Generate suggestions
        show_suggestions(
            last_word,
            bigram_counts,
            word_counts
        )


if __name__ == "__main__":
    main()
