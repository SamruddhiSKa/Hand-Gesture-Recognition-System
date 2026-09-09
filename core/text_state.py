"""Pure word and sentence state operations used by the Streamlit UI."""


def add_letter(word: str, letter: str) -> str:
    return word + letter if letter else word


def delete_letter(word: str) -> str:
    return word[:-1]


def complete_word(word: str, completed_words: list[str]) -> tuple[str, list[str]]:
    if not word:
        return word, completed_words
    return "", [*completed_words, word]


def clear_text() -> tuple[str, list[str]]:
    return "", []