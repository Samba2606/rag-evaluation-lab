import re

def lexical_overlap(
    answer: str,
    contexts: list[dict],
) -> float:
    def words(text: str) -> set[str]:
        return set(
            re.findall(
                r"[a-zA-Z]{3,}",
                text.lower(),
            )
        )

    answer_words = words(answer)
    context_words = words(
        " ".join(
            context["text"]
            for context in contexts
        )
    )

    if not answer_words:
        return 0.0

    return (
        len(answer_words & context_words)
        / len(answer_words)
    )
