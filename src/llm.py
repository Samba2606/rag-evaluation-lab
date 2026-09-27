import os

from groq import Groq

from src.config import MODEL_NAME

def _client():
    key = os.getenv("GROQ_API_KEY")

    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is missing."
        )

    return Groq(api_key=key)

def answer_question(
    question: str,
    contexts: list[dict],
) -> str:
    context_text = "\n\n".join(
        (
            f"[Source {i + 1} | "
            f"page={context.get('page')}]\n"
            f"{context['text']}"
        )
        for i, context in enumerate(contexts)
    )

    prompt = f"""
Answer only from the supplied context.

If the answer is not present, say:
"I don't have enough information in the document."

Use source markers such as [Source 1].

Context:
{context_text}

Question:
{question}
"""

    response = _client().chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful document QA assistant."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return response.choices[0].message.content

def groundedness_judge(
    question: str,
    answer: str,
    contexts: list[dict],
) -> dict:
    context_text = "\n".join(
        context["text"]
        for context in contexts
    )

    prompt = f"""
Evaluate whether the answer is supported by the context.

Question:
{question}

Answer:
{answer}

Context:
{context_text}

Return exactly:
score=<integer from 0 to 100>
reason=<short reason>
"""

    response = _client().chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    text = response.choices[0].message.content or ""
    score = 0
    reason = text

    for line in text.splitlines():
        if line.lower().startswith("score="):
            try:
                score = int(
                    line.split("=", 1)[1].strip()
                )
                score = max(
                    0,
                    min(100, score),
                )
            except ValueError:
                pass

        if line.lower().startswith("reason="):
            reason = line.split(
                "=",
                1,
            )[1].strip()

    return {
        "score": score,
        "reason": reason,
    }
