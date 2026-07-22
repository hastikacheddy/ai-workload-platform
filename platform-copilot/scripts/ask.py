"""Ask the copilot a question from the CLI (live: OpenSearch retrieval + Ollama).

Usage: python scripts/ask.py "what are the first steps for a drift alert?"
"""

from __future__ import annotations

import sys

from platform_copilot.dependencies import get_pipeline


def main() -> None:
    question = " ".join(sys.argv[1:]) or "What are the first steps for a demand-forecaster drift alert?"
    result = get_pipeline().answer(question)

    print(f"Q: {question}\n")
    print(f"ANSWER:\n{result.answer}\n")
    print("CITATIONS:")
    for citation in result.citations:
        print(f"  [{citation['n']}] {citation['source']}  ({citation['chunk_id']})")


if __name__ == "__main__":
    main()
