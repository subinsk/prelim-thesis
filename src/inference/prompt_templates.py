import re


def create_cot_prompt(question: str, doc1: str, doc2: str) -> str:
    """Create Chain-of-Thought prompt for 2-hop multi-hop QA.

    Uses neutral document references to avoid biasing the model toward
    any particular document for answer extraction.
    """
    prompt = f"""Answer the following question using the provided documents. Think step by step, then provide your final answer.

Document 1:
{doc1}

Document 2:
{doc2}

Question: {question}

Think step by step, then state your final answer in the format: "The answer is <answer>".

Step-by-step reasoning:"""

    return prompt


def create_3hop_cot_prompt(question: str, doc1: str, doc2: str, doc3: str) -> str:
    """Create Chain-of-Thought prompt for 3-hop multi-hop QA.

    Uses neutral document references to avoid biasing the model toward
    any particular document for answer extraction.
    """
    return f"""Answer the following question using the provided documents. Think step by step, then provide your final answer.

Document 1:
{doc1}

Document 2:
{doc2}

Document 3:
{doc3}

Question: {question}

Think step by step, then state your final answer in the format: "The answer is <answer>".

Step-by-step reasoning:"""


def create_direct_prompt(question: str, doc1: str, doc2: str) -> str:
    """Create direct answering prompt."""
    prompt = f"""Based on the following documents, answer the question with just the answer, no explanation.

Document 1:
{doc1}

Document 2:
{doc2}

Question: {question}

Answer:"""

    return prompt


def strip_think_block(response: str) -> str:
    """Strip <think>...</think> blocks from reasoning models (e.g. Qwen3)."""
    return re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)


def extract_answer(response: str) -> str:
    """Extract final answer from model response.

    Handles:
    - "The answer is X" pattern (preferred)
    - "Answer: X" pattern
    - Fallback: last non-empty line
    - Comma-safe: doesn't split on commas inside numbers
    """
    response = strip_think_block(response)
    response_lower = response.lower()

    answer_part = None

    # Check for "the answer is X" pattern
    if "the answer is" in response_lower:
        idx = response_lower.rfind("the answer is")  # use LAST occurrence
        answer_part = response[idx + 13:].strip()
        # Remove leading colon if present
        if answer_part.startswith(':'):
            answer_part = answer_part[1:].strip()

    # Check for "Answer: X" pattern
    elif "answer:" in response_lower:
        idx = response_lower.rfind("answer:")  # use LAST occurrence
        answer_part = response[idx + 7:].strip()

    if answer_part is not None:
        # Trim at sentence boundary, but NOT at commas (preserves "9,984,670")
        # Only split at period followed by space or end, or newline
        for delimiter_pattern in [r'\.\s', r'\.\Z', r'\n']:
            match = re.search(delimiter_pattern, answer_part)
            if match:
                answer_part = answer_part[:match.start()].strip()
                break
        # Remove trailing period if any
        answer_part = answer_part.rstrip('.')
        return answer_part.strip()

    # Fallback: return last non-empty line
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    if lines:
        return lines[-1].rstrip('.')

    return response.strip()


if __name__ == "__main__":
    tests = [
        ("Step 1: ...\nStep 2: ...\nThe answer is 1970.", "1970"),
        ("The answer is Denver.", "Denver"),
        ("The answer is 9,984,670 square km.", "9,984,670 square km"),
        ("The answer is Chief of Protocol.", "Chief of Protocol"),
        ("Reasoning... Answer: New York City", "New York City"),
        ("The answer is from 1986 to 2013.", "from 1986 to 2013"),
        ("<think>lots of thinking</think>\nThe answer is London.", "London"),
        ("Step 1: Inception directed by Nolan.\nStep 2: Nolan born 1970.\nThe answer is 1970. This is confirmed.", "1970"),
        ("Some text\nDenver", "Denver"),  # fallback: last line
        ("The answer is: Greenwich Village, New York City.", "Greenwich Village, New York City"),
    ]

    print("Testing extract_answer:")
    for response, expected in tests:
        got = extract_answer(response)
        status = "PASS" if got == expected else "FAIL"
        print(f"  [{status}] got='{got}' expected='{expected}'")
        if got != expected:
            print(f"         input: {repr(response[:80])}")
