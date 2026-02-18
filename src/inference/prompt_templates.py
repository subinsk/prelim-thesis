def create_cot_prompt(question: str, doc1: str, doc2: str) -> str:
    """Create Chain-of-Thought prompt for multi-hop QA."""

    prompt = f"""Answer the following question using the provided documents. Think step by step.

Document 1:
{doc1}

Document 2:
{doc2}

Question: {question}

Let's solve this step by step:
1. First, I'll identify relevant information from Document 1.
2. Then, I'll use that to find the answer in Document 2.
3. Finally, I'll provide the answer.

Step-by-step reasoning:"""

    return prompt


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


def extract_answer(response: str) -> str:
    """Extract final answer from model response."""

    response_lower = response.lower()

    # Check for "the answer is X" pattern
    if "the answer is" in response_lower:
        idx = response_lower.find("the answer is")
        answer_part = response[idx + 13:].strip()
        for delimiter in ['.', '\n', ',']:
            if delimiter in answer_part:
                answer_part = answer_part.split(delimiter)[0]
        return answer_part.strip()

    # Check for "Answer: X" pattern
    if "answer:" in response_lower:
        idx = response_lower.find("answer:")
        answer_part = response[idx + 7:].strip()
        for delimiter in ['.', '\n', ',']:
            if delimiter in answer_part:
                answer_part = answer_part.split(delimiter)[0]
        return answer_part.strip()

    # Default: return last line
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    if lines:
        return lines[-1]

    return response.strip()
