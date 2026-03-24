import re
import string


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison.

    Lowercases, strips whitespace, removes articles (a, an, the),
    removes ALL punctuation (including commas in numbers), and collapses
    multiple spaces. This ensures "9,984,670" matches "9984670".
    Follows the standard HotpotQA/SQuAD normalization approach.
    """
    answer = answer.lower().strip()
    # Remove articles
    answer = re.sub(r'\b(a|an|the)\b', ' ', answer)
    # Strip all punctuation (delete, don't replace with space)
    answer = re.sub(r'[^\w\s]', '', answer)
    # Collapse whitespace
    answer = re.sub(r'\s+', ' ', answer).strip()
    return answer


def check_answer(predicted: str, gold: str, fake: str = None) -> dict:
    """
    Check if predicted answer matches gold or fake.

    Uses exact match after normalization. CFR and POR are mutually exclusive:
    - correct (used_parametric): normalized prediction == normalized gold
    - followed_context: normalized prediction == normalized fake AND not correct
    - other: neither matches (model confused or hallucinated)

    Returns:
        dict with 'predicted', 'gold', 'correct', 'followed_context', 'used_parametric'
    """
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)

    is_correct = pred_norm == gold_norm

    result = {
        'predicted': predicted,
        'gold': gold,
        'correct': is_correct,
        'followed_context': False,
        'used_parametric': False,
    }

    if fake:
        fake_norm = normalize_answer(fake)
        result['fake'] = fake
        result['used_parametric'] = is_correct
        # Only count as following context if NOT correct (mutually exclusive)
        result['followed_context'] = (not is_correct) and (pred_norm == fake_norm)

    return result


if __name__ == "__main__":
    # Test cases
    tests = [
        # (predicted, gold, fake, expected_correct, expected_cfr)
        ("1970", "1970", "1955", True, False),
        ("1955", "1970", "1955", False, True),
        ("something else", "1970", "1955", False, False),
        ("The answer is 1970.", "1970", "1955", False, False),  # not exact match
        ("Chief of Protocol", "Chief of Protocol", None, True, False),
        ("chief of protocol", "Chief of Protocol", None, True, False),
        ("the Chief of Protocol", "Chief of Protocol", None, True, False),  # article removed
        ("9,984", "9,984", "5,432", True, False),
        ("Denver", "Denver", "London", True, False),
        ("London", "Denver", "London", False, True),
    ]

    print("Testing check_answer:")
    for pred, gold, fake, exp_correct, exp_cfr in tests:
        result = check_answer(pred, gold, fake)
        ok = result['correct'] == exp_correct and result['followed_context'] == exp_cfr
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] pred='{pred}' gold='{gold}' fake='{fake}' -> "
              f"correct={result['correct']} cfr={result['followed_context']}")

    print("\nTesting normalize_answer:")
    norm_tests = [
        ("The United Kingdom", "united kingdom"),
        ("9,984,670", "9984670"),
        ("9984670", "9984670"),
        ("Chief of Protocol.", "chief of protocol"),
        ("  Denver  ", "denver"),
        ("a cat", "cat"),
        ("Greenwich Village, New York City", "greenwich village new york city"),
    ]
    for inp, expected in norm_tests:
        got = normalize_answer(inp)
        status = "PASS" if got == expected else "FAIL"
        print(f"  [{status}] '{inp}' -> '{got}' (expected '{expected}')")
