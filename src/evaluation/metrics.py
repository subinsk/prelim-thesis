def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    return answer.lower().strip().rstrip('.').rstrip(',')


def check_answer(predicted: str, gold: str, fake: str = None) -> dict:
    """
    Check if predicted answer matches gold or fake.

    Returns:
        dict with 'correct', 'followed_context', 'used_parametric'
    """
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)

    result = {
        'predicted': predicted,
        'gold': gold,
        'correct': gold_norm in pred_norm or pred_norm in gold_norm,
        'followed_context': False,
        'used_parametric': False
    }

    if fake:
        fake_norm = normalize_answer(fake)
        result['fake'] = fake
        result['followed_context'] = fake_norm in pred_norm or pred_norm in fake_norm
        result['used_parametric'] = result['correct']

    return result
