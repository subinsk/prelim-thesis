"""
Dry-run test: validates the entire pipeline WITHOUT and WITH API calls.

Phase 1 (no API): Tests data loading, conflict injection, answer extraction,
    and evaluation logic on real dataset examples. Verifies that:
    - Intermediate answers are used for injection (not final answers)
    - Injection actually modifies the document (non-zero replacement count)
    - Document ordering is correct (bridge doc vs answer doc)
    - Answer extraction handles all formats correctly
    - Evaluation is exact match (not substring)
    - CFR and POR are mutually exclusive

Phase 2 (with API): Runs 1-2 examples through a real model to verify
    the full end-to-end pipeline produces sensible results.

Usage:
    python experiments/dry_run_test.py              # Phase 1 only (no API)
    python experiments/dry_run_test.py --with-api   # Phase 1 + Phase 2 (1-2 API calls)
"""

import json
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.hotpotqa_loader import HotpotQALoader
from src.data.musique_loader import MuSiQueLoader
from src.data.conflict_injector import ConflictInjector
from src.inference.prompt_templates import create_cot_prompt, create_3hop_cot_prompt, extract_answer
from src.evaluation.metrics import check_answer, normalize_answer


def separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def test_answer_extraction():
    """Test that answer extraction handles all formats correctly."""
    separator("TEST: Answer Extraction")

    tests = [
        ("Step 1: ...\nStep 2: ...\nThe answer is 1970.", "1970"),
        ("The answer is Denver.", "Denver"),
        ("The answer is 9,984,670 square km.", "9,984,670 square km"),
        ("The answer is Greenwich Village, New York City.", "Greenwich Village, New York City"),
        ("Reasoning... Answer: New York City", "New York City"),
        ("The answer is from 1986 to 2013.", "from 1986 to 2013"),
        ("<think>lots of thinking</think>\nThe answer is London.", "London"),
        ("The answer is 1970. This is confirmed by the document.", "1970"),
        ("Some text\nDenver", "Denver"),
        ("The answer is: Chief of Protocol.", "Chief of Protocol"),
    ]

    passed = 0
    for response, expected in tests:
        got = extract_answer(response)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] got='{got}' expected='{expected}'")
        if not ok:
            print(f"         input: {repr(response[:80])}")

    print(f"\n  Result: {passed}/{len(tests)} passed")
    return passed == len(tests)


def test_evaluation_logic():
    """Test that check_answer uses exact match and mutual exclusive CFR/POR."""
    separator("TEST: Evaluation Logic (Exact Match + Mutual Exclusive)")

    tests = [
        # (pred, gold, fake, exp_correct, exp_cfr, exp_por)
        ("1970", "1970", "1955", True, False, True),
        ("1955", "1970", "1955", False, True, False),
        ("something else", "1970", "1955", False, False, False),
        # Substring should NOT match (old bug)
        ("born in 1970 in London", "1970", "1955", False, False, False),
        ("John Smith went to the store", "John", "Jane", False, False, False),
        # Exact match after normalization
        ("the Chief of Protocol", "Chief of Protocol", "John Smith", True, False, True),
        ("9,984", "9,984", "5,432", True, False, True),
        ("chief of protocol", "Chief of Protocol", None, True, False, False),
    ]

    passed = 0
    for pred, gold, fake, exp_correct, exp_cfr, exp_por in tests:
        result = check_answer(pred, gold, fake)
        ok = (result['correct'] == exp_correct and
              result['followed_context'] == exp_cfr and
              result.get('used_parametric', False) == exp_por)
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] pred='{pred}' gold='{gold}' fake='{fake}'")
        if not ok:
            print(f"         got: correct={result['correct']} cfr={result['followed_context']} por={result.get('used_parametric')}")
            print(f"         exp: correct={exp_correct} cfr={exp_cfr} por={exp_por}")

    # Verify CFR + POR <= 1 always
    print(f"\n  Verifying CFR + POR mutual exclusivity...")
    for pred, gold, fake, _, _, _ in tests:
        result = check_answer(pred, gold, fake)
        cfr_val = 1 if result['followed_context'] else 0
        por_val = 1 if result.get('used_parametric', False) else 0
        assert cfr_val + por_val <= 1, f"CFR+POR > 1 for pred='{pred}'"
    print(f"  All mutual exclusivity checks passed.")

    print(f"\n  Result: {passed}/{len(tests)} passed")
    return passed == len(tests)


def test_3hop_injection():
    """Test that 3-hop injection uses intermediate answers at each hop."""
    separator("TEST: 3-Hop Conflict Injection (Intermediate Answers)")

    loader = MuSiQueLoader()
    loader.load('data/musique/validation.jsonl')
    examples = loader.get_questions_by_hops(3, n=5)

    injector = ConflictInjector(seed=42)
    all_ok = True

    for i, ex in enumerate(examples[:3]):
        question, docs, answer, step_answers = loader.extract_supporting_docs(ex)
        print(f"\n  --- Example {i+1} ---")
        print(f"  Q: {question}")
        print(f"  Final answer: {answer}")
        print(f"  Step answers: {step_answers}")

        if len(docs) < 3 or len(step_answers) < 3:
            print(f"  SKIP: not enough docs/answers")
            continue

        for hop in [1, 2, 3]:
            target = step_answers[hop - 1]
            doc = docs[hop - 1]

            # Verify target appears in doc
            in_doc = target.lower() in doc.lower()
            print(f"\n  Hop {hop}: target='{target}' in_doc={in_doc}")

            if not in_doc:
                print(f"  FAIL: target entity not found in hop {hop} document!")
                all_ok = False
                continue

            mod_doc, fake, succeeded = injector.inject_conflict(
                doc=doc, target_entity=target, question=question, hop=hop
            )
            print(f"    Fake: '{fake}' | Succeeded: {succeeded}")
            print(f"    Doc before (first 100): {doc[:100]}...")
            print(f"    Doc after  (first 100): {mod_doc[:100]}...")

            if not succeeded:
                print(f"    FAIL: injection did not succeed!")
                all_ok = False
            elif mod_doc == doc:
                print(f"    FAIL: document was not modified!")
                all_ok = False
            else:
                # Verify original entity is gone and fake is present
                if target.lower() in mod_doc.lower():
                    print(f"    WARN: original entity still in doc (partial replacement?)")
                if fake.lower() in mod_doc.lower():
                    print(f"    OK: fake entity found in modified doc")
                else:
                    print(f"    FAIL: fake entity NOT in modified doc!")
                    all_ok = False

    print(f"\n  Result: {'ALL PASSED' if all_ok else 'SOME FAILURES'}")
    return all_ok


def test_2hop_doc_ordering():
    """Test that 2-hop doc ordering correctly identifies bridge and answer docs."""
    separator("TEST: 2-Hop Document Ordering")

    loader = HotpotQALoader()
    loader.load()
    examples = loader.get_bridge_questions(100)

    correct_order = 0
    has_bridge_entity = 0
    total = 0

    for ex in examples[:50]:
        q, bridge_doc, answer_doc, answer, bridge_entity = loader.extract_supporting_facts(ex)
        if not bridge_doc or not answer_doc:
            continue
        total += 1

        # Verify answer is in answer_doc
        a_in_answer_doc = answer.lower() in answer_doc.lower()
        if a_in_answer_doc:
            correct_order += 1

        if bridge_entity:
            has_bridge_entity += 1

    print(f"  Total tested: {total}")
    print(f"  Answer in answer_doc: {correct_order}/{total} ({correct_order/total*100:.1f}%)")
    print(f"  Bridge entity found: {has_bridge_entity}/{total} ({has_bridge_entity/total*100:.1f}%)")

    # Also test injection at both hops
    injector = ConflictInjector(seed=42)
    hop1_ok = 0
    hop2_ok = 0

    for ex in examples[:20]:
        q, bridge_doc, answer_doc, answer, bridge_entity = loader.extract_supporting_facts(ex)
        if not bridge_doc or not answer_doc:
            continue

        # Hop 2: replace answer in answer_doc (should always work)
        _, _, ok2 = injector.inject_conflict(answer_doc, answer, q, hop=2)
        if ok2:
            hop2_ok += 1

        # Hop 1: replace bridge entity in bridge_doc
        if bridge_entity:
            _, _, ok1 = injector.inject_conflict(bridge_doc, bridge_entity, q, hop=1)
            if ok1:
                hop1_ok += 1

    print(f"\n  Injection success (first 20 examples):")
    print(f"    Hop 1 (bridge entity): {hop1_ok}/20")
    print(f"    Hop 2 (final answer): {hop2_ok}/20")

    ok = correct_order >= total * 0.95 and hop2_ok >= 18
    print(f"\n  Result: {'PASSED' if ok else 'NEEDS REVIEW'}")
    return ok


def test_2hop_with_api():
    """Run 1-2 examples through a real model (requires API key)."""
    separator("TEST: 2-Hop End-to-End with API (1 example)")

    loader = HotpotQALoader()
    loader.load()
    examples = loader.get_bridge_questions(10)
    injector = ConflictInjector(seed=42)

    # Use Gemini (cheapest)
    from src.inference.gemini_client import GeminiClient
    client = GeminiClient(model="gemini-2.5-flash-lite")

    ex = examples[0]
    q, bridge_doc, answer_doc, answer, bridge_entity = loader.extract_supporting_facts(ex)

    print(f"  Q: {q}")
    print(f"  Answer: {answer}")
    print(f"  Bridge entity: {bridge_entity}")

    # Baseline
    prompt = create_cot_prompt(q, bridge_doc, answer_doc)
    response = client.generate(prompt, max_tokens=1024)
    pred = extract_answer(response)
    result = check_answer(pred, answer)
    print(f"\n  BASELINE:")
    print(f"    Response (first 200): {response[:200]}...")
    print(f"    Extracted: '{pred}'")
    print(f"    Correct: {result['correct']}")

    # Hop 2 conflict
    mod_doc, fake, ok = injector.inject_conflict(answer_doc, answer, q, hop=2)
    if ok:
        prompt = create_cot_prompt(q, bridge_doc, mod_doc)
        response = client.generate(prompt, max_tokens=1024)
        pred = extract_answer(response)
        result = check_answer(pred, answer, fake)
        print(f"\n  HOP 2 CONFLICT (replaced '{answer}' with '{fake}'):")
        print(f"    Response (first 200): {response[:200]}...")
        print(f"    Extracted: '{pred}'")
        print(f"    Correct (POR): {result['correct']}")
        print(f"    Followed context (CFR): {result['followed_context']}")
    else:
        print(f"\n  HOP 2: injection failed!")

    return True


def test_3hop_with_api():
    """Run 1 example through a real model for 3-hop."""
    separator("TEST: 3-Hop End-to-End with API (1 example)")

    loader = MuSiQueLoader()
    loader.load('data/musique/validation.jsonl')
    examples = loader.get_questions_by_hops(3, n=5)
    injector = ConflictInjector(seed=42)

    from src.inference.gemini_client import GeminiClient
    client = GeminiClient(model="gemini-2.5-flash-lite")

    ex = examples[0]
    question, docs, answer, step_answers = loader.extract_supporting_docs(ex)

    print(f"  Q: {question}")
    print(f"  Answer: {answer}")
    print(f"  Step answers: {step_answers}")

    # Baseline
    prompt = create_3hop_cot_prompt(question, docs[0], docs[1], docs[2])
    response = client.generate(prompt, max_tokens=1024)
    pred = extract_answer(response)
    result = check_answer(pred, answer)
    print(f"\n  BASELINE:")
    print(f"    Extracted: '{pred}' | Correct: {result['correct']}")

    # Conflict at each hop
    for hop in [1, 2, 3]:
        target = step_answers[hop - 1]
        mod_doc, fake, ok = injector.inject_conflict(docs[hop-1], target, question, hop=hop)
        if not ok:
            print(f"\n  HOP {hop}: injection failed for target='{target}'")
            continue

        modified = list(docs)
        modified[hop-1] = mod_doc
        prompt = create_3hop_cot_prompt(question, modified[0], modified[1], modified[2])
        response = client.generate(prompt, max_tokens=1024)
        pred = extract_answer(response)
        result = check_answer(pred, answer, fake)
        print(f"\n  HOP {hop} CONFLICT ('{target}' -> '{fake}'):")
        print(f"    Extracted: '{pred}'")
        print(f"    Correct (POR): {result['correct']} | CFR: {result['followed_context']}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Dry-run pipeline validation")
    parser.add_argument('--with-api', action='store_true',
                        help="Include API-based tests (costs ~2-4 API calls)")
    args = parser.parse_args()

    print("=" * 70)
    print("  PIPELINE VALIDATION DRY RUN")
    print("=" * 70)

    results = {}

    # Phase 1: No API tests
    results['answer_extraction'] = test_answer_extraction()
    results['evaluation_logic'] = test_evaluation_logic()
    results['3hop_injection'] = test_3hop_injection()
    results['2hop_doc_ordering'] = test_2hop_doc_ordering()

    # Phase 2: API tests
    if args.with_api:
        results['2hop_api'] = test_2hop_with_api()
        results['3hop_api'] = test_3hop_with_api()

    # Summary
    separator("SUMMARY")
    all_pass = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {test_name}")

    print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME FAILURES -- FIX BEFORE RUNNING EXPERIMENTS'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
