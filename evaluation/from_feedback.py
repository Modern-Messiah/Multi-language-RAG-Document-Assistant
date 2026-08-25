"""Turn collected ratings into work for the golden set.

The golden cases in golden.py are questions I invented. They guard the
properties worth protecting, but they measure my guesses about what people ask.
A thumbs-down is the opposite: a real question that got a real bad answer, with
the documents that produced it and the request id that finds it in the log.

    python -m evaluation.from_feedback
    python -m evaluation.from_feedback --input data/feedback/feedback.jsonl --up

What it does not do is fill in the expectation. Which document *should* have
answered a question is a judgement about the corpus, and a stub that guessed it
would turn one bad answer into a permanently wrong benchmark. Each case is
printed with `expected_sources=[]` for a person to complete.

Stdlib only, like run_eval, so it runs against a production volume without
installing anything.
"""
import argparse
import sys
from pathlib import Path

from app.feedback import (
    FEEDBACK_FILENAME,
    golden_case_stub,
    read_records,
    summarise,
    unanswered_questions,
)

DEFAULT_INPUT = Path("data/feedback") / FEEDBACK_FILENAME


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default=str(DEFAULT_INPUT),
        help=f"the ratings file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--up", action="store_true",
        help="list the answers people liked instead of the ones they did not",
    )
    parser.add_argument(
        "--limit", type=int, default=20,
        help="how many cases to print (default: 20)",
    )
    args = parser.parse_args(argv)

    path = Path(args.input)
    if not path.exists():
        print(f"No ratings file at {path}.", file=sys.stderr)
        print(
            "Nothing has been rated yet, or FEEDBACK_DIR points elsewhere.",
            file=sys.stderr,
        )
        return 1

    records = read_records(path)

    if not records:
        print(f"{path} holds no readable records.")
        return 1

    counts = summarise(records)
    print(f"{counts['total']} ratings: {counts['up']} up, {counts['down']} down", end="")
    print(f" ({counts['down_rate']:.0%} negative)" if counts["down_rate"] is not None else "")

    if counts["down_by_source"]:
        print("\nDocuments behind the negative ratings:")
        for source, count in counts["down_by_source"].items():
            print(f"  {count:4}  {source}")

    wanted = "up" if args.up else "down"
    cases = unanswered_questions(records, rating=wanted)
    if not cases:
        print(f"\nNothing rated {wanted}.")
        return 0

    shown = cases[: args.limit]
    print(f"\n{len(shown)} of {len(cases)} cases rated {wanted}, newest first:\n")
    for record in shown:
        print(f"# {record.get('at')}  {record.get('client') or '?'}"
              f"  user={record.get('user_id')}")
        if record.get("comment"):
            print(f"# comment: {record['comment']}")
        print(golden_case_stub(record))
        print()

    if len(cases) > len(shown):
        # Said out loud rather than silently truncating: a list that stops at 20
        # reads as "that is all of them".
        print(f"({len(cases) - len(shown)} more not shown; raise --limit)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
