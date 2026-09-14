"""Upload downloaded benchmarks into Langfuse Datasets.

Usage:
    python scripts/upload_to_langfuse.py --dataset sberquad --count 15
    python scripts/upload_to_langfuse.py --dataset financebench --count 15
    python scripts/upload_to_langfuse.py --dataset all --count 10
"""
import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("upload_to_langfuse")

BASE_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmarks"


def upload_financebench(client, count=15):
    filepath = BASE_DIR / "financebench" / "financebench.jsonl"
    if not filepath.exists():
        logger.error("FinanceBench file not found at %s", filepath)
        return

    ds_name = "eval-financebench"
    try:
        client.get_dataset(ds_name)
    except Exception:
        client.create_dataset(
            name=ds_name,
            description="FinanceBench: Corporate 10-K financial reports evaluation",
        )

    logger.info("Uploading up to %d items to '%s'...", count, ds_name)
    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= count:
                break
            item = json.loads(line)
            client.create_dataset_item(
                dataset_name=ds_name,
                input={"question": item.get("question", "")},
                expected_output={
                    "answer": item.get("answer", ""),
                    "evidence": item.get("evidence", []),
                },
                metadata={
                    "company": item.get("company", ""),
                    "doc_name": item.get("doc_name", ""),
                    "question_type": item.get("question_type", ""),
                },
            )
    client.flush()
    logger.info("Successfully uploaded FinanceBench items to Langfuse!")


def upload_sberquad(client, count=15):
    filepath = BASE_DIR / "sberquad" / "sberquad_val.jsonl"
    if not filepath.exists():
        logger.error("SberQuAD file not found at %s", filepath)
        return

    ds_name = "eval-sberquad-ru"
    try:
        client.get_dataset(ds_name)
    except Exception:
        client.create_dataset(
            name=ds_name,
            description="SberQuAD: Russian Question Answering benchmark by Sber",
        )

    logger.info("Uploading up to %d items to '%s'...", count, ds_name)
    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= count:
                break
            item = json.loads(line)
            client.create_dataset_item(
                dataset_name=ds_name,
                input={"question": item.get("question", "")},
                expected_output={"answer": item.get("answer", "")},
                metadata={
                    "title": item.get("title", ""),
                    "context": item.get("context", "")[:1000],
                    "language": "ru",
                },
            )
    client.flush()
    logger.info("Successfully uploaded SberQuAD items to Langfuse!")


def upload_hotpotqa(client, count=15):
    filepath = BASE_DIR / "hotpotqa" / "hotpotqa_sample.jsonl"
    if not filepath.exists():
        logger.error("HotpotQA file not found at %s", filepath)
        return

    ds_name = "eval-hotpotqa-multihop"
    try:
        client.get_dataset(ds_name)
    except Exception:
        client.create_dataset(
            name=ds_name,
            description="HotpotQA: Multi-hop reasoning across multiple documents",
        )

    logger.info("Uploading up to %d items to '%s'...", count, ds_name)
    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= count:
                break
            item = json.loads(line)
            client.create_dataset_item(
                dataset_name=ds_name,
                input={"question": item.get("question", "")},
                expected_output={"answer": item.get("answer", "")},
                metadata={
                    "type": item.get("type", ""),
                    "level": item.get("level", ""),
                },
            )
    client.flush()
    logger.info("Successfully uploaded HotpotQA items to Langfuse!")


def upload_beir_scifact(client, count=15):
    queries_file = BASE_DIR / "beir_scifact" / "queries.jsonl"
    if not queries_file.exists():
        logger.error("BEIR SciFact queries not found at %s", queries_file)
        return

    ds_name = "eval-beir-scifact"
    try:
        client.get_dataset(ds_name)
    except Exception:
        client.create_dataset(
            name=ds_name,
            description="BEIR SciFact: Scientific claims and evidence retrieval",
        )

    logger.info("Uploading up to %d items to '%s'...", count, ds_name)
    with open(queries_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= count:
                break
            item = json.loads(line)
            client.create_dataset_item(
                dataset_name=ds_name,
                input={"question": item.get("text", "")},
                expected_output={"query_id": item.get("_id", "")},
                metadata={"domain": "scientific"},
            )
    client.flush()
    logger.info("Successfully uploaded BEIR SciFact items to Langfuse!")


def main():
    load_dotenv()
    from langfuse import Langfuse

    parser = argparse.ArgumentParser(description="Upload benchmarks to Langfuse Datasets")
    parser.add_argument(
        "--dataset",
        choices=["financebench", "sberquad", "hotpotqa", "scifact", "all"],
        default="all",
        help="Dataset to upload",
    )
    parser.add_argument("--count", type=int, default=15, help="Number of items to upload per dataset")
    args = parser.parse_args()

    client = Langfuse()
    logger.info("Connected to Langfuse at %s", client.base_url)

    if args.dataset in ("financebench", "all"):
        upload_financebench(client, args.count)
    if args.dataset in ("sberquad", "all"):
        upload_sberquad(client, args.count)
    if args.dataset in ("hotpotqa", "all"):
        upload_hotpotqa(client, args.count)
    if args.dataset in ("scifact", "all"):
        upload_beir_scifact(client, args.count)

    print("\n🎉 Upload to Langfuse completed!")
    print("Refresh the Langfuse Datasets page (http://localhost:3000) to explore your new datasets.")


if __name__ == "__main__":
    main()
