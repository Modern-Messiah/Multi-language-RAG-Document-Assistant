"""Upload downloaded benchmarks into Langfuse Datasets with multithreading.

Supports uploading partial slices or ALL items (thousands of questions) rapidly.
Deduplication is ensured by using deterministic unique IDs for each item.

Usage:
    python scripts/upload_to_langfuse.py --dataset sberquad --all
    python scripts/upload_to_langfuse.py --dataset financebench --all
    python scripts/upload_to_langfuse.py --dataset all --all
"""
import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import pyarrow.parquet as pq
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("upload_to_langfuse")

BASE_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmarks"
WORKERS = 16


def _batch_upload_items(client, dataset_name: str, items: List[Dict[str, Any]], desc: str = ""):
    """Upload items in parallel using ThreadPoolExecutor."""
    total = len(items)
    logger.info("Uploading %d items to '%s' (%s) with %d workers...", total, dataset_name, desc, WORKERS)

    def _send(item):
        try:
            client.create_dataset_item(
                dataset_name=dataset_name,
                id=item.get("id"),
                input=item.get("input"),
                expected_output=item.get("expected_output"),
                metadata=item.get("metadata"),
            )
            return True
        except Exception as exc:
            logger.debug("Item error: %s", exc)
            return False

    success = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(_send, item) for item in items]
        for idx, fut in enumerate(as_completed(futures), start=1):
            if fut.result():
                success += 1
            if idx % 500 == 0 or idx == total:
                logger.info("  [%s] Progress: %d / %d items processed", dataset_name, idx, total)

    client.flush()
    logger.info("✓ Finished '%s': %d / %d items uploaded successfully.", dataset_name, success, total)


def upload_financebench(client, max_count: int = None):
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
            description="FinanceBench: Corporate 10-K financial reports evaluation (Apple, Microsoft, 3M, etc.)",
        )

    items = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            fb_id = data.get("financebench_id", "")
            items.append({
                "id": f"fb-{fb_id}" if fb_id else None,
                "input": {"question": data.get("question", "")},
                "expected_output": {
                    "answer": data.get("answer", ""),
                    "evidence": data.get("evidence", []),
                },
                "metadata": {
                    "company": data.get("company", ""),
                    "doc_name": data.get("doc_name", ""),
                    "question_type": data.get("question_type", ""),
                },
            })
            if max_count and len(items) >= max_count:
                break

    _batch_upload_items(client, ds_name, items, "FinanceBench")


def upload_sberquad(client, max_count: int = None):
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

    items = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            item_id = data.get("id", "")
            items.append({
                "id": f"sber-{item_id}" if item_id else None,
                "input": {"question": data.get("question", "")},
                "expected_output": {"answer": data.get("answer", "")},
                "metadata": {
                    "title": data.get("title", ""),
                    "context": data.get("context", "")[:1000],
                    "language": "ru",
                },
            })
            if max_count and len(items) >= max_count:
                break

    _batch_upload_items(client, ds_name, items, "SberQuAD (Russian)")


def upload_hotpotqa(client, max_count: int = None):
    ds_name = "eval-hotpotqa-multihop"
    try:
        client.get_dataset(ds_name)
    except Exception:
        client.create_dataset(
            name=ds_name,
            description="HotpotQA: Multi-hop reasoning across multiple documents",
        )

    items = []
    # Try downloading full validation parquet if available
    try:
        pq_path = hf_hub_download(
            repo_id="hotpotqa/hotpot_qa",
            filename="distractor/validation-00000-of-00001.parquet",
            repo_type="dataset",
        )
        table = pq.read_table(pq_path)
        records = table.to_pylist()
        for r in records:
            items.append({
                "id": f"hotpot-{r['id']}",
                "input": {"question": r.get("question", "")},
                "expected_output": {"answer": r.get("answer", "")},
                "metadata": {
                    "type": r.get("type", ""),
                    "level": r.get("level", ""),
                },
            })
            if max_count and len(items) >= max_count:
                break
    except Exception as err:
        logger.warning("Falling back to local hotpot sample: %s", err)
        filepath = BASE_DIR / "hotpotqa" / "hotpotqa_sample.jsonl"
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                items.append({
                    "id": f"hotpot-{r['id']}",
                    "input": {"question": r.get("question", "")},
                    "expected_output": {"answer": r.get("answer", "")},
                    "metadata": {"type": r.get("type", ""), "level": r.get("level", "")},
                })
                if max_count and len(items) >= max_count:
                    break

    _batch_upload_items(client, ds_name, items, "HotpotQA")


def upload_beir_scifact(client, max_count: int = None):
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

    items = []
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            q_id = data.get("_id", "")
            items.append({
                "id": f"scifact-{q_id}" if q_id else None,
                "input": {"question": data.get("text", "")},
                "expected_output": {"query_id": q_id},
                "metadata": {"domain": "scientific"},
            })
            if max_count and len(items) >= max_count:
                break

    _batch_upload_items(client, ds_name, items, "BEIR SciFact")


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
    parser.add_argument("--count", type=int, default=None, help="Max number of items to upload per dataset")
    parser.add_argument("--all", action="store_true", help="Upload all items without limit")
    args = parser.parse_args()

    max_count = None if args.all else (args.count or 100)

    client = Langfuse()
    logger.info("Connected to Langfuse at %s", client.base_url)

    if args.dataset in ("financebench", "all"):
        upload_financebench(client, max_count)
    if args.dataset in ("sberquad", "all"):
        upload_sberquad(client, max_count)
    if args.dataset in ("scifact", "all"):
        upload_beir_scifact(client, max_count)
    if args.dataset in ("hotpotqa", "all"):
        upload_hotpotqa(client, max_count)

    print("\n🎉 Upload to Langfuse completed!")
    print("Refresh the Langfuse Datasets page (http://localhost:3000) to see all uploaded questions.")


if __name__ == "__main__":
    main()
