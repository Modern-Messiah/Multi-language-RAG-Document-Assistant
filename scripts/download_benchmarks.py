"""Download lightweight benchmarks for RAG testing.

Datasets included:
1. FinanceBench (real company financial reports and questions)
2. SberQuAD (Russian QA standard by Sber)
3. HotpotQA (multi-hop reasoning)
4. BEIR SciFact (scientific document retrieval)
5. TyDi QA (multilingual QA by Google, focused on Russian & English)
6. RAGTruth (hallucination detection in RAG)

All datasets together are under 25 MB.
"""
import io
import json
import logging
import urllib.request
import zipfile
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_benchmarks")

BASE_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmarks"


def download_financebench():
    """Download FinanceBench (150 QA pairs with evidence)."""
    out_dir = BASE_DIR / "financebench"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "financebench.jsonl"
    if out_file.exists() and out_file.stat().st_size > 1000:
        logger.info("FinanceBench already exists at %s", out_file)
        return out_file

    logger.info("Downloading FinanceBench...")
    url = "https://huggingface.co/datasets/PatronusAI/financebench/raw/main/financebench_merged.jsonl"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read()
    with open(out_file, "wb") as f:
        f.write(content)
    logger.info("FinanceBench saved (%d KB)", len(content) // 1024)
    return out_file


def download_sberquad():
    """Download SberQuAD validation dataset (Russian QA)."""
    out_dir = BASE_DIR / "sberquad"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "sberquad_val.jsonl"
    if out_file.exists() and out_file.stat().st_size > 1000:
        logger.info("SberQuAD already exists at %s", out_file)
        return out_file

    logger.info("Downloading SberQuAD validation parquet via HF hub...")
    pq_path = hf_hub_download(
        repo_id="sberquad",
        filename="sberquad/validation-00000-of-00001.parquet",
        repo_type="dataset",
    )
    table = pq.read_table(pq_path)
    records = table.to_pylist()

    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            ans_text = r["answers"]["text"][0] if r.get("answers") and r["answers"].get("text") else ""
            f.write(json.dumps({
                "id": r["id"],
                "title": r.get("title", ""),
                "context": r["context"],
                "question": r["question"],
                "answer": ans_text,
            }, ensure_ascii=False) + "\n")

    logger.info("SberQuAD saved (%d items, %d KB)", len(records), out_file.stat().st_size // 1024)
    return out_file


def download_hotpotqa(max_samples=200):
    """Download HotpotQA validation split (multi-hop QA)."""
    out_dir = BASE_DIR / "hotpotqa"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "hotpotqa_sample.jsonl"
    if out_file.exists() and out_file.stat().st_size > 1000:
        logger.info("HotpotQA already exists at %s", out_file)
        return out_file

    logger.info("Downloading HotpotQA validation parquet via HF hub...")
    pq_path = hf_hub_download(
        repo_id="hotpotqa/hotpot_qa",
        filename="distractor/validation-00000-of-00001.parquet",
        repo_type="dataset",
    )
    table = pq.read_table(pq_path)
    sample_table = table.slice(0, max_samples)
    records = sample_table.to_pylist()

    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "id": r["id"],
                "question": r["question"],
                "answer": r["answer"],
                "type": r.get("type", ""),
                "level": r.get("level", ""),
                "supporting_facts": r.get("supporting_facts", {}),
            }, ensure_ascii=False) + "\n")

    logger.info("HotpotQA saved (%d items, %d KB)", len(records), out_file.stat().st_size // 1024)
    return out_file


def download_beir_scifact():
    """Download BEIR SciFact dataset (scientific retrieval)."""
    out_dir = BASE_DIR / "beir_scifact"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "corpus.jsonl"
    if out_file.exists() and out_file.stat().st_size > 1000:
        logger.info("BEIR SciFact already exists at %s", out_dir)
        return out_dir

    logger.info("Downloading BEIR SciFact zip (~2.7 MB)...")
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        content = resp.read()

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        zf.extractall(BASE_DIR)

    # Rename extracted folder if needed
    extracted = BASE_DIR / "scifact"
    if extracted.exists() and extracted != out_dir:
        for item in extracted.glob("*"):
            item.rename(out_dir / item.name)
        extracted.rmdir()

    logger.info("BEIR SciFact extracted to %s", out_dir)
    return out_dir


def download_tydiqa(max_samples=200):
    """Download TyDi QA (multilingual QA, filtering Russian and English)."""
    out_dir = BASE_DIR / "tydiqa"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "tydiqa_sample.jsonl"
    if out_file.exists() and out_file.stat().st_size > 1000:
        logger.info("TyDi QA already exists at %s", out_file)
        return out_file

    logger.info("Downloading TyDi QA validation parquet via HF hub...")
    pq_path = hf_hub_download(
        repo_id="google-research-datasets/tydiqa",
        filename="primary_task/validation-00000-of-00001.parquet",
        repo_type="dataset",
    )
    table = pq.read_table(pq_path)
    records = table.to_pylist()

    # Filter Russian & English
    selected = [
        r for r in records
        if any(c in r.get("question_text", "") for c in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
        or r.get("document_url", "").startswith("http://ru.")
        or r.get("document_url", "").startswith("http://en.")
    ][:max_samples]

    if not selected:
        selected = records[:max_samples]

    with open(out_file, "w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps({
                "document_title": r.get("document_title", ""),
                "question": r.get("question_text", ""),
                "document_plaintext": r.get("document_plaintext", "")[:2000],
            }, ensure_ascii=False) + "\n")

    logger.info("TyDi QA saved (%d items, %d KB)", len(selected), out_file.stat().st_size // 1024)
    return out_file


def download_ragtruth():
    """Download RAGTruth QA questions from GitHub."""
    out_dir = BASE_DIR / "ragtruth"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "ragtruth_qa.jsonl"
    if out_file.exists() and out_file.stat().st_size > 1000:
        logger.info("RAGTruth already exists at %s", out_file)
        return out_file

    logger.info("Downloading RAGTruth dataset from GitHub...")
    url = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/response.jsonl"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read()

    with open(out_file, "wb") as f:
        f.write(content)

    logger.info("RAGTruth saved (%d KB)", len(content) // 1024)
    return out_file


def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n📂 Saving benchmarks to: {BASE_DIR}")

    download_financebench()
    download_sberquad()
    download_hotpotqa()
    download_beir_scifact()
    download_tydiqa()
    download_ragtruth()

    print("\n✅ All 6 benchmarks downloaded successfully!")
    for p in sorted(BASE_DIR.glob("**/*")):
        if p.is_file():
            print(f"  - {p.relative_to(BASE_DIR)}: {p.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
