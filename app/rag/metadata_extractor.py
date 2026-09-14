"""
Metadata Extractor for Document Ingestion and Query Filtering.

Extracts structured metadata (company, year, quarter, document type)
from document filenames and contents during ingestion, and detects
relevant filter criteria from search queries for scoped vector retrieval.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Regular expressions for metadata extraction
YEAR_PATTERN = re.compile(r"\b(19\d\d|20[0-3]\d)\b")
FY_YEAR_PATTERN = re.compile(r"\bFY\s*(?:20)?(\d{2,4})\b", re.IGNORECASE)
QUARTER_PATTERN = re.compile(r"\b(Q[1-4]|1Q|2Q|3Q|4Q)\b", re.IGNORECASE)
DOC_TYPE_PATTERN = re.compile(r"\b(10-?K|10-?Q|8-?K|EARNINGS|ANNUAL\s+REPORT)\b", re.IGNORECASE)

# Common known company names and tickers in financial & general domains
KNOWN_COMPANIES = {
    "WALMART",
    "VERIZON",
    "PFIZER",
    "ULTABEAUTY",
    "ULTA",
    "AMCOR",
    "ADOBE",
    "3M",
    "APPLE",
    "MICROSOFT",
    "AMAZON",
    "GOOGLE",
    "TESLA",
    "META",
    "NETFLIX",
    "NVIDIA",
    "BESTBUY",
    "CORNING",
    "PEPSICO",
    "AMD",
    "MGMRESORTS",
    "CVSHEALTH",
    "JOHNSON_JOHNSON",
    "JNJ",
}


def normalize_company(raw: str) -> str:
    """Clean and normalize company names for consistent matching."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
    if cleaned in ("ULTA", "ULTABEAUTY"):
        return "ULTABEAUTY"
    if cleaned in ("JNJ", "JOHNSONJOHNSON"):
        return "JOHNSON_JOHNSON"
    return cleaned


def extract_document_metadata(filename: str, text: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract structured metadata from document filename and optional text prefix.

    Examples:
        'WALMART_2020_10K_p50.txt' -> {'company': 'WALMART', 'year': 2020, 'doc_type': '10-K'}
        'Pfizer_2023Q2_10Q_p40.txt' -> {'company': 'PFIZER', 'year': 2023, 'quarter': 'Q2', 'doc_type': '10-Q'}
    """
    metadata: Dict[str, Any] = {}
    base_name = filename.rsplit("/", 1)[-1]
    # Remove hex prefixes from temporary tenant storage if present (e.g. 8483935b_WALMART...)
    clean_name = re.sub(r"^[0-9a-f]{16}_", "", base_name, flags=re.IGNORECASE)

    parts = [p for p in re.split(r"[_.\-\s]+", clean_name) if p]

    # 1. Company Extraction from tokens or clean_name
    for token in parts:
        token_upper = normalize_company(token)
        for known in KNOWN_COMPANIES:
            if known in token_upper or token_upper in known:
                metadata["company"] = normalize_company(known)
                break
        if "company" in metadata:
            break

    if "company" not in metadata and parts:
        first_tok = parts[0].upper()
        if len(first_tok) >= 2 and first_tok.isalpha():
            metadata["company"] = first_tok

    # 2. Year Extraction
    for token in parts:
        yr_m = re.search(r"(?<!\d)(19\d\d|20[0-3]\d)(?!\d)", token)
        if yr_m:
            metadata["year"] = int(yr_m.group(1))
            break

    if "year" not in metadata and text:
        fy_match = FY_YEAR_PATTERN.search(text[:600])
        if fy_match:
            raw_yr = fy_match.group(1)
            metadata["year"] = int(f"20{raw_yr}" if len(raw_yr) == 2 else raw_yr)
        else:
            text_yr = re.search(r"(?<!\d)(19\d\d|20[0-3]\d)(?!\d)", text[:400])
            if text_yr:
                metadata["year"] = int(text_yr.group(1))

    # 3. Quarter Extraction
    for token in parts:
        q_m = re.search(r"(?:(?<=\d)|(?<=\b))Q[1-4]\b|(?:\b[1-4]Q\b)", token, re.IGNORECASE)
        if q_m:
            raw_q = q_m.group(0).upper()
            if raw_q.endswith("Q"):
                raw_q = f"Q{raw_q[0]}"
            metadata["quarter"] = raw_q
            break

    # 4. Document Type Extraction
    for token in parts:
        t_m = re.search(r"\b(10K|10Q|8K|EARNINGS)\b", token, re.IGNORECASE)
        if t_m:
            raw_type = t_m.group(1).upper()
            if raw_type in ("10K", "10Q", "8K"):
                metadata["doc_type"] = f"{raw_type[:2]}-{raw_type[2:]}"
            else:
                metadata["doc_type"] = raw_type
            break

    if "doc_type" not in metadata and text:
        t_text = re.search(r"\b(10-?K|10-?Q|8-?K|EARNINGS)\b", text[:500], re.IGNORECASE)
        if t_text:
            raw_type = t_text.group(1).upper().replace("-", "")
            if raw_type in ("10K", "10Q", "8K"):
                metadata["doc_type"] = f"{raw_type[:2]}-{raw_type[2:]}"
            else:
                metadata["doc_type"] = raw_type

    return metadata


def extract_query_metadata(
    query: str,
    available_companies: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Detect metadata filter criteria referenced within a user's natural question.

    Examples:
        "What was Walmart's net revenue in FY2020?" -> {'company': 'WALMART', 'year': 2020}
        "Did Verizon increase its debt in 2022?" -> {'company': 'VERIZON', 'year': 2022}
    """
    filters: Dict[str, Any] = {}
    q_upper = query.upper()

    # 1. Detect Company
    companies_to_check = available_companies or KNOWN_COMPANIES
    for comp in companies_to_check:
        comp_norm = normalize_company(comp)
        # Match as full word
        pattern = rf"\b{re.escape(comp)}\b"
        if re.search(pattern, q_upper) or (comp_norm and re.search(rf"\b{re.escape(comp_norm)}\b", q_upper)):
            filters["company"] = comp_norm
            break

    # 2. Detect Year (only filter by year if a single distinct year is referenced;
    # multi-year questions such as 'FY2018 - FY2020' or 'between 2021 and 2022' should
    # retrieve across all relevant years for that company).
    all_years = [int(y) for y in re.findall(r"(?<!\d)(?:19\d\d|20[0-3]\d)(?!\d)", query)]
    for m in re.finditer(r"\bFY\s*(\d{2})\b", query, re.IGNORECASE):
        all_years.append(int(f"20{m.group(1)}"))
    distinct_years = sorted(set(all_years))
    if len(distinct_years) == 1:
        filters["year"] = distinct_years[0]

    # 3. Detect Quarter
    q_match = QUARTER_PATTERN.search(query)
    if q_match:
        q_str = q_match.group(1).upper()
        if q_str.endswith("Q"):
            q_str = f"Q{q_str[0]}"
        filters["quarter"] = q_str

    # 4. Detect Doc Type
    type_match = DOC_TYPE_PATTERN.search(query)
    if type_match:
        raw_type = type_match.group(1).upper().replace("-", "")
        if raw_type in ("10K", "10Q", "8K"):
            filters["doc_type"] = f"{raw_type[:2]}-{raw_type[2:]}"
        else:
            filters["doc_type"] = raw_type

    return filters


def build_chroma_filter(
    user_id: str,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a valid ChromaDB filter dictionary combining tenant isolation (user_id)
    and any optional metadata constraints using Chroma's '$and' syntax.

    ChromaDB specification:
    - Single field: {"user_id": user_id}
    - Multiple fields: {"$and": [{"user_id": user_id}, {"year": 2020}]}
    """
    if not user_id:
        raise ValueError("user_id is required for Chroma filter construction")

    clauses: List[Dict[str, Any]] = [{"user_id": user_id}]

    if metadata_filter:
        for k, v in metadata_filter.items():
            if k == "user_id" or v is None:
                continue
            clauses.append({k: v})

    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
