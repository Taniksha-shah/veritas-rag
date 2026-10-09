import argparse
import json
import hashlib
import random
import re
from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter


# ============================================================
# Configuration
# ============================================================

DEFAULT_INPUT = "claims_dataset.jsonl"
DEFAULT_OUTPUT_DIR = "prepared_data"

RANDOM_SEED = 42

# A passage with fewer than this many words is flagged.
MIN_PASSAGE_WORDS = 5

# Maximum number of claims allowed for one passage.
# This is only a quality flag, not an automatic rejection.
MAX_CLAIMS = 30

# Near-duplicate threshold.
# 0.95 means two passages with >=95% similarity are considered duplicates.
NEAR_DUP_THRESHOLD = 0.95


# ============================================================
# Utility functions
# ============================================================

def normalize_text(text):
    """Normalize text for duplicate detection."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def text_hash(text):
    """Create a stable hash for exact duplicate detection."""
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def word_count(text):
    return len(text.split())


def is_valid_record(record):
    """
    Basic structural validation.

    Expected format:
    {
        "passage": "...",
        "claims": ["...", "..."]
    }
    """

    if not isinstance(record, dict):
        return False, "record_is_not_object"

    if "passage" not in record:
        return False, "missing_passage"

    if "claims" not in record:
        return False, "missing_claims"

    if not isinstance(record["passage"], str):
        return False, "passage_is_not_string"

    if not record["passage"].strip():
        return False, "empty_passage"

    if not isinstance(record["claims"], list):
        return False, "claims_is_not_list"

    if len(record["claims"]) == 0:
        return False, "empty_claim_list"

    if not all(isinstance(c, str) for c in record["claims"]):
        return False, "claim_is_not_string"

    if not all(c.strip() for c in record["claims"]):
        return False, "empty_claim"

    return True, None


def normalize_claims(claims):
    """Remove duplicate claims while preserving order."""
    seen = set()
    cleaned = []

    for claim in claims:
        claim = re.sub(r"\s+", " ", claim.strip())

        key = claim.lower()

        if key not in seen:
            seen.add(key)
            cleaned.append(claim)

    return cleaned


# ============================================================
# Quality checks
# ============================================================

def quality_flags(passage, claims):
    """
    Return quality flags rather than immediately deleting records.

    This is important because the dataset is silver-labeled.
    We want to inspect questionable examples rather than silently
    throwing them away.
    """

    flags = []

    passage_words = word_count(passage)

    if passage_words < MIN_PASSAGE_WORDS:
        flags.append("very_short_passage")

    if len(claims) > MAX_CLAIMS:
        flags.append("many_claims")

    # Detect extremely short claims.
    for claim in claims:
        if len(claim.split()) < 3:
            flags.append("very_short_claim")
            break

    # Detect likely metadata claims.
    metadata_patterns = [
        r"\bauthored by\b",
        r"\baffiliated with\b",
        r"\bdepartment of\b",
        r"\bemail\b",
        r"\bcorrespondence\b",
        r"\bcontact\b",
        r"\bannual meeting\b",
        r"\bposter abstract\b",
        r"\btype:\s*(original research|review|case report)",
    ]

    for claim in claims:
        claim_lower = claim.lower()

        if any(re.search(pattern, claim_lower) for pattern in metadata_patterns):
            flags.append("possible_metadata_claim")
            break

    # Detect claims that are exact duplicates of each other.
    normalized = [normalize_text(c) for c in claims]

    if len(normalized) != len(set(normalized)):
        flags.append("duplicate_claims")

    return sorted(set(flags))


# ============================================================
# Load JSONL
# ============================================================

def load_jsonl(input_file):
    records = []
    errors = []

    with open(input_file, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
                records.append(record)

            except json.JSONDecodeError as e:
                errors.append({
                    "line": line_number,
                    "error": str(e)
                })

    return records, errors


# ============================================================
# Exact duplicate removal
# ============================================================

def remove_exact_duplicates(records):

    seen = set()
    unique_records = []
    duplicates = []

    for record in records:

        passage = record["passage"]
        key = text_hash(passage)

        if key in seen:
            duplicates.append(record)
            continue

        seen.add(key)
        unique_records.append(record)

    return unique_records, duplicates


# ============================================================
# Near duplicate detection
# ============================================================

def similarity(a, b):
    return SequenceMatcher(
        None,
        normalize_text(a),
        normalize_text(b)
    ).ratio()


def remove_near_duplicates(records, threshold=NEAR_DUP_THRESHOLD):

    unique_records = []
    duplicates = []

    # Compare against already accepted passages.
    #
    # This is O(n²), which is acceptable for a few thousand
    # passages but may become slow for very large datasets.

    for i, record in enumerate(records):

        passage = record["passage"]

        duplicate_found = False

        for existing in unique_records:

            score = similarity(
                passage,
                existing["passage"]
            )

            if score >= threshold:

                duplicates.append({
                    "record": record,
                    "duplicate_of": existing["passage"],
                    "similarity": round(score, 4)
                })

                duplicate_found = True
                break

        if not duplicate_found:
            unique_records.append(record)

    return unique_records, duplicates


# ============================================================
# Prepare records
# ============================================================

def prepare_records(records):

    valid_records = []
    rejected_records = []
    quality_report = []

    for index, record in enumerate(records):

        valid, reason = is_valid_record(record)

        if not valid:

            rejected_records.append({
                "index": index,
                "reason": reason,
                "record": record
            })

            continue

        passage = record["passage"].strip()

        claims = normalize_claims(record["claims"])

        flags = quality_flags(
            passage,
            claims
        )

        prepared = {
            "passage": passage,
            "claims": claims
        }

        valid_records.append(prepared)

        if flags:

            quality_report.append({
                "index": index,
                "flags": flags,
                "passage": passage,
                "claims": claims
            })

    return (
        valid_records,
        rejected_records,
        quality_report
    )


# ============================================================
# Train / validation / test split
# ============================================================

def split_dataset(records):

    records = records.copy()

    random.seed(RANDOM_SEED)
    random.shuffle(records)

    n = len(records)

    train_end = int(n * 0.80)
    val_end = train_end + int(n * 0.10)

    train = records[:train_end]
    validation = records[train_end:val_end]
    test = records[val_end:]

    return train, validation, test


# ============================================================
# Save JSONL
# ============================================================

def save_jsonl(records, output_file):

    with open(output_file, "w", encoding="utf-8") as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )


def save_json(records, output_file):

    with open(output_file, "w", encoding="utf-8") as f:

        json.dump(
            records,
            f,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Prepare Veritas atomic-claim silver-label dataset."
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Input JSONL dataset."
    )

    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for prepared datasets."
    )

    parser.add_argument(
        "--near_dup_threshold",
        type=float,
        default=NEAR_DUP_THRESHOLD,
        help="Similarity threshold for near-duplicate removal."
    )

    args = parser.parse_args()

    input_file = Path(args.input)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("VERITAS — STAGE B: DATA PREPARATION")
    print("=" * 70)

    print(f"\nInput:  {input_file}")
    print(f"Output: {output_dir}")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    records, json_errors = load_jsonl(input_file)

    print(f"\nLoaded records: {len(records)}")

    if json_errors:
        print(
            f"JSON parsing errors: {len(json_errors)}"
        )

        save_json(
            json_errors,
            output_dir / "json_errors.json"
        )

    # --------------------------------------------------------
    # Structural validation
    # --------------------------------------------------------

    (
        valid_records,
        rejected_records,
        quality_report
    ) = prepare_records(records)

    print(
        f"Valid records: {len(valid_records)}"
    )

    print(
        f"Rejected records: {len(rejected_records)}"
    )

    # --------------------------------------------------------
    # Exact duplicate removal
    # --------------------------------------------------------

    before_exact = len(valid_records)

    valid_records, exact_duplicates = remove_exact_duplicates(
        valid_records
    )

    print(
        f"Exact duplicates removed: "
        f"{before_exact - len(valid_records)}"
    )

    # --------------------------------------------------------
    # Near duplicate removal
    # --------------------------------------------------------

    before_near = len(valid_records)

    valid_records, near_duplicates = remove_near_duplicates(
        valid_records,
        threshold=args.near_dup_threshold
    )

    print(
        f"Near duplicates removed: "
        f"{before_near - len(valid_records)}"
    )

    # --------------------------------------------------------
    # Save quality reports
    # --------------------------------------------------------

    save_json(
        rejected_records,
        output_dir / "rejected_records.json"
    )

    save_json(
        quality_report,
        output_dir / "quality_flags.json"
    )

    save_json(
        near_duplicates,
        output_dir / "near_duplicates.json"
    )

    # --------------------------------------------------------
    # Train / validation / test
    # --------------------------------------------------------

    train, validation, test = split_dataset(
        valid_records
    )

    print("\nDataset split:")
    print(f"  Train:      {len(train)}")
    print(f"  Validation: {len(validation)}")
    print(f"  Test:       {len(test)}")

    # --------------------------------------------------------
    # Save prepared datasets
    # --------------------------------------------------------

    save_jsonl(
        train,
        output_dir / "train.jsonl"
    )

    save_jsonl(
        validation,
        output_dir / "validation.jsonl"
    )

    save_jsonl(
        test,
        output_dir / "test.jsonl"
    )

    save_jsonl(
        valid_records,
        output_dir / "all_clean.jsonl"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "input_records": len(records),
        "valid_records_before_dedup": len(valid_records)
            + len(exact_duplicates)
            + len(near_duplicates),
        "exact_duplicates_removed": len(exact_duplicates),
        "near_duplicates_removed": len(near_duplicates),
        "final_records": len(valid_records),
        "rejected_records": len(rejected_records),
        "quality_flagged_records": len(quality_report),
        "train_records": len(train),
        "validation_records": len(validation),
        "test_records": len(test),
        "random_seed": RANDOM_SEED,
        "near_duplicate_threshold": args.near_dup_threshold
    }

    save_json(
        summary,
        output_dir / "dataset_summary.json"
    )

    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70)

    print("\nGenerated files:")

    print(
        f"  {output_dir / 'all_clean.jsonl'}"
    )

    print(
        f"  {output_dir / 'train.jsonl'}"
    )

    print(
        f"  {output_dir / 'validation.jsonl'}"
    )

    print(
        f"  {output_dir / 'test.jsonl'}"
    )

    print(
        f"  {output_dir / 'quality_flags.json'}"
    )

    print(
        f"  {output_dir / 'rejected_records.json'}"
    )

    print(
        f"  {output_dir / 'near_duplicates.json'}"
    )

    print(
        f"  {output_dir / 'dataset_summary.json'}"
    )


if __name__ == "__main__":
    main()