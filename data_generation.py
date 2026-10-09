"""
Stage A — Silver-label data generation for atomic claim decomposition.

Purpose
-------
Your PubMed papers give you raw passages, not (passage -> atomic claims)
training pairs. This script produces those pairs by calling a strong LLM
ONE TIME, OFFLINE, purely to generate training labels. The model you
actually fine-tune (train_stage_a.py) and deploy (inference.py) never calls
this API — it runs standalone.

Works with OpenAI OR any OpenAI-compatible free-tier endpoint (Groq, etc.)
------------------------------------------------------------------------
Default settings point at Groq (same account/key you already use for
generation), since OpenAI requires paid credits with no free tier. To use
OpenAI instead, pass --api_base "" (empty) and --model gpt-4o-mini.

Usage (Groq — free, default)
-----------------------------
1. Put your PubMed papers (.txt or .pdf) in an input folder, e.g. ./raw_papers/
2. Put GROQ_API_KEY=gsk_... in a .env file in this same folder (loaded automatically)
3. Run:
       python data_generation.py --input_dir ./raw_papers --output_file claims_dataset.jsonl

Usage (OpenAI, if you have paid credits)
-----------------------------------------
       Put OPENAI_API_KEY=sk-... in your .env file instead/as well, then:
       python data_generation.py --input_dir ./raw_papers --output_file claims_dataset.jsonl \
           --api_base "" --api_key_env OPENAI_API_KEY --model gpt-4o-mini --rpm 0

Note: reads keys via python-dotenv, so a .env file in the current working
directory (or any parent directory) is picked up automatically — no need to
export environment variables manually or configure your terminal.

Output format (JSONL, one object per line):
    {"passage": "<source passage text>", "claims": ["<atomic claim 1>", "<atomic claim 2>", ...]}

Resumable: if --output_file already exists, passages already labeled there
are skipped on the next run (matched by exact passage text) instead of
re-labeled and re-billed/re-throttled. Useful since Groq's free tier caps
daily requests, so a big dataset may take more than one day's run to finish.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from datasketch import MinHash, MinHashLSH
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from tqdm import tqdm

load_dotenv()  # reads .env in the current (or a parent) directory into os.environ

# ---------------------------------------------------------------------------
# Few-shot prompt for atomic claim decomposition.
# Style follows FActScore / RARR-style decomposition: each claim should be a
# single, self-contained, checkable factual statement (no pronouns, no
# compound "and" statements, no unstated antecedents).
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert at decomposing scientific text into atomic factual claims.

An atomic claim is a single, self-contained, independently checkable proposition.

Rules:
- Each claim must express exactly ONE independently verifiable proposition.
- Split compound statements only when each resulting claim is independently
  meaningful and checkable.
- Do not split a claim merely because it contains multiple related details.
  Keep details together when they describe the same proposition, relationship,
  study objective, or measurement.
- Do not split a scientific finding from the numerical or statistical evidence
  that describes or quantifies that finding.
- Do not turn every item in a list into a separate claim unless each item
  represents a distinct independently verifiable proposition.
- Resolve all pronouns and implicit references (e.g. "it", "this", "the drug")
  using context from the passage, so each claim can be understood in isolation.
- Preserve the original meaning and certainty of every statement.
- Preserve epistemic qualifiers such as "may", "might", "could", "likely",
  "suggests", "is associated with", "was theorized", and "was assumed".
  Do not make a claim more or less certain than the passage.
- Preserve numbers, units, percentages, ranges, confidence intervals,
  statistical values, and named entities exactly as written in the passage.
- STRICTLY EXCLUDE bibliographic and author metadata from the output.
  Never create claims about author names, author degrees or credentials,
  affiliations, departments, institutions, email addresses, conference
  names, abstract titles, publication types, manuscript types, or citation
  information.
- If a passage contains both metadata and scientific content, ignore the
  metadata and extract only the scientific, methodological, experimental,
  statistical, or clinically relevant claims.
- Study methodology is NOT metadata. Claims about study design, sample size,
  participants, interventions, measurements, observation periods, statistical
  methods, and experimental procedures should be retained when they are
  independently verifiable.
- Do not include pure opinion, rhetorical language, or non-factual filler
  such as "further research is needed".
- Prefer fewer complete claims over fragmented claims.
- Every claim must be understandable and independently verifiable without
  relying on the original passage.
- When several statements describe different aspects or consequences of the
  same finding, intervention, relationship, or outcome, keep them together
  when separating them would lose their shared context or meaning.
- Output ONLY a JSON list of strings, nothing else.
"""

FEW_SHOT_EXAMPLES = [
    {
        "passage": (
            "Metformin is a first-line treatment for type 2 diabetes. It reduces "
            "hepatic glucose production and improves insulin sensitivity, and in a "
            "2019 meta-analysis of 34 trials it was associated with a 17% reduction "
            "in cardiovascular events compared to placebo."
        ),
        "claims": [
            "Metformin is a first-line treatment for type 2 diabetes.",
            "Metformin reduces hepatic glucose production.",
            "Metformin improves insulin sensitivity.",
            "A 2019 meta-analysis included 34 trials of metformin.",
            "In that 2019 meta-analysis, metformin was associated with a 17% reduction in cardiovascular events compared to placebo.",
        ],
    },
    {
        "passage": (
            "The Havel-Hakimi algorithm constructs a simple graph realizing a given "
            "degree sequence, or proves that none exists. It was published by Havel "
            "in 1955 and independently by Hakimi in 1962."
        ),
        "claims": [
            "The Havel-Hakimi algorithm constructs a simple graph realizing a given degree sequence.",
            "The Havel-Hakimi algorithm proves that no such graph exists if one cannot be constructed.",
            "The Havel-Hakimi algorithm was published by Havel in 1955.",
            "The Havel-Hakimi algorithm was independently published by Hakimi in 1962.",
        ],
    },
]


def build_messages(passage: str):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": f"Passage:\n{ex['passage']}"})
        messages.append({"role": "assistant", "content": json.dumps(ex["claims"])})
    messages.append({"role": "user", "content": f"Passage:\n{passage}"})
    return messages


def decompose_passage(client: OpenAI, passage: str, model: str, max_retries: int = 3):
    """Call the LLM once to decompose a passage into atomic claims. Returns a list of strings."""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=build_messages(passage),
                temperature=0.0,
                max_tokens=1024,
                reasoning_effort="low",
            )
            print(
                f"Tokens — prompt: {resp.usage.prompt_tokens}, "
                f"completion: {resp.usage.completion_tokens}, "
                f"total: {resp.usage.total_tokens}"
            )
            content = resp.choices[0].message.content.strip()

            print("\n--- RAW MODEL OUTPUT ---")
            print(repr(content))
            print("--- END RAW OUTPUT ---\n")

            content = re.sub(
                r"^```(?:json)?|```$",
                "",
                content,
                flags=re.MULTILINE
            ).strip()

            claims = json.loads(content)

            if isinstance(claims, list) and all(isinstance(c, str) for c in claims):
                return claims

        except Exception as e:
            print(f"  [retry {attempt + 1}/{max_retries}] error: {e}")
            print(
                f"  Raw content: "
                f"{repr(content) if 'content' in locals() else 'NO CONTENT'}"
            )
            time.sleep(2 ** attempt)

    return []


def load_text_from_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_into_passages(text: str, min_words: int = 25, max_words: int = 120):
    """Split raw paper text into paragraph-sized passages suitable for decomposition.

    Merges short paragraphs and splits long ones on sentence boundaries so each
    passage is roughly a coherent, checkable chunk (not a whole page, not a single
    fragment).
    """
    # Normalize whitespace, split on blank lines / paragraph breaks first.
    raw_paragraphs = re.split(r"\n\s*\n|\r\n\s*\r\n", text)
    sentences = []
    for para in raw_paragraphs:
        para = re.sub(r"\s+", " ", para).strip()
        if para:
            sentences.extend(re.split(r"(?<=[.!?])\s+", para))

    passages = []
    buffer = []
    word_count = 0
    for sent in sentences:
        n = len(sent.split())
        if word_count + n > max_words and word_count >= min_words:
            passages.append(" ".join(buffer))
            buffer, word_count = [], 0
        buffer.append(sent)
        word_count += n
    if buffer and word_count >= min_words:
        passages.append(" ".join(buffer))
    return passages


def _shingle(text: str, n: int = 5):
    """Word n-grams, used as MinHash input. n=5 is a common default for
    sentence/paragraph-level near-dup detection (shorter n catches more
    overlap but risks false positives on short boilerplate phrases)."""
    words = text.lower().split()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def dedup_passages(passages: list, jaccard_threshold: float = 0.85, num_perm: int = 128):
    """Remove exact and near-duplicate passages before they're sent to the LLM.

    Two stages:
    1. Exact dedup — normalize whitespace/case, drop literal repeats. Catches
       the same paragraph appearing in multiple papers fetched under
       different --query terms.
    2. Near-dup (MinHash + LSH) — catches paraphrased/overlapping passages
       that aren't byte-identical, e.g. near-identical methods-section
       boilerplate or an abstract that repeats most of its own introduction.
       jaccard_threshold=0.85 means passages need ~85% shingle overlap to be
       considered duplicates; lower it (e.g. 0.7) to dedup more aggressively.

    Keeps the first occurrence of each duplicate cluster, drops the rest.
    """
    # --- Stage 1: exact dedup ---
    seen_exact = set()
    exact_deduped = []
    for p in passages:
        key = re.sub(r"\s+", " ", p).strip().lower()
        if key and key not in seen_exact:
            seen_exact.add(key)
            exact_deduped.append(p)
    n_exact_dropped = len(passages) - len(exact_deduped)

    # --- Stage 2: near-dup via MinHash LSH ---
    lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=num_perm)
    unique = []
    for i, passage in enumerate(exact_deduped):
        mh = MinHash(num_perm=num_perm)
        for shingle in _shingle(passage):
            mh.update(shingle.encode("utf-8"))
        if lsh.query(mh):
            continue  # near-duplicate of something already kept
        lsh.insert(f"p{i}", mh)
        unique.append(passage)
    n_near_dropped = len(exact_deduped) - len(unique)

    print(f"Dedup: {len(passages)} -> {len(unique)} passages "
          f"(dropped {n_exact_dropped} exact, {n_near_dropped} near-duplicate)")
    return unique


def load_already_labeled(output_file: str):
    """Read passages already present in an existing output file, so a rerun
    (e.g. after hitting Groq's daily cap, or a crash) doesn't re-label and
    re-spend rate-limit budget on work that's already done."""
    done = set()
    path = Path(output_file)
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("passage"):
                    done.add(obj["passage"])
            except json.JSONDecodeError:
                continue
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Folder of .txt/.pdf PubMed papers")
    parser.add_argument("--output_file", type=str, default="claims_dataset.jsonl")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b",
                         help="Groq model (default, free). Use gpt-4o-mini with --api_base '' for OpenAI.")
    parser.add_argument("--api_base", type=str, default="https://api.groq.com/openai/v1",
                         help="OpenAI-compatible endpoint. Default is Groq's free endpoint. Pass '' for OpenAI's default.")
    parser.add_argument("--api_key_env", type=str, default="GROQ_API_KEY",
                         help="Env var holding the API key. Use OPENAI_API_KEY for OpenAI.")
    parser.add_argument("--rpm", type=float, default=28,
                         help="Max requests/minute, throttled client-side. Groq free tier caps at 30; "
                              "set to 0 to disable throttling (e.g. for OpenAI with paid credits).")
    parser.add_argument("--max_passages_per_paper", type=int, default=40)
    parser.add_argument("--limit", type=int, default=None, help="Cap total passages processed (for a quick test run)")
    parser.add_argument("--dedup_threshold", type=float, default=0.85,
                         help="Jaccard similarity threshold for near-duplicate removal (lower = more aggressive)")
    parser.add_argument("--no_dedup", action="store_true", help="Skip deduplication entirely")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Environment variable {args.api_key_env} is not set. "
                          f"export {args.api_key_env}=... first.")
    client_kwargs = {"api_key": api_key}
    if args.api_base:
        client_kwargs["base_url"] = args.api_base
    client = OpenAI(**client_kwargs)

    input_dir = Path(args.input_dir)
    files = sorted([p for p in input_dir.glob("**/*") if p.suffix.lower() in (".txt", ".pdf")])
    if not files:
        raise SystemExit(f"No .txt or .pdf files found in {input_dir}")

    print(f"Found {len(files)} papers.")

    all_passages = []
    for f in files:
        try:
            text = load_text_from_file(f)
        except Exception as e:
            print(f"  skipping {f.name}: {e}")
            continue
        passages = chunk_into_passages(text)[: args.max_passages_per_paper]
        all_passages.extend(passages)

    print(f"Chunked into {len(all_passages)} raw passages.")

    if not args.no_dedup:
        all_passages = dedup_passages(all_passages, jaccard_threshold=args.dedup_threshold)

    if args.limit:
        all_passages = all_passages[: args.limit]

    already_done = load_already_labeled(args.output_file)
    if already_done:
        before = len(all_passages)
        all_passages = [p for p in all_passages if p not in already_done]
        print(f"Resuming: {before - len(all_passages)} passages already labeled in "
              f"{args.output_file}, skipping those.")

    print(f"Decomposing {len(all_passages)} passages with {args.model} "
          f"(~{args.rpm:.0f} req/min throttle)" if args.rpm else
          f"Decomposing {len(all_passages)} passages with {args.model} ...")

    sleep_between = 60.0 / args.rpm if args.rpm else 0
    n_written = 0
    # Append mode: safe to resume without clobbering already-labeled lines.
    with open(args.output_file, "a", encoding="utf-8") as out_f:
        for passage in tqdm(all_passages):
            claims = decompose_passage(client, passage, model=args.model)
            if claims:
                out_f.write(json.dumps({"passage": passage, "claims": claims}) + "\n")
                out_f.flush()
                n_written += 1
            if sleep_between:
                time.sleep(sleep_between)

    print(f"Wrote {n_written} new labeled examples to {args.output_file}")


if __name__ == "__main__":
    main()