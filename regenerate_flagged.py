import json
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL = "openai/gpt-oss-20b"
INPUT_FILE = "prepared_data/quality_flags.json"
OUTPUT_FILE = "regenerated_flagged.jsonl"

SYSTEM_PROMPT = """You are an expert at decomposing scientific text into atomic factual claims.

An atomic claim is a single, self-contained, independently checkable proposition.

Rules:
- Each claim must express exactly ONE independently verifiable proposition.
- Split compound statements only when each resulting claim is independently
  meaningful and checkable.
- Do not split a claim merely because it contains multiple related details.
- Do not split a scientific finding from the numerical or statistical evidence
  that describes or quantifies that finding.
- Preserve the original meaning and certainty of every statement.
- Preserve epistemic qualifiers such as "may", "might", "could", "likely",
  "suggests", and "is associated with".
- Preserve numbers, units, percentages, ranges, statistical values, and
  named entities exactly as written.

STRICT METADATA EXCLUSION:
- Never output claims about author names, author degrees or credentials,
  affiliations, departments, institutions, email addresses, conference names,
  abstract titles, publication types, manuscript types, or citation information.
- If a passage contains both metadata and scientific content, ignore the
  metadata and extract only the scientific, methodological, experimental,
  statistical, or clinically relevant claims.
- Study methodology is NOT metadata. Retain claims about study design,
  sample size, participants, interventions, measurements, observation periods,
  statistical methods, and experimental procedures.

- Resolve pronouns and implicit references so every claim is understandable
  independently.
- Do not create claims from rhetorical language or filler.
- Prefer fewer complete claims over fragmented claims.

Output ONLY a JSON list of strings, nothing else.
"""


def regenerate(client, passage):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Passage:\n{passage}"}
        ],
        temperature=0.0,
    )
    print("Finish reason:", response.choices[0].finish_reason)
    print("Usage:", response.usage)
    print("Raw response:", repr(response.choices[0].message.content))

    content = response.choices[0].message.content
    print("RAW RESPONSE:", repr(content))
    content = (content or "").strip()

    if not content:
        raise ValueError("Model returned empty content.")

    claims = json.loads(content)

    if not isinstance(claims, list):
        raise ValueError("Model output is not a JSON list.")

    if not all(isinstance(c, str) for c in claims):
        raise ValueError("Claims list contains non-string values.")

    return claims


def main():
    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1"
    )

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        flagged = json.load(f)

    print(f"Found {len(flagged)} flagged passages.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for i, record in enumerate(flagged, start=1):
            passage = record["passage"]

            print(f"\n[{i}/{len(flagged)}] Processing index {record['index']}...")

            try:
                claims = regenerate(client, passage)

                result = {
                    "index": record["index"],
                    "passage": passage,
                    "claims": claims
                }

                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()

                print(f"  Generated {len(claims)} claims.")

            except Exception as e:
                print(f"  ERROR: {e}")

            # Stay below the ~28 RPM throttle you were using
            if i < len(flagged):
                time.sleep(3)

    print(f"\nSaved regenerated records to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()