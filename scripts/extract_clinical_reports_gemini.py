"""
Batch Clinical Radiology Report Extractor using Google Gemini Flash.
Optimized via Prompt Master (Template F: Few-Shot Pattern Lock + Calibrated Probability Scale).
Extracts continuous soft targets (0.00 -> 1.00) across 12 RSNA knee pathologies.
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
from typing import Dict, Any, Optional

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"
]

SYSTEM_PROMPT = """Role: You are an expert Musculoskeletal (MSK) Radiologist annotator specializing in knee MRI interpretation.

Task: Extract calibrated continuous probabilities (0.00 to 1.00) for all 12 target knee abnormalities from the radiology report provided below.

Grounding Constraints:
- Base your evaluations ONLY on the explicit text in the provided report. Do NOT extrapolate or assume findings that are not documented.
- If a finding is not mentioned or described as completely normal, assign a probability <= 0.05.
- Disentangle Synovitis and Effusion: Joint effusion alone does NOT imply synovitis. Score Synovitis > 0.20 ONLY if synovial thickening, proliferation, or pannus is explicitly stated.
- Output strictly valid JSON matching the exact schema below. Do not wrap in markdown or include conversational text.

Calibrated Probability Scale:
- 0.00 - 0.05: Normal / Unremarkable / Intact / Unmentioned
- 0.15 - 0.35: Trace / Minimal / Low-grade sprain / Incidental / Questionable
- 0.50 - 0.70: Possible / Equivocal / Partial-thickness tear / Moderate degeneration
- 0.75 - 0.85: Probable / High likelihood / High-grade partial tear
- 0.95 - 1.00: Definitive / Complete tear / Severe osteoarthritis / Full disruption

JSON Output Schema:
{
  "study_id": "<string>",
  "probabilities": {
    "ACL": <float between 0.00 and 1.00>,
    "MCL": <float between 0.00 and 1.00>,
    "Medial Meniscus": <float between 0.00 and 1.00>,
    "Lateral Meniscus": <float between 0.00 and 1.00>,
    "Medial OA": <float between 0.00 and 1.00>,
    "Lateral OA": <float between 0.00 and 1.00>,
    "PF OA": <float between 0.00 and 1.00>,
    "Effusion": <float between 0.00 and 1.00>,
    "Synovitis": <float between 0.00 and 1.00>,
    "Baker's": <float between 0.00 and 1.00>,
    "Contusion": <float between 0.00 and 1.00>,
    "Fracture": <float between 0.00 and 1.00>
  },
  "clinical_rationale": "<concise 1-sentence evidence summary>"
}

Few-Shot Anchor:
Input Report:
"IMPRESSION: Complete tear of the proximal anterior cruciate ligament with anterior translation of the tibia. Bone contusion in the posterior lateral tibial plateau and lateral femoral condyle. Small joint effusion. No meniscal tear or fracture identified."
Output:
{
  "study_id": "EXAMPLE_001",
  "probabilities": {
    "ACL": 1.00,
    "MCL": 0.00,
    "Medial Meniscus": 0.02,
    "Lateral Meniscus": 0.02,
    "Medial OA": 0.05,
    "Lateral OA": 0.05,
    "PF OA": 0.05,
    "Effusion": 0.35,
    "Synovitis": 0.05,
    "Baker's": 0.00,
    "Contusion": 0.98,
    "Fracture": 0.00
  },
  "clinical_rationale": "Definitive ACL disruption with classic pivot-shift osseous contusion pattern and small reactive effusion."
}
"""

def extract_single_report(client, model_name: str, study_id: str, report_text: str) -> Optional[Dict[str, Any]]:
    prompt = f"Input Report to Evaluate:\nReport ID: {study_id}\nText: \"\"\"\n{report_text}\n\"\"\""
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[SYSTEM_PROMPT, prompt],
            config={
                'response_mime_type': 'application/json',
                'temperature': 0.1,
            }
        )
        data = json.loads(response.text)
        return data
    except Exception as e:
        print(f"[ERROR] Failed to extract study {study_id}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Extract soft continuous labels from clinical reports using Gemini Flash.")
    parser.add_argument("--input_csv", type=str, default="meta/train.csv", help="Input CSV containing StudyInstanceUID and Report")
    parser.add_argument("--output_file", type=str, default="data/gemini_extracted_soft_labels.parquet", help="Output Parquet path")
    parser.add_argument("--model", type=str, default="gemini-1.5-flash", help="Gemini model name")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional sample limit for testing")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY or GOOGLE_API_KEY environment variable is not set.")
        sys.exit(1)

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
    except ImportError:
        try:
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=api_key)
            client = genai_legacy
        except ImportError:
            print("[ERROR] Neither 'google-genai' nor 'google-generativeai' package is installed.")
            print("Run: pip install google-genai")
            sys.exit(1)

    if not os.path.exists(args.input_csv):
        print(f"[ERROR] Input file {args.input_csv} does not exist.")
        sys.exit(1)

    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} total reports from {args.input_csv}")
    if args.max_samples:
        df = df.head(args.max_samples)
        print(f"Subsampled to {len(df)} records for extraction.")

    results = []
    t0 = time.time()
    for idx, row in df.iterrows():
        study_id = str(row.get("StudyInstanceUID", f"ID_{idx}"))
        report_text = str(row.get("Report", ""))
        if not report_text or report_text.strip() == "":
            continue

        res = extract_single_report(client, args.model, study_id, report_text)
        if res and "probabilities" in res:
            record = {"StudyInstanceUID": study_id}
            record.update(res["probabilities"])
            record["clinical_rationale"] = res.get("clinical_rationale", "")
            results.append(record)
        
        if (idx + 1) % 25 == 0 or (idx + 1) == len(df):
            elapsed = time.time() - t0
            print(f"Processed {idx + 1}/{len(df)} reports ({len(results)} successful) in {elapsed:.1f}s")

    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    if args.output_file.endswith(".parquet"):
        out_df.to_parquet(args.output_file, index=False)
    else:
        out_df.to_csv(args.output_file, index=False)

    print(f"Successfully saved {len(out_df)} extracted soft label records to {args.output_file}")

if __name__ == "__main__":
    main()
