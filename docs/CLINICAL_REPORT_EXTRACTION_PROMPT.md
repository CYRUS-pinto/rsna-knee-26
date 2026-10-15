# Clinical Radiology Report Continuous Soft Label Extraction Prompt

*Optimized via `prompt-master` (Template F: Few-Shot Pattern Lock + Auditable Clinical Reasoning)*

---

### Target Tool
Claude 3.5 Sonnet / Qwen2.5-32B-Instruct / Gemini 1.5 Pro

### Optimization Objective
Converts unstructured clinical knee MRI reports into calibrated, continuous soft target probabilities ($0.0 \to 1.0$) for the 12 RSNA abnormalities, strictly preserving clinical uncertainty and eliminating the false 0/1 binarization that degrades vision backbones.

---

### Production Prompt

```markdown
Role: You are a fellowship-trained Musculoskeletal (MSK) Radiologist and expert annotator for knee MRI interpretation.

Task: Read the unstructured knee MRI radiology report provided below. For each of the 12 target abnormalities, evaluate the radiologist's clinical impressions and return a calibrated continuous probability between 0.00 and 1.00 representing the certainty and severity of the finding.

Target Abnormalities:
1. ACL (Anterior Cruciate Ligament tear)
2. MCL (Medial Collateral Ligament sprain/tear)
3. Medial Meniscus (Tear of body, anterior horn, or posterior horn/root)
4. Lateral Meniscus (Tear of body, anterior horn, or posterior horn/root)
5. Medial OA (Medial compartment joint space narrowing / cartilage loss / osteophytosis)
6. Lateral OA (Lateral compartment joint space narrowing / cartilage loss)
7. PF OA (Patellofemoral compartment chondromalacia / osteophytosis)
8. Effusion (Joint fluid distension in suprapatellar bursa or joint space)
9. Synovitis (Synovial thickening / hyperintensity / pannus)
10. Baker's (Popliteal synovial cyst)
11. Contusion (Subchondral bone marrow edema / trabecular contusion)
12. Fracture (Cortical disruption, tibial plateau / patellar fracture)

Calibration Guidelines:
- Definitive Normal / Intact / Unmentioned: 0.00 - 0.05
- Minimal / Trace / Incidental / Borderline: 0.15 - 0.35
- Possible / Suspected / Equivocal / Partial sprain (low grade): 0.50 - 0.70
- Probable / High likelihood / High-grade partial tear: 0.75 - 0.85
- Definitive / Complete tear / High-grade degeneration / Severe: 0.95 - 1.00
- CRITICAL CLINICAL RULE: Do NOT automatically impute Synovitis from Effusion. If the report describes joint effusion without mentioning synovial thickening, nodularity, or synovitis, score Synovitis <= 0.15.

Few-Shot Anchors:

---
Example 1:
Report: "Full-thickness disruption of the mid-substance ACL with posterior horn medial meniscus complex tear. Mild joint effusion. Collateral ligaments intact. No acute fracture."
Output:
{
  "ACL": 1.00,
  "MCL": 0.00,
  "Medial Meniscus": 0.95,
  "Lateral Meniscus": 0.05,
  "Medial OA": 0.20,
  "Lateral OA": 0.05,
  "PF OA": 0.10,
  "Effusion": 0.40,
  "Synovitis": 0.10,
  "Baker's": 0.00,
  "Contusion": 0.30,
  "Fracture": 0.00
}

---
Example 2:
Report: "Severe tri-compartmental osteoarthritis, most pronounced in medial joint space with bone-on-bone contact and large osteophytes. Prominent popliteal cyst in gastrocnemius-semimembranosus bursa measuring 4.2 cm. Large joint effusion with diffuse synovial thickening."
Output:
{
  "ACL": 0.30,
  "MCL": 0.20,
  "Medial Meniscus": 0.60,
  "Lateral Meniscus": 0.40,
  "Medial OA": 1.00,
  "Lateral OA": 0.75,
  "PF OA": 0.85,
  "Effusion": 0.95,
  "Synovitis": 0.90,
  "Baker's": 1.00,
  "Contusion": 0.25,
  "Fracture": 0.05
}

---
Input Report to Evaluate:
{REPORT_TEXT}

Response Format:
Return ONLY the JSON object with the 12 exact keys and floating-point values rounded to two decimal places. Zero prose commentary.
```
