# Prompt Master Integration & Tool Documentation

## 1. Overview
The **Prompt Master** skill (`nidhinjs/prompt-master`, v1.8.2) has been downloaded, verified, and installed into both:
- **Global Customizations:** `C:\Users\Cyrus\.gemini\config\skills\prompt-master\`
- **Workspace Customizations:** `.agents\skills\prompt-master\`

---

## 2. Skill Capabilities & Architecture

`prompt-master` enforces strict, token-efficient, production-grade prompt engineering principles designed to eliminate hallucination, ambiguity, and re-prompt loops.

### Key Enforced Rules:
1. **Primacy & Recency Anchoring:** Critical instructions placed at the very top and bottom to counter middle-loss attenuation.
2. **Explicit Verification Criteria:** Zero vague verbs (e.g. "make it better" is banned; must specify deterministic acceptance criteria).
3. **No Chain-of-Thought Fabrication:** Requests auditable rationale, assumptions, and output matrices rather than private internal reasoning traces.
4. **Tool-Specific Optimization:** Adapts prompt structure specifically for:
   - Autonomous Coding Agents (Template H: ReAct + Stop Conditions)
   - IDE Code Modifiers (Template G: File-Scope)
   - Structured Clinical LLM Information Extraction (Template F: Few-Shot Pattern Lock + Auditable Reasoning)

---

## 3. Active Application in RSNA Knee Challenge

In the RSNA Knee Abnormality Detection challenge, the single largest technical lever identified across the Top 10 teams is **Privileged Information Distillation**: extracting continuous probability vectors ($0.0 \to 1.0$) from the $4,349$ free-text clinical radiology reports to supervise deep 2.5D vision backbones.

Using `prompt-master`, we have formulated the production-grade clinical extraction prompt in [`docs/CLINICAL_REPORT_EXTRACTION_PROMPT.md`](file:///c:/Users/Cyrus/Downloads/rsna%20knee%20fip/docs/CLINICAL_REPORT_EXTRACTION_PROMPT.md).
