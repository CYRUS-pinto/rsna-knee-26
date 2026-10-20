# Lessons for Future Competitions (portable, competition-agnostic)

Distilled from RSNA Knee 2026 (34 submissions, rank 583 → 688 while stuck at 0.958). Each lesson cost real score or time.

1. **Check the score distribution first.** If ~25 % of teams sit at one score, that's the public-notebook ceiling. Re-blending public weights won't beat it; only new signal will.
2. **Decide the prize track early.** A <500 s efficiency target capped our main-LB AUC for days. Use separate candidates for separate prizes.
3. **Build OOF infrastructure on day 1.** Without a full OOF matrix, blend weights become guesses and small gold sets get over-tuned.
4. **Rank-blend heterogeneous models.** Averaging raw probabilities, or power sharpening, cost us 0.015 (V42).
5. **No asserts in inference code.** A test that passes on 3 public studies can crash on the hidden ~1,300.
6. **Interactive runtime is not hidden-test runtime.** Multiply per-study time by the hidden N.
7. **Run Colab training as isolated scripts with per-epoch checkpoints pushed off-box.** Use gradient checkpointing for 384px transformers.
8. **Preflight every generated script locally** (`py_compile` plus a 1-batch CPU run) before handing it over.
9. **One canonical, auto-generated state file.** Hand-written handoffs go stale within hours and spread wrong numbers.
10. **Every submission is an experiment:** write down one change, the expected Δ and the evidence before you submit.
11. **Clinical priors look great on tiny gold sets and hurt on the LB.** Trust only large-sample validation.
12. **Keep a safety-floor submission selected, and freeze code 12 h before the deadline.**
