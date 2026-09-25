# Paper positioning

## Decision
**B/C: an empirical retrieval-analysis paper about a reproducible two-stage CSI system.** It is not a method paper (A). The brief's hypothesis B ("recall-aware coarse-to-fine cover retrieval") is kept as the framing, with "recall-aware" meaning *analysed through candidate recall*, not a new recall-aware algorithm.

Working title: **Where Two-Stage Cover Song Retrieval Loses Its Covers: Candidate Recall, Hubness and Alignment Cost on Da-TACOS**

## Why not the other types
- **A (method).** Every component is prior art (`novelty_audit.md`). The one intervention that works, global + window fusion, is late interaction.
- **D (reproducibility / evaluation methodology only).** The development-vs-benchmark analysis is one strong section, but the K sweep and decomposition are the bigger results.
- **E (not a paper).** The evidence is large-scale, locked and paired, and it answers questions the literature leaves open for CSI. It is publishable as an analysis, with the absolute-accuracy caveat stated up front.

## Central claim (every clause checked against `result_verification.md`)
> In coarse-to-fine cover retrieval, the shortlist is a measurable bottleneck: alignment can only reorder covers the coarse stage admits.
>
> - On Da-TACOS, improvements to the coarse stage (training data, training length, window fusion) account for most of the observed gains, while alignment resolution and hubness correction improve what the reranker does with them.
> - At the deployed K = 30, exclusion is the dominant failure. Enlarging the shortlist raises accuracy steadily but moves the dominant failure from exclusion to misranking around K = 100–200.
> - A 20-query development protocol could not observe any of this.

The brief's sentence "shortlist recall remains the dominant bottleneck" is **qualified to K = 30**. At K ≥ 200 it is false (B1).

The brief's sentence "narrow the gap to exhaustive alignment while retaining much lower per-query alignment cost" is **supported**:
- the ratio fell from 4.5× to 1.11×, at 30 instead of 14,999 alignments per query;
- the reference is a 96-frame exhaustive system, because the 384-frame one was never run.

## Research questions used
- **RQ1.** How does shortlist size K control accuracy and cost? (A1)
- **RQ2.** How much does Stage-1 coverage bound the reranker, and where does that bound stop being the main limit? (X1, X2, B1)
- **RQ3.** What moves the coarse stage? (training scale and length: runs 1–4, F1; fusion: A2; adaptive K: E1, negative)
- **RQ4.** How much error is due to hubness, and how does it depend on K? (X5, B1, P1)
- **RQ5.** What does alignment resolution buy, and what is established only on development? (DEV, X5, H1)
- **RQ6.** Can a small development protocol substitute for benchmark evaluation? (P2, X6)

The brief's candidate "why do some queries need much larger K" is **not** a separate RQ. D1 shows difficulty is predictable (AUROC 0.76) but gives no causal account, so it appears only as the motivation for E1.

## Audience and venue
An MIR audience: ISMIR main track or TISMIR. The full draft (`main.tex`) is longer than an ISMIR paper. A 6-page cut would keep sections 1–3, 5, 7 (main results, K sweep, failure shift), 9 and 11, and move the rest to the supplement.
