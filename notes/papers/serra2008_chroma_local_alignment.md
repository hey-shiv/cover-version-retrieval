# Serrà, Gómez, Herrera & Serra (2008) — Chroma binary similarity and local alignment for CSI

**Question.** How can two recordings of the same composition be matched despite changes of key, tempo, structure and instrumentation?

**Core idea.** Represent each recording as a sequence of chroma/HPCP vectors. Remove key differences by transposing one sequence to the other's "optimal transposition index" (the rotation that maximises the similarity of the global pitch-class profiles). Build a cross-similarity matrix, then run a *local* alignment that rewards long diagonal runs of similar frames, so partial and reordered matches still count.

**What this project implements.**
- the transposition step: 12 rotations, chosen from the global-profile cosine (D-004);
- a cosine-distance cross-similarity matrix;
- local alignment as **subsequence DTW** with slope constraints (D-006), not the paper's binary-similarity Qmax recurrence.

**What it deliberately does not copy.** The acoss `Serra09` implementation (AGPL-3.0) was not consulted for code. The recurrence here is the textbook DTW dynamic program from FMP §3.2.

**What key invariance does not solve.** A correct global rotation does not fix modulations within a song, differences in harmonic rhythm, or re-harmonised covers. Two different songs with a common progression (I–V–vi–IV) will still align well after rotation, which is a source of false positives.

**Criticism / uncertainty.** Subsequence DTW forces the *whole* query to align. Serrà's local alignment can instead match a shared section while ignoring the rest, which should be more robust to structural edits. Comparing the two is an obvious next experiment.
