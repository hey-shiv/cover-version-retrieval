# Yesiler et al. (2019) — Da-TACOS

**Question.** How can CSI systems be compared when data and implementations are fragmented and copyrighted audio cannot be shared?

**Contribution.** Pre-extracted features and metadata for two subsets: *Benchmark* (15,000 tracks: 1,000 works x 13 versions + 2,000 noise tracks) and *Cover Analysis* (5,000 two-version works). Also the `acoss` feature and benchmarking framework.

**Protocol extracted for this project.**
- Relevance is shared WID. The query itself is excluded. Noise tracks stay in the candidate pool.
- Metrics: MAP, MRR, mean rank of the first correct result (MR1), and top-k counts.
- The Benchmark is for benchmarking only. This project never tunes on it (D-001).

**Assumptions that matter here.** Features were extracted from MP3s with fixed parameters, and the work labels come from SecondHandSongs. Both are taken as ground truth.

**Criticism.** A catalogue of Western popular music assembled around 2019; no audio, so neither augmentation at the signal level nor re-extraction is possible. Two-version cliques limit supervised training.
