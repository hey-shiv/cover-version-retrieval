| system | n_queries | MAP | MRR | Recall@1 | Recall@10 | Recall@100 | Hit@1 | Hit@10 | Hit@100 | mean_first_rank | median_first_rank | ms_per_pair |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| profile_only | 20 | 0.266 | 0.266 | 0.150 | 0.550 | 0.850 | 0.150 | 0.550 | 0.850 | 24.850 | 8.500 |  |
| classical_none | 20 | 0.303 | 0.303 | 0.200 | 0.450 | 0.900 | 0.200 | 0.450 | 0.900 | 33.450 | 13.000 | 0.086 |
| classical_profile_cosine | 20 | 0.248 | 0.248 | 0.150 | 0.450 | 0.950 | 0.150 | 0.450 | 0.950 | 31.250 | 13.000 | 0.084 |
| classical_exhaustive_dtw | 20 | 0.248 | 0.248 | 0.150 | 0.450 | 0.950 | 0.150 | 0.450 | 0.950 | 30.100 | 13.000 | 0.993 |

Resolution sensitivity (analysis only; default stays at the configured n_frames):

| protocol | n_frames | n_queries | MAP | Hit@10 | mean_first_rank | ms_per_pair |
|---|---|---|---|---|---|---|
| calibration | 48 | 20 | 0.206 | 0.400 | 74.400 | 0.020 |
| dev | 48 | 20 | 0.285 | 0.450 | 26.250 | 0.034 |
| calibration | 96 | 20 | 0.206 | 0.450 | 72.400 | 0.054 |
| dev | 96 | 20 | 0.248 | 0.450 | 31.250 | 0.083 |
| calibration | 192 | 20 | 0.290 | 0.400 | 68.900 | 0.421 |
| dev | 192 | 20 | 0.293 | 0.450 | 33.000 | 0.234 |
| calibration | 384 | 20 | 0.304 | 0.400 | 63.750 | 0.953 |
| dev | 384 | 20 | 0.384 | 0.450 | 34.150 | 2.099 |
