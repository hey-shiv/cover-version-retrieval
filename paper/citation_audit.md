# Citation audit

## How verification was possible here
The environment's proxy blocked arXiv, export.arxiv, dblp, Crossref, Semantic Scholar, JMLR, Zenodo, ismir.net, the ISMIR archives, music-ir.org (MIREX) and web.archive.org. Checked 2026-09-25 by a delegated literature agent; notes are in the session scratchpad and are summarised here.

Three levels, recorded in each `references.bib` entry's `verification` field:
- **C.** Title, DOI, year, venue and surnames match a Crossref record stored in `notes/literature/sources/*.json`. That record was saved in an earlier session that could reach Crossref.
- **G.** Read on the authors' GitHub README (MTG/da-tacos, MTG/discogs-vi-dataset, raraz15/Discogs-VINet, furkanyesiler/re-move, Liu-Feng-deeplearning/CoverHunter).
- **S.** Seen only in web-search results. **Every S entry must be opened and checked before submission.**

## Entries

| key | level | notes |
|---|---|---|
| datacos | C, G | pages 327–334 from the GitHub citation |
| serra2008 | C | |
| serra2009qmax | S | DOI seen in the IOPscience URL |
| gomez2006 | S | PhD thesis, UPF |
| sakoe1978 | S | DOI from a search summary |
| muller2015 | S | 1st edition DOI; the 2021 2nd edition exists |
| move2020 | C | |
| remove2020 | G | **ISMIR 2020, not TASLP 2021** as the paper brief said |
| bytecover | C | arXiv 2010.14022 |
| bytecover2 | S | no DOI seen |
| bytecover3 | S | arXiv 2303.11692; two-stage ANN → re-ranking with local features (from search summary) |
| coverhunter | G, S | ICME 2023; arXiv 2306.09025 |
| discogsvi | G, S | arXiv 2410.17400; weights public, per-query rankings not released (G) |
| clews | S | ICML 2025, arXiv 2502.16936 |
| divers | S | ISMIR 2026, arXiv 2608.04543. More than 1.1M versions; splits compatible with Da-TACOS. Evaluated baselines not confirmed. |
| hachmeier2025 | S | iConference 2025, arXiv 2501.01333 |
| araz2026unified | S | arXiv 2608.19919 |
| livi2026 | S | ECIR 2026, DOI 10.1007/978-3-032-21289-4_4. The MaxSim re-ranking may be only in the extended arXiv version (2601.11262). Second author's given name varies between sources, so only the initial "B." is used. |
| colbert | S | |
| aucouturier2008 | C | DOI 10.1016/j.patcog.2007.04.012 (Crossref). A search summary gave …04.025, which was **rejected** in favour of the stored record. |
| radovanovic2010, schnitzer2012 | S | |
| seo2022, li2018, hu2019 | C | |
| csls | C | arXiv 1710.04087 |
| qbnorm | C | page range not in the stored record, so omitted |
| wang2011cascade, culpepper2016 | S | DOIs seen in ACM DL URLs |
| jacob2024drowning | S | arXiv 2411.11767 |
| bertinmahieux2012 | S | |
| bai2018tcn, khosla2020supcon | S | also summarised in `notes/papers/` |
| efron1993, field2007 | S | |

## Numerical claims about other work
| claim in the paper | status | action |
|---|---|---|
| Qmax 0.333 MAP on HPCP (Da-TACOS) | recorded in `notes/literature/novelty_assessment.md` as Yesiler et al. 2019, Table 2. No stored excerpt, and it could not be re-read here. | **Blocking before submission:** read Table 2. |
| Discogs-VI has 1.9M versions | G | ok |
| DiVers has more than 1.1M versions | S | re-check |
| MIREX 2024 ByteCover2 mAP 0.877; MIREX 2025 MuCrossCover3 0.958 with a data-issue caveat (from the paper brief) | **U: not found** | **Not used in the paper.** Only "MIREX uses private collections, not comparable" is stated. |
| MOVE 0.507, Re-MOVE 0.534, ByteCover 0.714 on Da-TACOS (repository notes) | not re-verified | **Not used in the paper.** They appear only on the website's placement figure, where the source is cited. |

## Claims about the paper brief
The brief supplied these items, which could not be confirmed and were therefore **not** written into the paper:
- MIREX 2024 and 2025 numbers;
- the MIREX 2025 data-issue wording;
- "validation still drifting upward at epoch 150" (contradicted by the history file);
- "119 tests" (the repository collects 142).
