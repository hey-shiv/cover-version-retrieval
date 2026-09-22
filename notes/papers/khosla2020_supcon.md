# Khosla et al. (2020) — Supervised contrastive learning

**Idea.** Generalise the InfoNCE/NT-Xent loss to labelled data. For each anchor, every other sample with the same label is a positive and all other samples are negatives. The loss averages log-softmax similarities over positives (the "L_out" form, Eq. 2), with normalised embeddings and a temperature τ.

**Why it is used here.** With batches of P works x 2 recordings, each anchor gets one positive and 2(P−1) negatives, which uses far more comparisons than one triplet per anchor (D-011).

**Implementation notes.** Written explicitly in `models/losses.py`: the self-similarity is masked with −inf before the log-sum-exp, and a unit test checks the loss against a hand-computed formula. Temperature is 0.1.

**Caveat.** The paper's evidence comes from image classification. Transferring the loss to open-set cover retrieval is an assumption, and it is tested here only by retrieval MAP on disjoint works.
