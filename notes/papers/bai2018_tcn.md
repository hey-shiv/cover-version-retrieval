# Bai, Kolter & Koltun (2018) — Generic convolutional vs recurrent sequence models

**Claim.** Simple temporal convolutional networks (dilated convolutions + residual blocks) match or beat LSTMs/GRUs on many sequence tasks, with a longer effective memory and parallel training.

**Use here.** A non-causal TCN encoder over `(12, 256)` HPCP (D-011). Dilations 1–32 with kernel 3 and two convolutions per block give a receptive field of 1 + 2·2·63 = 253 frames, so the top block sees nearly the whole (resampled) track before global mean+max pooling.

**What was not adopted.** Causal padding and weight normalisation (the task is offline, whole-track encoding; batch normalisation is used instead).

**Open question.** Whether a Transformer or a pretrained music model would help is left for later. The TCN is the justified cheap baseline.
