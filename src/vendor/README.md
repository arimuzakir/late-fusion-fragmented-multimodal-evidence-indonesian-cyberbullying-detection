# HEALNet source provenance

Source: https://github.com/konst-int-i/healnet
Commit: `90459b65a3d4a4ef9fd405671c457b5a1163cc7d`
File: `healnet/models/healnet.py`
Authors: Konstantin Hemker, Nikola Simidjievski, Mateja Jamnik.
Paper: HEALNet: Multimodal Fusion for Heterogeneous Biomedical Data, NeurIPS 2024, https://arxiv.org/abs/2311.09115 .
License: Apache 2.0; original LICENSE retained as HEALNET_LICENSE.

`healnet_official.py` is the unmodified source. `healnet_checked.py` applies exactly two reliability fixes:

1. Removes broad try/except/pass around cross-attention, so a shape or runtime error is visible rather than silently dropping the modality update.
2. Moves the missing-modality `continue` outside `if verbose`, so missing tensors are skipped even with verbose=False. Without this fix, removing silent exception suppression exposes attempts to normalise a None tensor.

No attention equation, temperature, activation, residual connection, or latent update is replaced. The caller adapts the inputs to V12 embeddings and the output to binary classification. Hyperparameters differ from the biomedical experiments and are explicitly saved. Do not label this an exact replication of biomedical experiments or of another cyberbullying paper.

The caller groups each mini-batch by modality-availability pattern and passes None for unavailable modalities. Tests check all four speech/video patterns and verify that changing masked feature values does not alter predictions. Existing author notices inside the source have been retained.
