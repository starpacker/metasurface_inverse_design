# approaches/autoencoder_unidir — Unsupervised Autoencoder for Unidirectional Transmission

> Originally a standalone repo named **auto-encoder-meta-surface-design**.
> Merged into the parent `metasurface_inverse_design` repo as one of the
> `approaches/` archive entries. The original repo's README is preserved
> as `_original_README.md`.

## What this is

An **unsupervised** inverse-design approach for nanostructures with
**unidirectional transmission**. A CNN-based **autoencoder** jointly
encodes (and reconstructs) the structure parameters *and* the near-field
distribution into a shared latent space. By traversing that latent
space, novel structure–field pairs can be generated **without needing
labelled training data**. Bayesian Optimization is used to expand the
explored region of the latent space.

For comparison, a classic **Genetic Algorithm** baseline (DEAP) is also
included.

Corresponds to the paper *“Inverse Design of Unidirectional Transmission
Nanostructures Based on Unsupervised Machine Learning”*.

## Layout

```
autoencoder_unidir/
├── autoencoder.py     ← AE training (structure + near-field encoder/decoder)
├── VAE_CNN.py         ← VAE variant
├── VAE_test.py        ← VAE evaluation / latent traversal
├── dense_CNN.py       ← dense CNN building blocks
├── GA_deap_one_max.py ← Genetic Algorithm baseline (single-objective)
└── GA_deap_fun_max.py ← Genetic Algorithm baseline (function-target)
```

## Why this approach

The headline feature is **no labels required**. Generating labelled
metasurface data is expensive (each label is an FDTD simulation), so an
unsupervised method that learns directly from a pool of unlabelled
geometry+field samples sidesteps that bottleneck.

## Status

Earlier work. Kept for reference. The current main line uses a
**supervised** CRNNAG forward+inverse network trained on FDTD-validated
data — see the top-level README of `metasurface_inverse_design`.
