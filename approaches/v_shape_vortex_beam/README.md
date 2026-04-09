# approaches/v_shape_vortex_beam — V-shape MIM for Vortex Beams & Bifunctional Metalens

> Originally a standalone repo named **V-shape-material-design**.
> Merged into the parent `metasurface_inverse_design` repo as one of the
> `approaches/` archive entries. The original repo's README is preserved
> as `_original_README.md`.

## What this is

Forward + inverse **deep neural networks** for designing **V-shape MIM
(metal–insulator–metal) metasurface unit cells**, applied to two
publications:

1. **Arbitrary Multifunctional Vortex Beam Designed by Deep Neural Network** —
   given a desired phase response, the inverse network outputs the
   meta-atom hyperparameters that produce the corresponding vortex beam.
   Top-level scripts (`train_for.py`, `train_back.py`, `integrate.py`,
   `datacode.py`) belong to this work.
2. **Polarization Multiplexing Bifunctional Metalens Designed by Deep Neural
   Networks** — design of a reflective metalens with two functions selected
   by polarization. Code lives under `metalens/`.

## Layout

```
v_shape_vortex_beam/
├── train_for.py     ← forward network training
├── train_back.py    ← inverse network training
├── integrate.py     ← end-to-end integration
├── datacode.py      ← dataset construction
├── basic_optim/     ← simple baseline optimisers
├── LSTM/            ← LSTM variant (for sequential phase response)
├── metalens/        ← bifunctional reflective metalens code
└── QR_code/         ← QR-code-style pattern experiments
```

## Status

Earlier work on the project. Kept as reference for the V-shape MIM
geometry and the dual forward/inverse network training pattern. The
current main line uses CRNNAG (see top-level README of
`metasurface_inverse_design`).
