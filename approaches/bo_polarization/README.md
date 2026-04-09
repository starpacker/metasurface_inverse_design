# approaches/bo_polarization — Bayesian-Optimized DNN for Polarization Control

> Originally a standalone repo named **BO_meta_surface_design**.
> Merged into the parent `metasurface_inverse_design` repo as one of the
> `approaches/` archive entries. The original repo's README is preserved
> as `_original_README.md`.

## What this is

A **BO-NET**: a deep neural network surrogate that maps nanostructure
geometry to its optical response, paired with **Bayesian Optimization**
that searches over geometries to hit a target response. Once BO
saturates, `frac_optimization.py` performs a **random pixel-flip
refinement** pass to squeeze out a bit more performance.

The target application is **arbitrary polarization-control waveplates**
— precisely controlling the polarization state of incident light via
the metasurface pattern.

Corresponds to the paper *“Self-design of arbitrary polarization-control
waveplates via deep neural networks”*.

## Layout

```
bo_polarization/
├── BO_main.py                     ← main entry: BO + DNN surrogate
├── frac_optimization.py           ← random pixel-flip refinement (post-BO)
├── FDTD_api.py                    ← Lumerical FDTD wrapper
├── arbitrary polarization-control ← project subfolder for the polarization task
└── LICENSE
```

## How it works (one paragraph)

BO is sample-efficient but slow per-sample, and random pixel-flip is
fast per-sample but inefficient at exploring. The combination
**BO first, then frac-optimization** keeps BO doing the cheap, smart
exploration while the random refinement only kicks in once BO is near
its plateau — which is when small local changes are most likely to
help.

## Status

Earlier work. Kept for reference. The current main line uses CRNNAG
(see top-level README of `metasurface_inverse_design`).
