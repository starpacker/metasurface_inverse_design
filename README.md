# metasurface_inverse_design

> **The current main line** of metasurface inverse-design research in this account.
>
> This repo holds the latest **CRNNAG-based forward + inverse network**
> (top level) and, under `approaches/`, also archives **five earlier or
> alternative inverse-design approaches** that were previously kept as
> separate small repos. They are gathered here so the whole evolution
> of the project lives in one place.

---

## 🧭 The current main line (top-level files)

A two-stage system for designing metasurface unit cells:

1. **Forward model** — predicts the optical response from the geometry.
   Implemented under `forward_network/` and `core_model/`. The current best
   architecture is **CRNNAG (Convolutional + Recurrent + AutoRegressive)
   with attention** + teacher forcing — found to clearly beat plain CNN on
   complex patterns (e.g. L-shape + rectangle), while plain CNN remains
   competitive on the simplest patterns when data is limited.
2. **Inverse model** — given a target spectrum, generates the geometry
   parameters that produce it. Trained on data synthesized by the forward
   model and on a set of FDTD-validated samples.

`FDTDModel.py` wraps the Lumerical FDTD solver so the loop
*generate → simulate → score → re-train* can be automated. Datasets
live under `dataset/`, training utilities under `data_process/`, and
`compare_data.ipynb` is the analysis notebook.

### Key empirical findings (logged during development)

- **Activation function matters a lot.** Switching ReLU ↔ ELU produced very
  large differences in final loss; this is now controlled explicitly.
- **Teacher-forcing ratio ~0.3** is the sweet spot for CRNNAG. Ratios near
  0 or 0.8 both underperform.
- **Early stopping** stabilises CRNNAG but can hurt CNN when the dataset
  is small (CNN stops too early).
- **Data quality > raw data quantity** — 2000 carefully validated samples
  beat 8000 noisy ones in some experiments. After filtering ~24000 useful
  samples out of 30954, both CNN and CRNNAG cleared their previous best CE
  losses; CRNNAG eventually pulled ahead on complex patterns.
- Best test loss to date: **CRNNAG + attention, teacher-forcing 0.3**
  (reproduced).
- The trained inverse model has been used to generate **exceptional-point**
  metasurface designs — overall structure recovers well, fine details still
  need work.

The full chronological research log used to live in this README; it has been
moved to `RESEARCH_LOG.md` so the README itself can stay readable.

---

## 📁 Repository layout

```
metasurface_inverse_design/
├── core_model/          ← CRNNAG inverse-network code
├── forward_network/     ← forward (geometry → spectrum) network code
├── data_process/        ← dataset construction and filtering
├── dataset/             ← training data
├── FDTDModel.py         ← Lumerical FDTD wrapper
├── compare_data.ipynb   ← analysis notebook
├── RESEARCH_LOG.md      ← chronological research notes (moved out of README)
└── approaches/          ← five earlier / alternative inverse-design methods
    ├── ppo_rl/                  ← PPO reinforcement learning over patterns
    ├── v_shape_vortex_beam/     ← V-shape MIM for arbitrary vortex beams
    ├── bo_polarization/         ← Bayesian-optimized polarization waveplates
    ├── autoencoder_unidir/      ← unsupervised autoencoder for unidirectional transmission
    └── llm_prompt_engineering/  ← LLM prompt engineering / prompt tuning for metasurface design
```

---

## 🗂️ The `approaches/` archive

Each subfolder under `approaches/` was previously a standalone GitHub
repo. They are kept here as a record of the different inverse-design
methods that were tried before the current CRNNAG main line, and as
reference implementations that future work can compare against.

| Folder | Original repo | Method | Target problem |
|---|---|---|---|
| `approaches/ppo_rl/` | PPO_for_metasurface_inverse_design | **PPO reinforcement learning** — agent edits a pattern to optimise a Jones-matrix–based objective; uses the original `jones_model_origin.fsp` FDTD project | Polarization control via pixelated metasurfaces |
| `approaches/v_shape_vortex_beam/` | V-shape-material-design | **Forward + inverse DNNs** for V-shape MIM unit cells (LSTM and basic optim variants) | Arbitrary multifunctional **vortex beam** generation; bifunctional reflective **metalens** |
| `approaches/bo_polarization/` | BO_meta_surface_design | **Bayesian Optimization + DNN** (BO-NET); after BO saturates, `frac_optimization.py` performs random pixel-flip refinement | Arbitrary polarization-control waveplates |
| `approaches/autoencoder_unidir/` | auto-encoder-meta-surface-design | **Unsupervised autoencoder** (CNN encoder/decoder) jointly reconstructing structure + near-field, expanded by BO; also includes a GA (DEAP) baseline | **Unidirectional transmission** nanostructures |
| `approaches/llm_prompt_engineering/` | MetaSurfaceDesignWithLLM | **LLM prompt engineering** and **prompt tuning** (~650-sample dataset) for metasurface design | Direct LLM-driven design exploration |

> **Why merged?** All five repos targeted the same overall problem
> (metasurface inverse design) but each tried a different paradigm:
> RL, plain DNN, BO-augmented DNN, unsupervised autoencoder, LLM. Keeping
> them in one place makes it possible to compare them without juggling six
> repos. Each subfolder retains its own original `README.md` (where one
> existed) so its individual context is preserved.

---

## 🔬 Related, kept separate

- **`MetaEP-RL`** — the active RL line for **exceptional-point** metasurface
  design — is intentionally kept as its own repo, since it has diverged
  enough to warrant independent iteration.

---

## 📜 License

See `LICENSE`.
