# approaches/llm_prompt_engineering — LLM Prompt Engineering & Tuning for Metasurface Design

> Originally a standalone repo named **MetaSurfaceDesignWithLLM**.
> Merged into the parent `metasurface_inverse_design` repo as one of the
> `approaches/` archive entries.

## What this is

An exploration of using **Large Language Models** to drive metasurface
design — comparing two complementary techniques:

- **`prompt_engineering/`** — hand-crafted prompts and a workflow that
  asks an LLM to propose / iterate on metasurface designs. Includes
  data construction (`construct_json.py`), drawing utilities
  (`draw.py`) and the workflow driver (`workflow.py`).
- **`prompt_tuning/`** — soft prompt / prompt tuning over a curated
  dataset of **650 metasurface samples**
  (`meta_surface_dataset_650.json`). Includes JSON/YAML construction
  and a tuning script (`tune.py`), plus a `hack.py` utility for
  experiments.

## Layout

```
llm_prompt_engineering/
├── prompt_engineering/
│   ├── workflow.py
│   ├── construct_json.py
│   └── draw.py
├── prompt_tuning/
│   ├── tune.py
│   ├── construct_json.py
│   ├── construct_yaml.py
│   ├── hack.py
│   ├── meta_surface_dataset_650.json     ← 650-sample dataset
│   └── prompt_config_650.yaml
└── LICENSE
```

## Status

Exploratory. Kept here as the **LLM-based** entry in the
`approaches/` archive — distinct in spirit from the
network-trained-from-scratch lines (CRNNAG, autoencoder, etc.) and
from the search-based lines (PPO, BO). The current main line uses
CRNNAG (see top-level README of `metasurface_inverse_design`).
