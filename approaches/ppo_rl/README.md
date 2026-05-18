# approaches/ppo_rl — PPO Reinforcement Learning for Metasurface Inverse Design

> Originally a standalone repo named **PPO_for_metasurface_inverse_design**.
> Merged into the parent `metasurface_inverse_design` repo as one of the
> `approaches/` archive entries.

## What this is

A **Proximal Policy Optimization (PPO)** agent that learns to *edit a
metasurface pattern* (pixel-level structure) so that its simulated
optical response matches a target — typically a desired Jones-matrix
behaviour for **polarization control**.

The agent's environment wraps a Lumerical FDTD project
(`PPO_for_material_searching/jones_model_origin.fsp`); each step modifies
the pattern, runs (or surrogate-runs) the FDTD simulation, scores the
result against the target Jones matrix, and uses that as the reward
signal.

## Layout

```
PPO_for_material_searching/
├── algorithms/      ← PPO implementation
├── envs/            ← gym-style environment wrapping the FDTD model
├── train/           ← training entry points
├── ruuner/          ← (sic) runners / rollout workers
├── judge/           ← reward / scoring utilities
├── utils/
├── config.py
├── jones_model_origin.fsp   ← Lumerical FDTD project file
├── script.lsf       ← Lumerical script
└── yjh.md           ← author's running notes
```

## Status

This was an early **RL-based** approach to inverse design. It is kept
here for reference; the current main line of work in this repo uses a
**CRNNAG forward + inverse network** instead (see top-level README of
`metasurface_inverse_design`). For exceptional-point–targeted RL work,
see the separate **`MetaEP-RL`** repo.
