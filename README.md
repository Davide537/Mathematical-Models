# Mathematical Models

A collection of mathematical and computational modeling projects, spanning large-scale neural network simulations of the macaque cortex and agent-based / PDE-based models of collective behavior.

The repository is organized into two independent sub-projects:

| Folder | Description |
|---|---|
| [`WM Model`](./WM%20Model) | Large-scale spiking-rate model of working memory in macaque cortex, extended to study the effects of aging (Master's thesis work). |
| [`Applications of PDEs`](./Applications%20of%20PDEs) | Lattice and off-lattice agent-based models (population dynamics, flocking/"dancers", avoidance behavior) illustrating PDE-inspired collective-motion models. |

---

## WM Model — WM Large Scale Model

Large-scale simulation of working memory dynamics across the macaque cortex, based on the multi-area model of **Mejías & Wang (2022)**, extended to incorporate biological aging mechanisms.

**Highlights**
- 30-area macaque cortical parcellation (occipital, parietal, temporal, frontal, prefrontal), using the [Scalable Brain Atlas](https://scalablebrainatlas.incf.org/macaque/MERetal14_on_F99) (MERetal14 on F99).
- Models three age groups (**Young**, **Middle Aged**, **Aged**) via biologically-motivated aging mechanisms:
  - Dendritic spine density loss (from electrophysiology data).
  - Shifts in the f-I (frequency-current) transfer function.
- Parameter sweeps over synaptic gain (`J_max`) and global coupling (`G`), with results stored in HDF5 and explored with heatmaps and radar/spider plots of cortical activity per lobe.
- 3D rendered visualization of the macaque cortical surface (`brain3d.m`, MATLAB, using the SBA atlas), callable from Python via `matlab.engine`.

**Contents**
- `Aging model.ipynb` — core model build (`BuildModel`), simulation loop, cortical map / heatmap visualizations, aging & inactivation experiments.
- `Spine aging.ipynb` — extraction and extrapolation of dendritic spine count data across ages.
- `FR aging.ipynb` — firing-rate / f-I curve fitting across age groups.
- `brain3d.m` — MATLAB function for 3D rendering of the macaque cortex with area-level activity overlays.
- `data/` — electrophysiology and spine-count datasets (`.xlsx`), and SBA atlas / connectome data (`.mat`, `.csv`) used for the 3D brain rendering and area parcellation.
- `Fig/` — generated figures (heatmaps, cortical maps, 3D brain renders, distractor and FR/spine analysis plots).

## Applications of PDEs

Notebook exploring PDE-inspired agent-based models of collective behavior:
- **Population model** — base population dynamics.
- **Dancers class** — shared agent representation used across the lattice/off-lattice models.
- **On-lattice model** — collective motion simulated on a discrete lattice, with optional frame-by-frame plotting and video export.
- **Off-lattice model** — continuous-space analogue of the lattice model.
- **Avoidance model** — agents with obstacle/collision-avoidance behavior.

Each model exposes `Showplots` and `Saveframes` options to visualize the simulation live or export frames/video of its time evolution. A rendered report is available as `APDE.pdf`.

---


## References

- Mejías, J. F., & Wang, X.-J. (2022). Mechanisms of distributed working memory in a large-scale network of macaque neocortex.
- Scalable Brain Atlas — macaque parcellation: https://scalablebrainatlas.incf.org/macaque/MERetal14_on_F99
