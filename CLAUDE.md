# Truss Optimization

Structural truss weight minimization using FEM-based sensitivity analysis with OC and MMA solvers.

## Environment

Python virtual environment is at `.venv/`. Always use it:

```bash
source .venv/bin/activate
python sensitivity_analysis.py
```

Or prefix commands: `.venv/bin/python sensitivity_analysis.py`

## Project Structure

- `sensitivity_analysis.py` — main script; runs optimization and generates GIFs in `results/`
- `utils.py` — FEM helpers (stiffness assembly, displacement solver, sensitivity computation)
- `initial_structure/` — default truss dataset (nodes, members, loads, support CSVs)
- `initial_structure_bridge/` — bridge variant dataset
- `results/oc/`, `results/mma/` — per-algorithm output plots and GIFs

## Dependencies

```bash
pip install -r requirements.txt  # numpy, matplotlib, pandas, imageio[pillow]
```

## Branches

- `main` — stable
- `refactor/mma` — MMA solver integration (active)
