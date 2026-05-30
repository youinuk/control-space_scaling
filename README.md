# Control-space scaling for transport-induced phase networks

This repository contains the code and source data used to reproduce the figures in the manuscript:

**Control-space scaling of transport-induced correlated phases in parallel spin-shuttling arrays**

## Repository structure

```text
control-space-scaling/
  README.md
  LICENSE
  environment.yml
  requirements.txt
  scripts/
    simulation_stage1_final.py
    simulation_stage2_final.py
    simulation_stage2_robustness.py
  data/
    source_data_stage1_*.csv
    source_data_stage2_*.csv
    source_data_stage2_robustness_*.csv
  figures/
    fig_summary.pdf
    fig_summary_full6_appendix.pdf
    fig_2_summary.pdf
    fig_3_robustness_summary.pdf
    fig_2A_inference_reconstruction.pdf
    fig_2B_trajectory_optimization.pdf
    fig_4_extended_robustness.pdf
  manuscript/
    main.tex
    refs.bib
```

## Quick start

Create the environment:

```bash
conda env create -f environment.yml
conda activate control-space-scaling
```

or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the simulations:

```bash
python scripts/simulation_stage1_final.py
python scripts/simulation_stage2_final.py
python scripts/simulation_stage2_robustness.py
```

By default, scripts write figures and source-data CSV files to `/content/outputs` unless the `OUTDIR` environment variable is set:

```bash
OUTDIR=outputs python scripts/simulation_stage1_final.py
```

## Figure mapping

| Manuscript figure | Output file | Script |
|---|---|---|
| Figure 2 | `fig_summary.pdf` | `simulation_stage1_final.py` |
| Figure 3 | `fig_2_summary.pdf` | `simulation_stage2_final.py` |
| Figure 4 | `fig_3_robustness_summary.pdf` | `simulation_stage2_robustness.py` |
| Appendix Fig. B.1 | `fig_2B_trajectory_optimization.pdf` | `simulation_stage2_final.py` |
| Appendix Fig. B.2 | `fig_2A_inference_reconstruction.pdf` | `simulation_stage2_final.py` |
| Appendix Fig. C.1 | `fig_4_extended_robustness.pdf` | `simulation_stage2_robustness.py` |
| Appendix Stage-1 diagnostics | `fig_summary_full6_appendix.pdf` | `simulation_stage1_final.py` |

## Source data

The `data/` directory contains CSV files used to generate the plotted curves and scalar summaries. These files are intended as figure source data for journal review and archival release.

## Reproducibility notes

The scripts use fixed NumPy random seeds. Stage 1 uses 30 Monte Carlo trials and 1000 bootstrap resamples for the power-law confidence interval. Stage 2 scaling and robustness scans use 15 Monte Carlo trials, while the fixed-K inference overhead scan uses 30 trials.

## License

Code is released under the MIT License. Figure source data are released under CC BY 4.0.
