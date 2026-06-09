# Control-Space Scaling of Correlated Phase Networks

This repository contains the manuscript source, simulation scripts, generated figures, and source-data CSV files for the manuscript:

**Control-Space Scaling of Correlated Phase Networks in Parallel Spin Shuttling Architectures**

## Repository layout

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
    fig_1.pdf
    fig_operational_validation_lifecycle.pdf
    fig_summary.pdf
    fig_2_summary.pdf
    fig_3_robustness_summary.pdf
    fig_2A_inference_reconstruction.pdf
    fig_2B_trajectory_optimization.pdf
    fig_4_extended_robustness.pdf
    fig_summary_full6_appendix.pdf

    all_outputs/
      stage1/
      stage2/
      stage2_robustness/

  manuscript/
    main.tex
    ref.bib
```

The manuscript-facing figure files are kept directly in `figures/`. Additional simulation outputs, including both PDF and PNG versions of intermediate diagnostic figures, are stored under `figures/all_outputs/`.

## Environment

Recommended Python version: **Python 3.10**.

Create the environment with conda:

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

## Reproducing the figures

Run the three simulation scripts:

```bash
python scripts/simulation_stage1_final.py
python scripts/simulation_stage2_final.py
python scripts/simulation_stage2_robustness.py
```

By default, the scripts write generated figures and CSV source data to `/content/outputs`. To write outputs to a local directory:

```bash
OUTDIR=outputs python scripts/simulation_stage1_final.py
OUTDIR=outputs python scripts/simulation_stage2_final.py
OUTDIR=outputs python scripts/simulation_stage2_robustness.py
```

After generation, copy the manuscript-facing figure PDFs into `figures/` and the full PDF/PNG output set into the appropriate `figures/all_outputs/` subfolder.

## Manuscript figure mapping

| Manuscript item | Figure file | Source |
|---|---|---|
| Figure 1 | `figures/fig_1.pdf` | schematic |
| Figure 2 | `figures/fig_summary.pdf` | `simulation_stage1_final.py` |
| Figure 3 | `figures/fig_operational_validation_lifecycle.pdf` | operational validation schematic |
| Figure 4 | `figures/fig_2_summary.pdf` | `simulation_stage2_final.py` |
| Figure 5 | `figures/fig_3_robustness_summary.pdf` | `simulation_stage2_robustness.py` |
| Appendix Fig. B.1 | `figures/fig_summary_full6_appendix.pdf` | `simulation_stage1_final.py` |
| Appendix Fig. C.1 | `figures/fig_2B_trajectory_optimization.pdf` | `simulation_stage2_final.py` |
| Appendix Fig. C.2 | `figures/fig_2A_inference_reconstruction.pdf` | `simulation_stage2_final.py` |
| Appendix Fig. D.1 | `figures/fig_4_extended_robustness.pdf` | `simulation_stage2_robustness.py` |

The file names are intentionally stage-oriented rather than manuscript-number-oriented. This preserves traceability from each plotted result to the script that generated it, while the table above records the manuscript numbering.

## Source-data mapping

| CSV file | Associated figure(s) | Description |
|---|---|---|
| `source_data_stage1_rank_threshold.csv` | Fig. 2(a) | threshold-dependent effective rank |
| `source_data_stage1_speed.csv` | Fig. 2(b), Appendix B.1 | static/dynamic rank and speed sweep |
| `source_data_stage1_FT_pairs.csv` | Fig. 2(c) | FT-relevant pair counts |
| `source_data_stage1_powerlaw_fit.csv` | Fig. 2(c) | bootstrap fit and confidence interval |
| `source_data_stage1_calibration_time.csv` | Fig. 2(d) | assumed sequential timing model |
| `source_data_stage1_rank_disorder.csv` | Appendix B.1 | positional-disorder diagnostic |
| `source_data_stage2_inference_overhead.csv` | Fig. 4(a), Appendix C.2 | overhead scan for multiplexed inference |
| `source_data_stage2_reconstruction_scatter.csv` | Appendix C.2 | true vs reconstructed matrix elements |
| `source_data_stage2_trajectory_history.csv` | Appendix C.1 | trajectory-optimization history |
| `source_data_stage2_trajectory_profile.csv` | Appendix C.1 | nominal vs optimized phase profile |
| `source_data_stage2_scaling.csv` | Fig. 4(b,c) | scaling scan and calibration-time comparison |
| `source_data_stage2_summary_scalars.csv` | Fig. 4, Appendix C.1 | scalar diagnostics |
| `source_data_stage2_robustness_subarray_overhead.csv` | Fig. 5(a), Appendix D.1 | sub-array overhead |
| `source_data_stage2_robustness_subarray_F_at_3K.csv` | Appendix D.1 | fixed-budget sub-array fidelity |
| `source_data_stage2_robustness_prior_mismatch.csv` | Fig. 5(b), Appendix D.1 | prior-mismatch scan |
| `source_data_stage2_robustness_selfphase_Kscan.csv` | Fig. 5(c), Appendix D.1 | self-phase noise and no-compensation baseline |

## Reproducibility notes

The scripts use fixed NumPy random seeds. Stage 1 uses 30 Monte Carlo trials and 1000 bootstrap resamples for the power-law confidence interval. Stage 2 scaling and robustness scans use 15 Monte Carlo trials, while the fixed-\(K\) inference overhead scan uses 30 trials.

The numerical results are intended as reference scaling and robustness benchmarks under the normalized simulation model, not as hardware-calibrated predictions for a specific device.

## Archival release

The public repository for this submission is:

GitHub: https://github.com/youinuk/control-space_scaling

Zenodo DOI: https://doi.org/10.5281/zenodo.20603260


## License

Recommended:
- Code: MIT License
- Figure source data and generated data tables: CC BY 4.0
