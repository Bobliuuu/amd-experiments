# Report Utilities (`report`)

This directory contains the main report markdown files and runnable figure-generation scripts.

## Run Location

```bash
cd /root/workspace/amd-experiments
```

## Regenerate Figures

```bash
# Current report figure set
python3 report/generate_figures_v2.py

# Earlier report figure set
python3 report/generate_figures.py
```

## Main Documents

- `final_report_v2.md`: current comprehensive report (includes **§14** — repository engineering closure, Figs 30–31)
- `final_report.md`: earlier report version (includes **§14.5** — same closure summary)
- `paper.md`: paper-length narrative (includes **§5.13**)
- **Decode closure (repo vs deployment):** [`../docs/repo_decode_bottleneck_closure.md`](../docs/repo_decode_bottleneck_closure.md)

`generate_figures_v2.py` also writes **`figures_v2/fig30_decode_whole_step_rocprof_buckets.png`** (rocprof bucket shares from `results/decode_whole_step_rocprof_bucket_compare.json`) and **`figures_v2/fig31_repo_engineering_closure_vs_deployment.png`** (in-repo vs stack handoff table).
