# Report UI (`report-ui`)

Interactive React/Vite viewer for `final_report.md` and `final_report_v2.md` with figure galleries.

## What It Loads

- Markdown reports from `report-ui/public/content/`
- Figure assets from:
  - `report-ui/public/content/figures/`
  - `report-ui/public/content/figures_v2/`

The app intentionally serves report assets from `public/content` so they are available in both dev and production builds.

**Benchmark-driven numbers** (`#problem` section, live scoreboard): `public/content/experiment_kv_metrics.json` is a snapshot of `results/bench_tq3_decode_*.json` and `results/bench_runtime_ratio_all_methods.json` (MI300X). After new runs, refresh that file so the UI shows measured `KV_fp16`, `KV_compressed`, and `ratio_observed` from the VM, not hand-derived constants.

## Run Locally

```bash
cd /root/workspace/amd-experiments/report-ui
npm install
npm run dev
```

Then open the Vite URL printed in the terminal (typically `http://localhost:5173`).

## Build and Preview

```bash
cd /root/workspace/amd-experiments/report-ui
npm run build
npm run preview
```

Build output is written to `report-ui/dist/`.

## Refresh Content From `report/`

When reports/figures are regenerated, copy them into `public/content` before running the UI:

```bash
cd /root/workspace/amd-experiments
# Regenerate v2 PNGs (includes story fig27–fig31 from results/*.json; fig30–31 = rocprof buckets + repo-vs-deployment closure)
python3 report/generate_figures_v2.py --results-dir results --output-dir report/figures_v2

cp report/final_report.md report-ui/public/content/
cp report/final_report_v2.md report-ui/public/content/
rm -rf report-ui/public/content/figures report-ui/public/content/figures_v2
cp -r report/figures report-ui/public/content/
cp -r report/figures_v2 report-ui/public/content/
```

## Troubleshooting

- Blank report page usually means missing files under `report-ui/public/content/`.
- If images fail to load, verify figure filenames in `report/figures*` match those referenced by `report-ui/src/dataSources.js`.
- If a stale build appears, clear `dist/` and rebuild with `npm run build`.
