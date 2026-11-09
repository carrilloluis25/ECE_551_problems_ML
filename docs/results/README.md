# Results data

Evaluation rows copied from `slurm/eval_*.csv` on Easley (stride 10 = 355-image test subset;
stride 1 = the full 3,501-image set). `docs/images_src/build_images.py` reads these files to draw
`docs/images/fig_psnr_vs_step.png`. When new rows appear on the cluster, copy the file here again
(`scp lc2025@easley.alliance.unm.edu:ECE_551_problems_ML/slurm/eval_baseline_bs16_lr2e-5.csv .`)
and rerun the script.

- `eval_baseline_bs16_lr2e-5.csv`: from-scratch baseline, one row per checkpoint (25,000-step interval).
- `eval_lr_test.csv`: 10,000-step learning-rate test (two runs merged, with an `lr` column added).
- `eval_ldfdct_smoke_test.csv`: the authors' checkpoint (step 500,000) on the same subset.
