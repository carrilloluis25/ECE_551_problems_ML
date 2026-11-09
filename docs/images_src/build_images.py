#!/usr/bin/env python3
"""Builds the tutorial images in docs/images/ (standard library + headless Chrome, no pip installs).

Run from anywhere:  python3 docs/images_src/build_images.py
  - term_*.png : terminal-style renderings of REAL command output from this project, with numbered
                 call-outs saying what to look for. The text lives in TERMINALS below; other users'
                 usernames are replaced by user_a, user_b, ...
  - fig_psnr_vs_step.png : PSNR and SSIM against training step, drawn from docs/results/*.csv.
                 Copy a newer eval CSV from Easley into docs/results/ and rerun to update it.
HTML pages go to docs/images_src/html/ (kept, so the images can be inspected or edited).
"""
import csv
import html
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.dirname(HERE)
OUT = os.path.join(DOCS, "images")
HTML_DIR = os.path.join(HERE, "html")
RES = os.path.join(DOCS, "results")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
W = 1000          # content width in px
LINE_H = 20       # terminal line height in px
WRAP_CHARS = 124  # characters that fit on one terminal line before it wraps

# Each terminal image: a list of blocks (title, [(text, marker or None)]) and a legend {marker: text}.
# A line whose text starts with "$ " is a typed command.
TERMINALS = {
    "term_squeue_states": {
        "caption": "Reading `squeue --me`: running, waiting on an earlier job, and held (real output, 2026-11-08/09)",
        "blocks": [
            ("squeue --me   (the baseline chain, 2026-11-09)", [
                ("$ squeue --me", None),
                ("   JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)", None),
                ("1410428      h100 fastddpm   lc2025 PD       0:00      1 (Dependency)", 2),
                ("1410426      h100 fastddpm   lc2025 PD       0:00      1 (Dependency)", 2),
                ("1410424      h100 fastddpm   lc2025  R    1:18:45      1 easley051", 1),
                ("1410430 interacti fastddpm   lc2025 PD       0:00      1 (Dependency)", 2),
            ]),
            ("squeue --me   (a held job, 2026-11-08)", [
                ("$ squeue --me", None),
                ("   JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)", None),
                ("1410412      h100     wrap   lc2025 PD       0:00      1 (user env retrieval failed requeued held)", 3),
            ]),
        ],
        "legend": {
            1: "ST = R: running. TIME is how long it has run; NODELIST names the node. This is chunk 2 of the chain.",
            2: "ST = PD with (Dependency): waiting for the earlier job in the chain to finish. Normal; nothing to do.",
            3: "ST = PD with a 'held' reason: SLURM gave up starting it and it will never run on its own. scancel it.",
        },
    },
    "term_h100_nodes": {
        "caption": "Check what is on the H100 nodes before you pin or wait (real output, 2026-11-08; other users renamed)",
        "blocks": [
            ("which nodes are free", [
                ('$ sinfo -p h100 -N -o "%N %t %G %C"', None),
                ("NODELIST STATE GRES CPUS(A/I/O/T)", None),
                ("easley051 idle gpu:h100:2 0/64/0/64", 1),
                ("easley052 down* gpu:h100:2 0/0/64/64", 2),
                ("easley053 mix- gpu:h100:2 22/42/0/64", 3),
                ("easley054 idle gpu:h100:2 0/64/0/64", 1),
            ]),
            ("who is using them", [
                ('$ squeue -p h100 -o "%.9i %.9u %.10T %.10M %.11l %.8N %b"', None),
                ("    JOBID      USER      STATE       TIME  TIME_LIMIT NODELIST TRES_PER_NODE", None),
                ("  1410389    lc2025    PENDING       0:00    14:00:00          gres/gpu:h100:1", 4),
                ("  1410156    user_b    RUNNING    3:45:03     8:00:00 easley053 gres/gpu:1", 3),
                ("  1410100    user_c    RUNNING    4:18:16     8:00:00 easley053 gres/gpu:h100:1", 3),
            ]),
        ],
        "legend": {
            1: "idle: both H100 cards free. Jobs not pinned to a node could start here at once.",
            2: "down*: the node is out of service. Exclude it (--exclude=easley052); the scheduler already avoids it.",
            3: "easley053 is busy: two other users hold its two cards for up to 8 h. A job pinned there would wait.",
            4: "our pinned job: it waited here while easley051 and easley054 sat idle.",
        },
    },
    "term_job_start": {
        "caption": "What a healthy start of the baseline looks like (real output, job 1410422, 2026-11-08)",
        "blocks": [
            ("head -n 12 fastddpm-base-1410422.out", [
                ("$ head -n 12 fastddpm-base-1410422.out", None),
                ("=== job=1410422 node=easley054 gpus=0 | NVIDIA H100 NVL, 95830 MiB ===", 1),
                ("=== doc baseline_bs16_lr2e-5 | step 0 -> 125000 (total 500000) | fresh start ===", 2),
                ("35:training:", None),
                ("36:    batch_size: 16", 3),
                ("39:    snapshot_freq: 25000", 3),
                ("43:    batch_size: 8", None),
                ("48:    batch_size: 59", None),
                ("52:    batch_size: 16", None),
                ("55:optim:", None),
                ("58:    lr: 0.00002", 3),
                ("Start training your Fast-DDPM model on LDFDCT dataset.", None),
                ("The scheduler sampling type is uniform. The number of involved time steps is 10 out of 1000.", None),
            ]),
            ("tail -n 3 fastddpm-base-1410422.err", [
                ("$ tail -n 3 fastddpm-base-1410422.err", None),
                ("INFO - diffusion.py - 2026-11-08 21:44:31,120 - step: 1379, loss: 2968.958740234375", 4),
                ("INFO - diffusion.py - 2026-11-08 21:44:31,465 - step: 1380, loss: 9752.75390625", 4),
                ("INFO - diffusion.py - 2026-11-08 21:44:31,811 - step: 1381, loss: 3574.048583984375", 4),
            ]),
        ],
        "legend": {
            1: "The GPU is an H100 NVL and the job is running on a compute node (not the login node).",
            2: "'fresh start' means no checkpoint was found. On a resumed chunk this says --resume_training.",
            3: "The values you set: training batch 16, a checkpoint every 25,000 steps, learning rate 0.00002. The other batch_size lines belong to the sampling sections.",
            4: "One line per step, about 0.345 s apart (31.120, 31.465, 31.811). The loss jumps between ~3,000 and ~10,000; that is normal and does not track quality.",
        },
    },
    "term_eval_csv": {
        "caption": "Scoring checkpoints: the evaluation CSV of the baseline's first chunk (real output, 2026-11-09)",
        "blocks": [
            ("eval_baseline_bs16_lr2e-5.csv", [
                ("$ cat ~/ECE_551_problems_ML/slurm/eval_baseline_bs16_lr2e-5.csv", None),
                ("ckpt,psnr,ssim,s_per_batch,stride", None),
                ("25000,21.297526798197776,0.505467816136362,3.6154983270735968,10", 1),
                ("50000,24.222188712495498,0.6066381819114556,3.6160633791060675,10", 1),
                ("75000,28.38989679103902,0.7190344049146473,3.615858168829055,10", 1),
                ("100000,32.487506631391724,0.8144058427923756,3.620847452254522,10", 1),
                ("125000,34.983822260968225,0.86670524245379,3.622118359520322,10", 2),
            ]),
        ],
        "legend": {
            1: "One row per checkpoint. PSNR (dB) and SSIM should rise from row to row; the gain per 25,000 steps is the number to watch.",
            2: "The authors' fully trained model scores 37.46 dB / 0.9153 on this same subset. stride 10 = every 10th slice (355 images).",
        },
    },
    "term_sacct_chunk": {
        "caption": "Confirming a chunk and its evaluation finished cleanly (real output, 2026-11-09)",
        "blocks": [
            ("sacct", [
                ("$ sacct -j 1410422,1410423 -X --format=JobID,JobName,State,ExitCode,Elapsed,NodeList", None),
                ("JobID           JobName      State ExitCode    Elapsed        NodeList", None),
                ("------------ ---------- ---------- -------- ---------- ---------------", None),
                ("1410422      fastddpm-+  COMPLETED      0:0   12:01:15       easley054", 1),
                ("1410423      fastddpm-+  COMPLETED      0:0   00:11:39       easley057", 2),
            ]),
        ],
        "legend": {
            1: "COMPLETED with ExitCode 0:0 is what lets the next chunk start (--dependency=afterok). 12 h of training.",
            2: "The evaluation ran on a different node (an L40S on the interactive partition), so it did not use an H100.",
        },
    },
    "term_held_diagnosis": {
        "caption": "Diagnosing a held job (real output, job 1410402, 2026-11-08)",
        "blocks": [
            ("scontrol and sacct", [
                ('$ scontrol show job 1410402 | grep -E "JobState|Reason|NodeList|BatchHost|Restarts"', None),
                ("   JobState=PENDING Reason=user_env_retrieval_failed_requeued_held Dependency=(null)", 1),
                ("   Requeue=1 Restarts=1 BatchFlag=2 Reboot=0 ExitCode=0:0", 2),
                ("   ReqNodeList=(null) ExcNodeList=easley052", None),
                ("   NodeList=", 3),
                ("$ sacct -j 1410402 -X --format=JobID,State,NodeList,Start,End", None),
                ("JobID             State        NodeList               Start                 End", None),
                ("------------ ---------- --------------- ------------------- -------------------", None),
                ("1410402         PENDING   None assigned             Unknown             Unknown", 3),
            ]),
        ],
        "legend": {
            1: "The reason names the failure: SLURM could not retrieve the user's environment on the node.",
            2: "Restarts=1: it was given a node, failed, and was requeued into a hold. It will not retry by itself.",
            3: "No node is recorded, so you cannot see where it failed. Cause found by testing: --export=ALL,VAR=... on the H100 nodes.",
        },
    },
    "term_dry_run": {
        "caption": "A dry run of the baseline chain: what would be submitted, nothing submitted (real output, 2026-11-04)",
        "blocks": [
            ("DRY=1 bash submit_baseline.sh   (first two chunks and the last line)", [
                ("$ DRY=1 bash submit_baseline.sh", None),
                ("[dry] sbatch --nodelist=easley053 --export=ALL,TOTAL=500000,CHUNK=125000,DOC=baseline_bs16_lr2e-5 baseline_train.sh", 1),
                ("chunk 1: steps 0 -> 125000   train job J323", 2),
                ("[dry] sbatch --dependency=afterok:J323 --export=ALL,DOC=baseline_bs16_lr2e-5,CKPTS=25000 50000 75000 100000 125000,STRIDE=10 eval_checkpoints.sh", 3),
                ("         eval job J558   checkpoints: 25000 50000 75000 100000 125000", 3),
                ("[dry] sbatch --nodelist=easley053 --dependency=afterok:J323 --export=ALL,TOTAL=500000,CHUNK=125000,DOC=baseline_bs16_lr2e-5 baseline_train.sh", 4),
                ("chunk 2: steps 125000 -> 250000   train job J541", None),
                ("...", None),
                ("final: full test set at step 500000   eval job J477   (results: eval_baseline_bs16_lr2e-5.csv, stride column 1)", 5),
            ]),
        ],
        "legend": {
            1: "The [dry] lines are the sbatch commands that would run. Job IDs (J323, ...) are placeholders so the chain can be read.",
            2: "Chunk boundaries: four chunks of 125,000 steps. Check the arithmetic here before a real submit.",
            3: "Each chunk is followed by an evaluation of its checkpoints, started only after that chunk succeeds (afterok).",
            4: "Chunk 2 depends on chunk 1 (afterok:J323), so a failed chunk stops everything behind it. (This 2026-11-04 version still used --export and --nodelist; see the troubleshooting log.)",
            5: "The last job scores the final checkpoint on the full 3,501-image test set. The '...' line is elided here.",
        },
    },
}


def term_html(spec):
    parts = []
    height = 48                      # top padding + caption
    for title, lines in spec["blocks"]:
        rows = []
        n_rows = 0
        for text, marker in lines:
            n_rows += max(1, -(-len(text) // WRAP_CHARS))
            cls = "cmd" if text.startswith("$ ") else ("hl" if marker else "")
            badge = f'<span class="badge">{marker}</span>' if marker else ""
            rows.append(f'<div class="ln {cls}">{html.escape(text)}{badge}</div>')
        parts.append(
            f'<div class="term"><div class="bar"><i></i><i></i><i></i><span>{html.escape(title)}</span></div>'
            f'<div class="body">{"".join(rows)}</div></div>'
        )
        height += 34 + 24 + LINE_H * n_rows + 18
    legend = "".join(
        f'<li><span class="badge">{k}</span>{html.escape(v)}</li>' for k, v in spec["legend"].items()
    )
    legend_lines = sum(-(-len(v) // 118) for v in spec["legend"].values())
    height += 34 + 22 * legend_lines + 10 * len(spec["legend"]) + 16
    page = f"""<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;background:#f4f5f7;font-family:-apple-system,Helvetica,Arial,sans-serif;color:#1b1f24}}
.wrap{{width:{W}px;padding:12px 20px 0 20px}}
h1{{font-size:15px;margin:2px 0 12px 0;font-weight:600}}
.term{{background:#1e2227;border-radius:8px;margin-bottom:18px;overflow:hidden}}
.bar{{background:#2c313a;height:34px;display:flex;align-items:center;padding:0 12px}}
.bar i{{width:11px;height:11px;border-radius:50%;background:#e06c75;margin-right:7px;display:inline-block}}
.bar i:nth-child(2){{background:#e5c07b}} .bar i:nth-child(3){{background:#98c379}}
.bar span{{color:#9da5b4;font-size:12px;margin-left:10px;font-family:Menlo,Consolas,monospace}}
.body{{padding:12px 14px}}
.ln{{font-family:Menlo,Consolas,'DejaVu Sans Mono',monospace;font-size:12.5px;line-height:{LINE_H}px;min-height:{LINE_H}px;
white-space:pre-wrap;word-break:break-all;color:#d7dae0}}
.ln.cmd{{color:#98c379}}
.ln.hl{{background:#2f3541;box-shadow:inset 3px 0 0 #e5c07b;margin:0 -14px;padding:0 14px 0 17px}}
.badge{{display:inline-block;background:#e5c07b;color:#1e2227;border-radius:9px;min-width:18px;height:18px;line-height:18px;
text-align:center;font:600 11px Helvetica,Arial,sans-serif;margin-left:10px;vertical-align:1px}}
.legend{{margin:0;padding:0;list-style:none;font-size:13.5px;line-height:22px}}
.legend li{{margin-bottom:10px}}
.legend .badge{{margin:0 10px 0 0;background:#c98a00;color:#fff}}
h2{{font-size:13px;margin:0 0 10px 0;font-weight:600;color:#4a5260;text-transform:uppercase;letter-spacing:.04em}}
</style></head><body><div class="wrap"><h1>{html.escape(spec["caption"])}</h1>{"".join(parts)}
<h2>What to look for</h2><ul class="legend">{legend}</ul></div></body></html>"""
    return page, height


def read_csv(name):
    path = os.path.join(RES, name)
    with open(path) as f:
        return list(csv.DictReader(f))


def chart_html():
    base = read_csv("eval_baseline_bs16_lr2e-5.csv")
    lr = read_csv("eval_lr_test.csv")
    ref = read_csv("eval_ldfdct_smoke_test.csv")[0]
    ref_psnr, ref_ssim = float(ref["psnr"]), float(ref["ssim"])
    thr_psnr, thr_ssim = 37.15, 0.910
    total = 500_000
    last = max(int(r["ckpt"]) for r in base)

    def panel(ox, metric, y0, y1, ticks, ref_v, thr_v, fmt, ylabel):
        pw, ph, top = 430, 250, 50
        def sx(x):
            return ox + 56 + (x / total) * (pw - 56 - 12)
        def sy(v):
            return top + (1 - (v - y0) / (y1 - y0)) * ph
        s = [f'<text x="{ox + 56}" y="{top - 24}" font-size="14" font-weight="600" fill="#1b1f24">{ylabel}</text>']
        # pending region
        s.append(f'<rect x="{sx(last)}" y="{top}" width="{sx(total) - sx(last)}" height="{ph}" fill="#eef0f3"/>')
        s.append(f'<text x="{(sx(last) + sx(total)) / 2}" y="{top + ph * 0.62}" font-size="11" fill="#7a8290" text-anchor="middle">not yet run (chunks 2 to 4)</text>')
        for t in ticks:
            s.append(f'<line x1="{ox + 56}" x2="{ox + pw - 12}" y1="{sy(t)}" y2="{sy(t)}" stroke="#d8dce2"/>')
            s.append(f'<text x="{ox + 50}" y="{sy(t) + 4}" font-size="11" fill="#4a5260" text-anchor="end">{fmt(t)}</text>')
        for xv in range(0, total + 1, 100_000):
            s.append(f'<line x1="{sx(xv)}" x2="{sx(xv)}" y1="{top + ph}" y2="{top + ph + 4}" stroke="#4a5260"/>')
            lab = "0" if xv == 0 else f"{xv // 1000}k"
            s.append(f'<text x="{sx(xv)}" y="{top + ph + 18}" font-size="11" fill="#4a5260" text-anchor="middle">{lab}</text>')
        s.append(f'<text x="{ox + 56 + (pw - 68) / 2}" y="{top + ph + 38}" font-size="12" fill="#4a5260" text-anchor="middle">training step</text>')
        s.append(f'<line x1="{ox + 56}" x2="{ox + pw - 12}" y1="{top + ph}" y2="{top + ph}" stroke="#4a5260"/>')
        # reference and threshold
        s.append(f'<line x1="{ox + 56}" x2="{ox + pw - 12}" y1="{sy(ref_v)}" y2="{sy(ref_v)}" stroke="#555" stroke-dasharray="6 4" stroke-width="1.5"/>')
        s.append(f'<text x="{ox + pw - 14}" y="{sy(ref_v) - 6}" font-size="11" fill="#333" text-anchor="end">authors\' model (step 500k): {fmt(ref_v)}</text>')
        s.append(f'<line x1="{ox + 56}" x2="{ox + pw - 12}" y1="{sy(thr_v)}" y2="{sy(thr_v)}" stroke="#c92a2a" stroke-dasharray="2 3" stroke-width="1.5"/>')
        s.append(f'<text x="{ox + pw - 14}" y="{sy(thr_v) + 14}" font-size="11" fill="#c92a2a" text-anchor="end">pass threshold: {fmt(thr_v)}</text>')
        # LR test points
        for lrv, col, lab in (("0.00002", "#2f9e44", "lr test 2e-5"), ("0.0002", "#d9480f", "lr test 2e-4")):
            pts = [(int(r["ckpt"]), float(r[metric])) for r in lr if r["lr"] == lrv]
            for x, v in pts:
                s.append(f'<circle cx="{sx(x)}" cy="{sy(v)}" r="4.5" fill="#fff" stroke="{col}" stroke-width="2"/>')
        # baseline
        pts = [(int(r["ckpt"]), float(r[metric])) for r in base]
        path = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(v):.1f}" for i, (x, v) in enumerate(pts))
        s.append(f'<path d="{path}" fill="none" stroke="#1f5fbf" stroke-width="2.5"/>')
        for x, v in pts:
            s.append(f'<circle cx="{sx(x)}" cy="{sy(v)}" r="4" fill="#1f5fbf"/>')
        x, v = pts[-1]
        s.append(f'<text x="{sx(x) + 8}" y="{sy(v) + 16}" font-size="11" fill="#1f5fbf">{fmt(v)} at {x // 1000}k</text>')
        return "".join(s)

    left = panel(0, "psnr", 10, 40, [10, 15, 20, 25, 30, 35, 40], ref_psnr, thr_psnr, lambda v: f"{v:.2f}" if isinstance(v, float) and v % 1 else f"{v:g}", "PSNR (dB), higher is better")
    right = panel(500, "ssim", 0.3, 1.0, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ref_ssim, thr_ssim, lambda v: f"{v:.3f}" if abs(v * 100 - round(v * 100)) > 1e-9 or v in (ref_ssim, thr_ssim) else f"{v:.1f}", "SSIM, higher is better")
    legend = (
        '<g font-size="12" fill="#1b1f24">'
        '<line x1="30" x2="60" y1="378" y2="378" stroke="#1f5fbf" stroke-width="2.5"/><circle cx="45" cy="378" r="4" fill="#1f5fbf"/>'
        '<text x="68" y="382">baseline from scratch (batch 16, lr 2e-5), every 25k steps</text>'
        '<circle cx="470" cy="378" r="4.5" fill="#fff" stroke="#2f9e44" stroke-width="2"/><text x="482" y="382">10k-step test, lr 2e-5</text>'
        '<circle cx="660" cy="378" r="4.5" fill="#fff" stroke="#d9480f" stroke-width="2"/><text x="672" y="382">10k-step test, lr 2e-4</text>'
        '</g>'
    )
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="400" font-family="Helvetica,Arial,sans-serif">{left}{right}{legend}</svg>'
    page = f"""<!doctype html><html><head><meta charset="utf-8"><style>body{{margin:0;background:#fff}}
.wrap{{width:1000px;padding:10px 20px 0 20px}}h1{{font:600 15px Helvetica,Arial,sans-serif;margin:6px 0 0 0;color:#1b1f24}}</style></head>
<body><div class="wrap"><h1>Fast-DDPM LDFDCT denoising trained from scratch on Easley: quality on the 355-image test subset</h1>{svg}</div></body></html>"""
    return page, 450


def render(name, page, height, width=W + 40):
    os.makedirs(HTML_DIR, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    html_path = os.path.join(HTML_DIR, name + ".html")
    png_path = os.path.join(OUT, name + ".png")
    with open(html_path, "w") as f:
        f.write(page)
    subprocess.run(
        [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
         f"--window-size={width},{height}", f"--screenshot={png_path}", "file://" + html_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print("wrote", os.path.relpath(png_path, DOCS))


if __name__ == "__main__":
    for name, spec in TERMINALS.items():
        page, h = term_html(spec)
        render(name, page, h)
    page, h = chart_html()
    render("fig_psnr_vs_step", page, h)
