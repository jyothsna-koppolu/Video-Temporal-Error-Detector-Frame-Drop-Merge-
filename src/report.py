import os, json
import pandas as pd
import matplotlib.pyplot as plt
from jinja2 import Template

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_charts(df: pd.DataFrame, charts_dir: str):
    ensure_dir(charts_dir)

    # Label timeline
    plt.figure()
    y = df["label"].map({"Normal":0, "Frame Drop":1, "Frame Merge":2}).fillna(0).to_numpy()
    plt.plot(df["frame_idx"], y)
    plt.yticks([0,1,2], ["Normal","Drop","Merge"])
    plt.xlabel("Frame Index")
    plt.ylabel("Class")
    plt.title("Frame-level Classification Timeline")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "label_timeline.png"), dpi=160)
    plt.close()

    # Diff scores
    plt.figure()
    plt.plot(df["frame_idx"], df["diff_prev"])
    plt.xlabel("Frame Index")
    plt.ylabel("Diff to Prev")
    plt.title("Diff Score (Prev) Over Time")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "diff_scores.png"), dpi=160)
    plt.close()

    # Optical flow
    plt.figure()
    plt.plot(df["frame_idx"], df["flow_mean"])
    plt.xlabel("Frame Index")
    plt.ylabel("Flow Mean")
    plt.title("Optical Flow Mean Over Time")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "flow_scores.png"), dpi=160)
    plt.close()

def build_dashboard(df: pd.DataFrame, report_dir: str, annotated_video_rel: str):
    ensure_dir(report_dir)
    charts_dir = os.path.join(report_dir, "charts")
    save_charts(df, charts_dir)

    summary = {
        "frames": int(df.shape[0]),
        "normal": int((df["label"]=="Normal").sum()),
        "drops": int((df["label"]=="Frame Drop").sum()),
        "merges": int((df["label"]=="Frame Merge").sum()),
        "input_fps": float(df["fps"].iloc[0]) if "fps" in df.columns and len(df) else None,
    }
    with open(os.path.join(report_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    tpl = Template("""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Temporal Error Detector Dashboard</title>
  <style>
    body{font-family:Arial, sans-serif; margin:20px; color:#111;}
    .cards{display:flex; gap:12px; flex-wrap:wrap;}
    .card{border:1px solid #ddd; border-radius:12px; padding:14px; min-width:180px;}
    h1{margin:0 0 10px 0;}
    .grid{display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:16px; margin-top:14px;}
    img{max-width:100%; border:1px solid #eee; border-radius:12px;}
    code{background:#f6f6f6; padding:2px 6px; border-radius:6px;}
  </style>
</head>
<body>
  <h1>Temporal Error Detector — Report</h1>
  <div class="cards">
    <div class="card"><b>Total Frames</b><br>{{summary.frames}}</div>
    <div class="card"><b>Normal</b><br>{{summary.normal}}</div>
    <div class="card"><b>Frame Drops</b><br>{{summary.drops}}</div>
    <div class="card"><b>Frame Merges</b><br>{{summary.merges}}</div>
  </div>

  <p>
    Outputs:
    <ul>
      <li>Annotated video: <code>{{annotated_video_rel}}</code> (open from filesystem)</li>
      <li>Frame CSV: <code>output/report.csv</code></li>
      <li>Summary JSON: <code>reports/summary.json</code></li>
    </ul>
  </p>

  <div class="grid">
    <div>
      <h3>Label Timeline</h3>
      <img src="charts/label_timeline.png" alt="timeline">
    </div>
    <div>
      <h3>Diff Scores</h3>
      <img src="charts/diff_scores.png" alt="diff">
    </div>
    <div>
      <h3>Optical Flow</h3>
      <img src="charts/flow_scores.png" alt="flow">
    </div>
  </div>

  <h2>How detection works</h2>
  <ol>
    <li><b>Drop</b>: large motion/discontinuity spike above robust threshold.</li>
    <li><b>Merge</b>: medium-high diffs to both neighbors + reduced sharpness/ghosting cues.</li>
  </ol>
</body>
</html>
""")
    html = tpl.render(summary=summary, annotated_video_rel=annotated_video_rel)
    with open(os.path.join(report_dir, "dashboard.html"), "w", encoding="utf-8") as f:
        f.write(html)
