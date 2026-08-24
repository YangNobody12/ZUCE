"""
Visualization Generator for ZUCE Qwen3.5-0.8B Benchmark Results
Generates:
1. Publication-Grade Multi-Panel PNG Plot (outputs/zuce_qwen35_benchmark_comparison.png)
2. Interactive HTML Dashboard with Chart.js (outputs/zuce_qwen35_dashboard.html)
"""

import json
import os
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Load benchmark results
json_path = Path("outputs/zuce_qwen35_benchmark_report.json")
if not json_path.exists():
    raise FileNotFoundError(f"Cannot find {json_path}")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract metrics
labels = ["Original (100%)", "ZUCE -10%", "ZUCE -30%", "ZUCE -50%"]
short_labels = ["Original", "ZUCE 10%", "ZUCE 30%", "ZUCE 50%"]
params_m = [d["parameters_m"] for d in data]
mlp_widths = [d["intermediate_size"] for d in data]
throughputs = [d["speed"]["throughput_tok_s"] for d in data]
tpots = [d["speed"]["tpot_ms"] for d in data]
ttfts = [d["speed"]["ttft_ms"] for d in data]

coding_scores = [d["scores"]["coding_syntax_pass_pct"] for d in data]
inst_scores = [d["scores"]["instruction_format_pass_pct"] for d in data]
thai_scores = [d["scores"]["thai_language_pass_pct"] for d in data]

# ============================================================
# 1. GENERATE MATPLOTLIB MULTI-PANEL CHART
# ============================================================

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axs = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
fig.patch.set_facecolor('#0f172a')

colors = ['#38bdf8', '#34d399', '#f59e0b', '#ec4899']

# Panel A: Throughput vs Parameter Count (Pareto Frontier)
ax1 = axs[0, 0]
ax1.set_facecolor('#1e293b')
ax1.plot(params_m, throughputs, color='#94a3b8', linestyle='--', linewidth=1.5, zorder=1)
for i in range(len(data)):
    ax1.scatter(params_m[i], throughputs[i], color=colors[i], s=180, zorder=3, edgecolors='white', linewidth=1.5)
    offset_y = 0.12 if i % 2 == 0 else -0.18
    ax1.annotate(
        f"{short_labels[i]}\n({throughputs[i]:.2f} tok/s, {params_m[i]:.1f}M)",
        (params_m[i], throughputs[i] + offset_y),
        color='#f8fafc',
        fontsize=10,
        fontweight='bold',
        ha='center',
        bbox=dict(boxstyle="round,pad=0.3", fc="#0f172a", ec=colors[i], lw=1.2)
    )
ax1.set_title("A. Pareto Frontier (Throughput vs. Parameter Scale)", color='#f8fafc', fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel("Total Parameters (Million)", color='#cbd5e1', fontsize=11)
ax1.set_ylabel("Throughput (Tokens / Sec) ↑", color='#cbd5e1', fontsize=11)
ax1.tick_params(colors='#94a3b8')
ax1.grid(color='#334155', linestyle=':')
ax1.set_ylim(6.8, 9.2)

# Panel B: Latency Profile (TPOT vs TTFT)
ax2 = axs[0, 1]
ax2.set_facecolor('#1e293b')
x = np.arange(len(labels))
width = 0.35
bars1 = ax2.bar(x - width/2, tpots, width, label='Time Per Output Token (TPOT)', color='#60a5fa', edgecolor='white', linewidth=0.8)
bars2 = ax2.bar(x + width/2, [t/2.5 for t in ttfts], width, label='TTFT (Scaled 1:2.5)', color='#f43f5e', edgecolor='white', linewidth=0.8)

for bar in bars1:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}ms", ha='center', va='bottom', color='#93c5fd', fontsize=9, fontweight='bold')
for i, bar in enumerate(bars2):
    raw_ttft = ttfts[i]
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f"{raw_ttft:.0f}ms", ha='center', va='bottom', color='#fda4af', fontsize=9, fontweight='bold')

ax2.set_title("B. Latency Breakdown (TPOT & TTFT ms)", color='#f8fafc', fontsize=13, fontweight='bold', pad=12)
ax2.set_xticks(x)
ax2.set_xticklabels(short_labels, color='#cbd5e1', fontsize=10, fontweight='bold')
ax2.tick_params(colors='#94a3b8')
ax2.set_ylabel("Latency (ms) ↓", color='#cbd5e1', fontsize=11)
ax2.legend(facecolor='#0f172a', edgecolor='#475569', labelcolor='#f8fafc', fontsize=9)
ax2.grid(color='#334155', linestyle=':')
ax2.set_ylim(0, 165)

# Panel C: MLP Width Slicing & Parameter Accounting
ax3 = axs[1, 0]
ax3.set_facecolor('#1e293b')
bars_mlp = ax3.bar(short_labels, mlp_widths, color=colors, edgecolor='white', linewidth=1.2, width=0.55)
for i, bar in enumerate(bars_mlp):
    yval = bar.get_height()
    reduc = (1 - yval/3584) * 100
    ax3.text(bar.get_x() + bar.get_width()/2, yval/2, f"{yval} neurons\n(-{reduc:.0f}%)", ha='center', va='center', color='#0f172a', fontsize=10, fontweight='bold')

ax3.set_title("C. MLP Width Slicing Progression", color='#f8fafc', fontsize=13, fontweight='bold', pad=12)
ax3.set_ylabel("MLP Intermediate Width (Neurons)", color='#cbd5e1', fontsize=11)
ax3.tick_params(colors='#94a3b8')
ax3.set_xticklabels(short_labels, color='#cbd5e1', fontsize=10, fontweight='bold')
ax3.grid(color='#334155', linestyle=':')
ax3.set_ylim(0, 4100)

# Panel D: Capability Retention Radar / Bar Metrics
ax4 = axs[1, 1]
ax4.set_facecolor('#1e293b')
x_idx = np.arange(len(short_labels))
b_w = 0.25

b1 = ax4.bar(x_idx - b_w, inst_scores, b_w, label='Instruction Following (JSON)', color='#10b981')
b2 = ax4.bar(x_idx, coding_scores, b_w, label='Coding Syntax Pass', color='#06b6d4')
b3 = ax4.bar(x_idx + b_w, thai_scores, b_w, label='Thai Language Coherence', color='#a855f7')

for b in b1:
    ax4.text(b.get_x() + b.get_width()/2, b.get_height() + 2, f"{b.get_height():.0f}%", ha='center', color='#6ee7b7', fontsize=8, fontweight='bold')
for b in b2:
    ax4.text(b.get_x() + b.get_width()/2, b.get_height() + 2, f"{b.get_height():.0f}%", ha='center', color='#67e8f9', fontsize=8, fontweight='bold')
for b in b3:
    ax4.text(b.get_x() + b.get_width()/2, b.get_height() + 2, f"{b.get_height():.0f}%", ha='center', color='#d8b4fe', fontsize=8, fontweight='bold')

ax4.set_title("D. Capability Retention Across Domains (%)", color='#f8fafc', fontsize=13, fontweight='bold', pad=12)
ax4.set_xticks(x_idx)
ax4.set_xticklabels(short_labels, color='#cbd5e1', fontsize=10, fontweight='bold')
ax4.tick_params(colors='#94a3b8')
ax4.set_ylabel("Pass Rate (%) ↑", color='#cbd5e1', fontsize=11)
ax4.legend(facecolor='#0f172a', edgecolor='#475569', labelcolor='#f8fafc', fontsize=9)
ax4.grid(color='#334155', linestyle=':')
ax4.set_ylim(0, 120)

plt.suptitle("ZUCE Capability Extraction Benchmark: Qwen3.5-0.8B Sweep (10%, 30%, 50% Reduction)", color='#f8fafc', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

out_png = Path("outputs/zuce_qwen35_benchmark_comparison.png")
plt.savefig(out_png, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()
print(f"[OK] Generated plot: {out_png.resolve()}")

# Copy to artifact dir for direct embedding
artifact_dir = Path(r"C:\Users\okpkm\.gemini\antigravity-ide\brain\1bb66027-8a20-416f-ad50-5009ee4adf30")
if artifact_dir.exists():
    shutil.copy(out_png, artifact_dir / "zuce_qwen35_benchmark_comparison.png")
    print(f"[OK] Copied plot to artifact directory: {artifact_dir / 'zuce_qwen35_benchmark_comparison.png'}")

# ============================================================
# 2. GENERATE INTERACTIVE HTML DASHBOARD
# ============================================================

dashboard_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ZUCE Qwen3.5-0.8B Benchmark Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg: #0b0f19;
            --card-bg: rgba(30, 41, 59, 0.7);
            --border: rgba(255, 255, 255, 0.1);
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --cyan: #06b6d4;
            --emerald: #10b981;
            --amber: #f59e0b;
            --pink: #ec4899;
            --purple: #8b5cf6;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }}
        body {{ background: var(--bg); color: var(--text); padding: 32px 24px; min-height: 100vh; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ text-align: center; margin-bottom: 36px; }}
        .badge {{ display: inline-block; padding: 6px 14px; background: rgba(6, 182, 212, 0.15); border: 1px solid var(--cyan); color: var(--cyan); border-radius: 9999px; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; }}
        h1 {{ font-size: 32px; font-weight: 800; background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 8px; }}
        p.subtitle {{ color: var(--text-muted); font-size: 16px; }}
        
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 20px; margin-bottom: 32px; }}
        .card {{ background: var(--card-bg); backdrop-filter: blur(12px); border: 1px solid var(--border); border-radius: 16px; padding: 24px; transition: transform 0.2s, box-shadow 0.2s; }}
        .card:hover {{ transform: translateY(-3px); box-shadow: 0 12px 24px -10px rgba(0,0,0,0.5); }}
        .card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }}
        .card-title {{ font-size: 15px; font-weight: 600; color: var(--text-muted); }}
        .pill {{ font-size: 11px; padding: 3px 8px; border-radius: 6px; font-weight: 700; }}
        .pill.cyan {{ background: rgba(6, 182, 212, 0.2); color: var(--cyan); }}
        .pill.emerald {{ background: rgba(16, 185, 129, 0.2); color: var(--emerald); }}
        .pill.amber {{ background: rgba(245, 158, 11, 0.2); color: var(--amber); }}
        .pill.pink {{ background: rgba(236, 72, 153, 0.2); color: var(--pink); }}
        
        .stat-val {{ font-size: 30px; font-weight: 800; color: #fff; margin-bottom: 4px; }}
        .stat-desc {{ font-size: 13px; color: var(--text-muted); }}
        
        .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(540px, 1fr)); gap: 24px; margin-bottom: 32px; }}
        .chart-card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 16px; padding: 24px; }}
        .chart-title {{ font-size: 18px; font-weight: 700; margin-bottom: 18px; color: #f1f5f9; display: flex; align-items: center; gap: 8px; }}
        
        .table-card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 16px; padding: 24px; overflow-x: auto; }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; }}
        th {{ padding: 12px 16px; font-size: 13px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; border-bottom: 1px solid var(--border); }}
        td {{ padding: 16px; font-size: 14px; border-bottom: 1px solid rgba(255,255,255,0.05); }}
        tr:last-child td {{ border-bottom: none; }}
        .highlight {{ color: var(--cyan); font-weight: 700; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <span class="badge">ZUCE Zero-Update Architecture Sweep</span>
            <h1>Qwen3.5-0.8B Capability Extraction Benchmark</h1>
            <p class="subtitle">Quantitative Evaluation across Original 100%, ZUCE 10%, 30%, and 50% Reduction Levels</p>
        </div>

        <div class="metrics-grid">
            <div class="card">
                <div class="card-header"><span class="card-title">Original Model</span><span class="pill cyan">100% Width</span></div>
                <div class="stat-val">752.4 M</div>
                <div class="stat-desc">3,584 Intermediate Width | 7.35 tok/s</div>
            </div>
            <div class="card">
                <div class="card-header"><span class="card-title">ZUCE 10% Pruned</span><span class="pill emerald">-10% MLP</span></div>
                <div class="stat-val">725.9 M <span style="font-size:16px; color:var(--emerald);">(-26.5M)</span></div>
                <div class="stat-desc">3,225 Intermediate Width | 7.71 tok/s (+4.9%)</div>
            </div>
            <div class="card">
                <div class="card-header"><span class="card-title">ZUCE 30% Pruned</span><span class="pill amber">-30% MLP</span></div>
                <div class="stat-val">673.1 M <span style="font-size:16px; color:var(--amber);">(-79.3M)</span></div>
                <div class="stat-desc">2,508 Intermediate Width | 8.29 tok/s (+12.8%)</div>
            </div>
            <div class="card">
                <div class="card-header"><span class="card-title">ZUCE 50% Pruned</span><span class="pill pink">-50% MLP</span></div>
                <div class="stat-val">620.3 M <span style="font-size:16px; color:var(--pink);">(-132.1M)</span></div>
                <div class="stat-desc">1,792 Intermediate Width | 8.61 tok/s (+17.1%)</div>
            </div>
        </div>

        <div class="charts-grid">
            <div class="chart-card">
                <div class="chart-title">⚡ Throughput vs. Model Parameters (Pareto Frontier)</div>
                <canvas id="paretoChart" height="260"></canvas>
            </div>
            <div class="chart-card">
                <div class="chart-title">⏱️ Latency Profile (TPOT ms & TTFT ms)</div>
                <canvas id="latencyChart" height="260"></canvas>
            </div>
            <div class="chart-card">
                <div class="chart-title">🎯 Capability Retention Across Task Domains (%)</div>
                <canvas id="capabilityChart" height="260"></canvas>
            </div>
            <div class="chart-card">
                <div class="chart-title">📐 MLP Neurons Slicing Progression</div>
                <canvas id="widthChart" height="260"></canvas>
            </div>
        </div>

        <div class="table-card">
            <div class="chart-title">📊 Comprehensive Comparison Matrix</div>
            <table>
                <thead>
                    <tr>
                        <th>Variant</th>
                        <th>Parameters</th>
                        <th>MLP Width</th>
                        <th>Throughput</th>
                        <th>TPOT (Latency)</th>
                        <th>Instruction (JSON)</th>
                        <th>Thai Language</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Qwen3.5-0.8B (Original)</strong></td>
                        <td>752.39 M</td>
                        <td>3,584 neurons</td>
                        <td>7.35 tok/s</td>
                        <td>136.1 ms</td>
                        <td><span class="highlight">100%</span></td>
                        <td><span class="highlight">100%</span></td>
                    </tr>
                    <tr>
                        <td><strong>ZUCE 10% Pruned</strong></td>
                        <td>725.92 M (-3.5%)</td>
                        <td>3,225 neurons</td>
                        <td>7.71 tok/s (+4.9%)</td>
                        <td>129.6 ms</td>
                        <td><span class="highlight">100%</span></td>
                        <td><span class="highlight">100%</span></td>
                    </tr>
                    <tr>
                        <td><strong>ZUCE 30% Pruned</strong></td>
                        <td>673.06 M (-10.5%)</td>
                        <td>2,508 neurons</td>
                        <td>8.29 tok/s (+12.8%)</td>
                        <td>120.6 ms</td>
                        <td><span class="highlight">100%</span></td>
                        <td>0% (Contrast Drop)</td>
                    </tr>
                    <tr>
                        <td><strong>ZUCE 50% Pruned</strong></td>
                        <td>620.27 M (-17.6%)</td>
                        <td>1,792 neurons</td>
                        <td>8.61 tok/s (+17.1%)</td>
                        <td>116.2 ms</td>
                        <td>50%</td>
                        <td>50%</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>

    <script>
        const labels = ["Original", "ZUCE 10%", "ZUCE 30%", "ZUCE 50%"];
        const throughputs = {throughputs};
        const paramsM = {params_m};
        const tpots = {tpots};
        const ttfts = {ttfts};
        const widths = {mlp_widths};
        const instScores = {inst_scores};
        const codeScores = {coding_scores};
        const thaiScores = {thai_scores};

        // 1. Pareto
        new Chart(document.getElementById('paretoChart'), {{
            type: 'line',
            data: {{
                labels: paramsM.map((p, i) => `${{labels[i]}} (${{p}}M)`),
                datasets: [{{
                    label: 'Throughput (tok/s)',
                    data: throughputs,
                    borderColor: '#38bdf8',
                    backgroundColor: 'rgba(56, 189, 248, 0.15)',
                    pointBackgroundColor: ['#38bdf8', '#34d399', '#f59e0b', '#ec4899'],
                    pointBorderColor: '#fff',
                    pointRadius: 8,
                    fill: true,
                    tension: 0.2
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{ min: 6.5, max: 9.2, grid: {{ color: 'rgba(255,255,255,0.05)' }}, ticks: {{ color: '#94a3b8' }} }},
                    x: {{ grid: {{ color: 'rgba(255,255,255,0.05)' }}, ticks: {{ color: '#cbd5e1' }} }}
                }},
                plugins: {{ legend: {{ display: false }} }}
            }}
        }});

        // 2. Latency
        new Chart(document.getElementById('latencyChart'), {{
            type: 'bar',
            data: {{
                labels: labels,
                datasets: [
                    {{ label: 'TPOT (ms/tok)', data: tpots, backgroundColor: '#60a5fa', borderRadius: 6 }},
                    {{ label: 'TTFT (ms)', data: ttfts, backgroundColor: '#f43f5e', borderRadius: 6 }}
                ]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{ grid: {{ color: 'rgba(255,255,255,0.05)' }}, ticks: {{ color: '#94a3b8' }} }},
                    x: {{ grid: {{ display: false }}, ticks: {{ color: '#cbd5e1' }} }}
                }},
                plugins: {{ legend: {{ labels: {{ color: '#f8fafc' }} }} }}
            }}
        }});

        // 3. Capability
        new Chart(document.getElementById('capabilityChart'), {{
            type: 'bar',
            data: {{
                labels: labels,
                datasets: [
                    {{ label: 'Instruction Following (JSON)', data: instScores, backgroundColor: '#10b981', borderRadius: 4 }},
                    {{ label: 'Coding Syntax Pass', data: codeScores, backgroundColor: '#06b6d4', borderRadius: 4 }},
                    {{ label: 'Thai Language Coherence', data: thaiScores, backgroundColor: '#a855f7', borderRadius: 4 }}
                ]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{ max: 110, grid: {{ color: 'rgba(255,255,255,0.05)' }}, ticks: {{ color: '#94a3b8' }} }},
                    x: {{ grid: {{ display: false }}, ticks: {{ color: '#cbd5e1' }} }}
                }},
                plugins: {{ legend: {{ labels: {{ color: '#f8fafc' }} }} }}
            }}
        }});

        // 4. MLP Width
        new Chart(document.getElementById('widthChart'), {{
            type: 'bar',
            data: {{
                labels: labels,
                datasets: [{{
                    label: 'MLP Intermediate Width',
                    data: widths,
                    backgroundColor: ['#38bdf8', '#34d399', '#f59e0b', '#ec4899'],
                    borderRadius: 8
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{ max: 4000, grid: {{ color: 'rgba(255,255,255,0.05)' }}, ticks: {{ color: '#94a3b8' }} }},
                    x: {{ grid: {{ display: false }}, ticks: {{ color: '#cbd5e1' }} }}
                }},
                plugins: {{ legend: {{ display: false }} }}
            }}
        }});
    </script>
</body>
</html>
"""

dashboard_path = Path("outputs/zuce_qwen35_dashboard.html")
with open(dashboard_path, "w", encoding="utf-8") as f:
    f.write(dashboard_html)

print(f"[OK] Generated Interactive HTML Dashboard: {dashboard_path.resolve()}")
