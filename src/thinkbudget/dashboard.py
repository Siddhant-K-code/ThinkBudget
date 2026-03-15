"""Dashboard application for ThinkBudget.

Serves the web dashboard UI that visualizes query history,
cost savings, tier distribution, and GPU metrics in real-time.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


DASHBOARD_DIR = Path(__file__).parent.parent.parent / "dashboard"


def create_dashboard_app() -> FastAPI:
    """Create the dashboard sub-application."""
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    async def dashboard_index():
        html_path = DASHBOARD_DIR / "index.html"
        if html_path.exists():
            return HTMLResponse(content=html_path.read_text())
        return HTMLResponse(content=_fallback_dashboard())

    return app


def _fallback_dashboard() -> str:
    """Inline fallback dashboard if the HTML file is missing."""
    return DASHBOARD_HTML


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ThinkBudget Dashboard</title>
<style>
  :root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --surface2: #1a1a2e;
    --border: #2a2a3e;
    --text: #e0e0e8;
    --text-dim: #8888a0;
    --accent: #6c5ce7;
    --accent2: #a29bfe;
    --green: #00e676;
    --orange: #ff9100;
    --red: #ff5252;
    --blue: #448aff;
    --cyan: #18ffff;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', monospace;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }

  .header {
    padding: 20px 32px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .header h1 {
    font-size: 20px;
    font-weight: 600;
    color: var(--accent2);
  }

  .header h1 span { color: var(--text-dim); font-weight: 400; }

  .status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--green);
    display: inline-block;
    margin-right: 8px;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  .grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    padding: 24px 32px;
  }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
  }

  .card-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text-dim);
    margin-bottom: 8px;
  }

  .card-value {
    font-size: 28px;
    font-weight: 700;
    color: var(--text);
  }

  .card-value.green { color: var(--green); }
  .card-value.accent { color: var(--accent2); }
  .card-value.orange { color: var(--orange); }
  .card-value.cyan { color: var(--cyan); }

  .card-sub {
    font-size: 12px;
    color: var(--text-dim);
    margin-top: 4px;
  }

  .section {
    padding: 0 32px 24px;
  }

  .section-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 16px;
  }

  .tier-bar-container {
    display: flex;
    gap: 8px;
    align-items: end;
    height: 120px;
    padding: 0 20px;
  }

  .tier-bar-wrapper {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    height: 100%;
    justify-content: flex-end;
  }

  .tier-bar {
    width: 100%;
    max-width: 60px;
    border-radius: 6px 6px 0 0;
    transition: height 0.5s ease;
    min-height: 4px;
  }

  .tier-label {
    font-size: 10px;
    color: var(--text-dim);
    margin-top: 6px;
    text-transform: uppercase;
  }

  .tier-count {
    font-size: 12px;
    color: var(--text);
    margin-bottom: 4px;
    font-weight: 600;
  }

  .history-table {
    width: 100%;
    border-collapse: collapse;
  }

  .history-table th {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-dim);
    text-align: left;
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
  }

  .history-table td {
    padding: 10px 12px;
    font-size: 13px;
    border-bottom: 1px solid var(--border);
    color: var(--text);
  }

  .history-table tr:hover { background: var(--surface2); }

  .tier-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
  }

  .tier-trivial { background: #1a3a1a; color: var(--green); }
  .tier-simple { background: #1a2a3a; color: var(--blue); }
  .tier-moderate { background: #2a2a1a; color: var(--orange); }
  .tier-complex { background: #2a1a2a; color: var(--accent2); }
  .tier-deep { background: #3a1a1a; color: var(--red); }

  .utilization-bar {
    width: 80px;
    height: 6px;
    background: var(--surface2);
    border-radius: 3px;
    overflow: hidden;
    display: inline-block;
    vertical-align: middle;
    margin-right: 6px;
  }

  .utilization-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s ease;
  }

  .gpu-panel {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
  }

  .gpu-metric {
    background: var(--surface2);
    border-radius: 8px;
    padding: 16px;
    text-align: center;
  }

  .gpu-metric-value {
    font-size: 24px;
    font-weight: 700;
    margin: 8px 0 4px;
  }

  .gpu-metric-label {
    font-size: 11px;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .empty-state {
    text-align: center;
    padding: 60px 20px;
    color: var(--text-dim);
  }

  .empty-state h3 { font-size: 16px; margin-bottom: 8px; color: var(--text); }
  .empty-state p { font-size: 13px; }
</style>
</head>
<body>

<div class="header">
  <h1>ThinkBudget <span>v0.1.0</span></h1>
  <div>
    <span class="status-dot"></span>
    <span id="gpu-name" style="font-size: 13px; color: var(--text-dim);">Detecting GPU...</span>
  </div>
</div>

<!-- Stats Cards -->
<div class="grid">
  <div class="card">
    <div class="card-label">Total Queries</div>
    <div class="card-value accent" id="total-queries">0</div>
  </div>
  <div class="card">
    <div class="card-label">Total GPU Cost</div>
    <div class="card-value orange" id="total-cost">$0.000000</div>
  </div>
  <div class="card">
    <div class="card-label">Tokens Saved</div>
    <div class="card-value green" id="tokens-saved">0</div>
    <div class="card-sub" id="cost-saved">$0.00 saved</div>
  </div>
  <div class="card">
    <div class="card-label">Avg Latency</div>
    <div class="card-value cyan" id="avg-latency">0ms</div>
  </div>
</div>

<!-- Tier Distribution -->
<div class="section">
  <div class="section-title">Tier Distribution</div>
  <div class="card">
    <div class="tier-bar-container" id="tier-bars">
      <div class="empty-state" style="width:100%">
        <p>No queries yet</p>
      </div>
    </div>
  </div>
</div>

<!-- GPU Status -->
<div class="section">
  <div class="section-title">GPU Status</div>
  <div class="card">
    <div class="gpu-panel">
      <div class="gpu-metric">
        <div class="gpu-metric-label">Power</div>
        <div class="gpu-metric-value orange" id="gpu-power">—</div>
      </div>
      <div class="gpu-metric">
        <div class="gpu-metric-label">Utilization</div>
        <div class="gpu-metric-value cyan" id="gpu-util">—</div>
      </div>
      <div class="gpu-metric">
        <div class="gpu-metric-label">Memory</div>
        <div class="gpu-metric-value accent" id="gpu-mem">—</div>
      </div>
    </div>
  </div>
</div>

<!-- Query History -->
<div class="section">
  <div class="section-title">Recent Queries</div>
  <div class="card" style="overflow-x: auto;">
    <table class="history-table">
      <thead>
        <tr>
          <th>Time</th>
          <th>Tier</th>
          <th>Prompt</th>
          <th>Budget</th>
          <th>Used</th>
          <th>Utilization</th>
          <th>Cost</th>
          <th>Saved</th>
          <th>Latency</th>
        </tr>
      </thead>
      <tbody id="history-body">
        <tr>
          <td colspan="9" class="empty-state">
            <h3>Waiting for queries</h3>
            <p>Send requests to the proxy to see them here</p>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</div>

<script>
const API_BASE = window.location.origin;
const TIER_COLORS = {
  trivial: '#00e676',
  simple: '#448aff',
  moderate: '#ff9100',
  complex: '#a29bfe',
  deep: '#ff5252',
};

async function fetchStats() {
  try {
    const resp = await fetch(`${API_BASE}/api/stats`);
    const data = await resp.json();

    document.getElementById('total-queries').textContent = data.total_queries.toLocaleString();
    document.getElementById('total-cost').textContent = `$${data.total_cost.toFixed(6)}`;
    document.getElementById('tokens-saved').textContent = data.total_tokens_saved.toLocaleString();
    document.getElementById('cost-saved').textContent = `$${data.total_cost_saved.toFixed(4)} saved`;
    document.getElementById('avg-latency').textContent = `${data.avg_latency_ms.toFixed(0)}ms`;

    if (data.gpu_info) {
      document.getElementById('gpu-name').textContent =
        data.gpu_info.available ? data.gpu_info.name : 'No GPU (estimation mode)';
    }

    // Tier distribution bars
    const tierDist = data.tier_distribution;
    if (Object.keys(tierDist).length > 0) {
      const maxCount = Math.max(...Object.values(tierDist));
      const barsHtml = ['trivial', 'simple', 'moderate', 'complex', 'deep'].map(tier => {
        const count = tierDist[tier] || 0;
        const height = maxCount > 0 ? Math.max(4, (count / maxCount) * 100) : 4;
        return `
          <div class="tier-bar-wrapper">
            <div class="tier-count">${count}</div>
            <div class="tier-bar" style="height: ${height}px; background: ${TIER_COLORS[tier]}"></div>
            <div class="tier-label">${tier}</div>
          </div>
        `;
      }).join('');
      document.getElementById('tier-bars').innerHTML = barsHtml;
    }
  } catch (e) {
    console.error('Failed to fetch stats:', e);
  }
}

async function fetchHistory() {
  try {
    const resp = await fetch(`${API_BASE}/api/history?limit=20`);
    const data = await resp.json();

    if (data.length === 0) return;

    const rows = data.map(r => {
      const time = new Date(r.timestamp * 1000).toLocaleTimeString();
      const utilPct = (r.budget_utilization * 100).toFixed(0);
      const utilColor = r.budget_utilization > 0.9 ? '#ff5252' :
                        r.budget_utilization > 0.7 ? '#ff9100' : '#00e676';
      return `
        <tr>
          <td style="color: var(--text-dim); font-size: 12px;">${time}</td>
          <td><span class="tier-badge tier-${r.tier}">${r.tier}</span></td>
          <td style="max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${r.prompt_preview}</td>
          <td>${r.thinking_budget.toLocaleString()}</td>
          <td>${r.thinking_tokens_used.toLocaleString()}</td>
          <td>
            <div class="utilization-bar">
              <div class="utilization-fill" style="width: ${utilPct}%; background: ${utilColor}"></div>
            </div>
            ${utilPct}%
          </td>
          <td style="color: var(--orange);">$${r.cost_dollars.toFixed(6)}</td>
          <td style="color: var(--green);">${r.tokens_saved.toLocaleString()}</td>
          <td>${r.latency_ms.toFixed(0)}ms</td>
        </tr>
      `;
    }).join('');

    document.getElementById('history-body').innerHTML = rows;
  } catch (e) {
    console.error('Failed to fetch history:', e);
  }
}

async function fetchGPU() {
  try {
    const resp = await fetch(`${API_BASE}/api/gpu`);
    const data = await resp.json();

    document.getElementById('gpu-power').textContent =
      data.available ? `${data.power_watts.toFixed(0)}W` : '—';
    document.getElementById('gpu-util').textContent =
      data.available ? `${data.utilization.toFixed(0)}%` : '—';
    document.getElementById('gpu-mem').textContent =
      data.available ? `${(data.memory_used_mb / 1024).toFixed(1)}GB` : '—';
  } catch (e) {
    console.error('Failed to fetch GPU:', e);
  }
}

// Poll every 2 seconds
setInterval(() => {
  fetchStats();
  fetchHistory();
  fetchGPU();
}, 2000);

// Initial fetch
fetchStats();
fetchHistory();
fetchGPU();
</script>
</body>
</html>"""
