"""
live_dashboard.py
─────────────────────────────────────────────────────────────
Live web dashboard for AI anomaly detection on deez.pkt network.
Runs detector in background thread; pushes updates to browser
via Server-Sent Events (SSE). No page refresh needed.

Usage:
  python detection/live_dashboard.py
  Then open:  http://localhost:5000
"""

import json
import time
import threading
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
import joblib
from flask import Flask, Response, render_template_string

# ── paths ─────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent.parent
MODELS = BASE / "models"
DATA   = BASE / "data" / "traffic_dataset.csv"

FEATURES = [
    "duration","protocol","src_port","dst_port",
    "fwd_packets","bwd_packets","fwd_bytes","bwd_bytes",
    "pkt_len_mean","pkt_len_std","flow_iat_mean","flow_iat_std",
    "fwd_iat_mean","bwd_iat_mean","fin_flag_cnt","syn_flag_cnt",
    "rst_flag_cnt","psh_flag_cnt","ack_flag_cnt","down_up_ratio",
]

LABEL_MAP = {0:"Normal",1:"DDoS",2:"Port Scan",3:"Brute Force",4:"Data Exfil"}

DEVICE_MAP = {
    "192.168.10.10":"HR-PC1","192.168.10.11":"HR-PC2","192.168.10.1":"ASA-Inside",
    "192.168.20.10":"IT-PC1","192.168.20.11":"IT-PC2",
    "192.168.20.100":"NMS-Srv1","192.168.20.50":"AP1","192.168.20.1":"DS1-VLAN20",
    "192.168.30.10":"Finance-PC1","192.168.30.11":"Finance-PC2","192.168.30.1":"DS1-VLAN30",
    "203.0.113.2":"ASA-Outside","203.0.113.1":"R1-WAN","203.0.113.100":"Internet-Srv",
    "10.0.0.2":"DS1-Uplink","172.16.1.10":"DNSSrv1","172.16.1.20":"WebSrv1",
    "45.33.32.156":"ATTACKER-1","185.220.101.42":"ATTACKER-2",
    "23.129.64.10":"ATTACKER-3","91.219.236.222":"ATTACKER-4","104.248.30.77":"ATTACKER-5",
    "198.51.100.50":"ExtHost-1","198.51.100.80":"ExtHost-2","198.51.100.99":"ExtHost-3",
}

def get_device(ip): return DEVICE_MAP.get(str(ip), str(ip))
def get_vlan(ip):
    for p,v in [("192.168.10","VLAN10-HR"),("192.168.20","VLAN20-IT"),
                ("192.168.30","VLAN30-Finance"),("172.16.1","VLAN99-DMZ"),
                ("203.0.113","WAN"),("10.0.0","Internal-Link")]:
        if str(ip).startswith(p): return v
    return "External"

# ── shared state ──────────────────────────────────────────────────────
state = {
    "flows":       deque(maxlen=300),   # last 300 flow events
    "counts":      {"Normal":0,"DDoS":0,"Port Scan":0,"Brute Force":0,"Data Exfil":0},
    "total":       0,
    "anomalies":   0,
    "vlan_hits":   {},
    "device_hits": {},
    "running":     False,
    "lock":        threading.Lock(),
}

# ── detector thread ───────────────────────────────────────────────────
def run_detector():
    rf     = joblib.load(MODELS/"random_forest.pkl")
    iso    = joblib.load(MODELS/"isolation_forest.pkl")
    scaler = joblib.load(MODELS/"scaler.pkl")

    df = pd.read_csv(DATA)
    # loop forever through dataset
    idx = 0
    state["running"] = True
    while state["running"]:
        row = df.iloc[[idx % len(df)]]
        idx += 1

        X  = scaler.transform(row[FEATURES].fillna(0).values)
        rf_pred  = int(rf.predict(X)[0])
        rf_proba = float(rf.predict_proba(X)[0].max())
        iso_pred = iso.predict(X)[0]
        iso_anom = bool(iso_pred == -1)

        label = LABEL_MAP.get(rf_pred, "Unknown")
        is_anom = label != "Normal"

        src_ip = str(row["src_ip"].iloc[0]) if "src_ip" in row.columns else "?"
        dst_ip = str(row["dst_ip"].iloc[0]) if "dst_ip" in row.columns else "?"

        event = {
            "ts":         datetime.now().strftime("%H:%M:%S"),
            "label":      label,
            "conf":       round(rf_proba * 100, 1),
            "iso":        iso_anom,
            "anomaly":    is_anom,
            "src_device": get_device(src_ip),
            "dst_device": get_device(dst_ip),
            "src_vlan":   get_vlan(src_ip),
            "dst_vlan":   get_vlan(dst_ip),
            "protocol":   int(row["protocol"].iloc[0]),
            "dst_port":   int(row["dst_port"].iloc[0]),
        }

        with state["lock"]:
            state["flows"].append(event)
            state["counts"][label] = state["counts"].get(label, 0) + 1
            state["total"] += 1
            if is_anom:
                state["anomalies"] += 1
                dv = event["dst_vlan"]
                dd = event["dst_device"]
                state["vlan_hits"][dv]   = state["vlan_hits"].get(dv, 0) + 1
                state["device_hits"][dd] = state["device_hits"].get(dd, 0) + 1

        time.sleep(0.4)   # one flow every 400ms for smooth live feel

# ── SSE endpoint ──────────────────────────────────────────────────────
def event_stream():
    last_seen = 0
    while True:
        with state["lock"]:
            flows = list(state["flows"])
            new   = flows[last_seen:]
            last_seen = len(flows)
            snap = {
                "counts":      state["counts"].copy(),
                "total":       state["total"],
                "anomalies":   state["anomalies"],
                "vlan_hits":   dict(sorted(state["vlan_hits"].items(),   key=lambda x:-x[1])[:6]),
                "device_hits": dict(sorted(state["device_hits"].items(), key=lambda x:-x[1])[:8]),
                "new_flows":   new[-20:],   # latest 20
            }
        yield f"data: {json.dumps(snap)}\n\n"
        time.sleep(0.5)

# ── Flask app ──────────────────────────────────────────────────────────
app = Flask(__name__)

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AI Network Anomaly Detection — deez.pkt</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
  *{box-sizing:border-box;margin:0;padding:0}
  :root{
    --bg:#0d1117;--card:#161b22;--border:#30363d;--accent:#00d4ff;
    --text:#e6edf3;--dim:#8b949e;--green:#6bcb77;--red:#ff6b6b;
    --yellow:#ffd93d;--purple:#c77dff;--orange:#ff9f43;
  }
  body{background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;min-height:100vh}
  header{
    background:linear-gradient(135deg,#0d1117 0%,#161b22 100%);
    border-bottom:1px solid var(--border);
    padding:18px 28px;display:flex;align-items:center;justify-content:space-between;
    position:sticky;top:0;z-index:100;backdrop-filter:blur(10px);
  }
  .logo{display:flex;align-items:center;gap:12px}
  .logo-icon{
    width:40px;height:40px;border-radius:10px;
    background:linear-gradient(135deg,var(--accent),#0070ff);
    display:flex;align-items:center;justify-content:center;font-size:20px;
  }
  .logo h1{font-size:17px;font-weight:700;color:var(--text)}
  .logo p{font-size:11px;color:var(--dim);margin-top:2px}
  .status-bar{display:flex;align-items:center;gap:20px}
  .pulse{width:10px;height:10px;border-radius:50%;background:var(--green);
    box-shadow:0 0 0 0 rgba(107,203,119,.4);
    animation:pulse 1.5s infinite}
  @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(107,203,119,.4)}
    70%{box-shadow:0 0 0 10px rgba(107,203,119,0)}100%{box-shadow:0 0 0 0 rgba(107,203,119,0)}}
  .stat-pill{background:var(--card);border:1px solid var(--border);
    border-radius:20px;padding:5px 14px;font-size:12px;color:var(--dim)}
  .stat-pill span{color:var(--text);font-weight:600;margin-left:4px}
  main{padding:24px 28px;display:grid;
    grid-template-columns:repeat(4,1fr);
    grid-template-rows:auto auto auto;
    gap:16px;max-width:1600px;margin:0 auto}
  .card{background:var(--card);border:1px solid var(--border);
    border-radius:14px;padding:18px;position:relative;overflow:hidden}
  .card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,var(--accent),transparent)}
  .card-title{font-size:11px;font-weight:600;color:var(--dim);
    text-transform:uppercase;letter-spacing:.8px;margin-bottom:14px}

  /* KPI row */
  .kpi-grid{grid-column:span 4;display:grid;grid-template-columns:repeat(4,1fr);gap:16px}
  .kpi{background:var(--card);border:1px solid var(--border);border-radius:14px;
    padding:18px 22px;display:flex;align-items:center;gap:16px;position:relative;overflow:hidden}
  .kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:2px}
  .kpi.total::before{background:var(--accent)}
  .kpi.anom::before{background:var(--red)}
  .kpi.rate::before{background:var(--yellow)}
  .kpi.top::before{background:var(--purple)}
  .kpi-icon{width:44px;height:44px;border-radius:12px;display:flex;
    align-items:center;justify-content:center;font-size:22px;flex-shrink:0}
  .kpi.total .kpi-icon{background:rgba(0,212,255,.12)}
  .kpi.anom .kpi-icon{background:rgba(255,107,107,.12)}
  .kpi.rate .kpi-icon{background:rgba(255,217,61,.12)}
  .kpi.top .kpi-icon{background:rgba(199,125,255,.12)}
  .kpi-val{font-size:28px;font-weight:700;line-height:1}
  .kpi-label{font-size:12px;color:var(--dim);margin-top:4px}
  .kpi.total .kpi-val{color:var(--accent)}
  .kpi.anom .kpi-val{color:var(--red)}
  .kpi.rate .kpi-val{color:var(--yellow)}
  .kpi.top .kpi-val{color:var(--purple)}

  /* chart cards */
  .chart-donut{grid-column:span 1}
  .chart-bar{grid-column:span 2}
  .chart-line{grid-column:span 1}
  .chart-hbar{grid-column:span 2}
  .chart-device{grid-column:span 2}
  canvas{max-height:220px}

  /* Feed */
  .feed{grid-column:span 4}
  .feed-inner{max-height:260px;overflow-y:auto;display:flex;flex-direction:column;gap:6px}
  .feed-inner::-webkit-scrollbar{width:4px}
  .feed-inner::-webkit-scrollbar-track{background:transparent}
  .feed-inner::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px}
  .flow-row{
    display:grid;
    grid-template-columns:70px 120px 1fr 1fr 80px 80px;
    gap:8px;align-items:center;
    background:#0d1117;border:1px solid var(--border);
    border-radius:8px;padding:8px 12px;font-size:11px;
    font-family:'JetBrains Mono',monospace;
    animation:slideIn .3s ease;
  }
  @keyframes slideIn{from{opacity:0;transform:translateY(-8px)}to{opacity:1;transform:none}}
  .flow-row.anomaly{border-color:rgba(255,107,107,.35);background:rgba(255,107,107,.04)}
  .label-badge{
    display:inline-block;padding:2px 8px;border-radius:12px;
    font-size:10px;font-weight:600;text-align:center;
  }
  .badge-Normal{background:rgba(107,203,119,.15);color:var(--green)}
  .badge-DDoS{background:rgba(255,107,107,.15);color:var(--red)}
  .badge-Port.Scan,.badge-Port{background:rgba(255,217,61,.15);color:var(--yellow)}
  .badge-Brute.Force,.badge-Brute{background:rgba(199,125,255,.15);color:var(--purple)}
  .badge-Data.Exfil,.badge-Data{background:rgba(255,159,67,.15);color:var(--orange)}
  .flow-ts{color:var(--dim)}
  .flow-src{color:var(--accent);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .flow-dst{color:var(--text);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .conf-bar{height:4px;border-radius:2px;background:var(--border);margin-top:3px}
  .conf-fill{height:100%;border-radius:2px;background:var(--accent)}
  .iso-flag{font-size:10px;padding:1px 6px;border-radius:10px}
  .iso-anom{background:rgba(255,107,107,.15);color:var(--red)}
  .iso-norm{background:rgba(107,203,119,.15);color:var(--green)}

  @media(max-width:1100px){
    main{grid-template-columns:repeat(2,1fr)}
    .kpi-grid,.feed{grid-column:span 2}
    .chart-bar,.chart-hbar,.chart-device{grid-column:span 2}
  }
</style>
</head>
<body>
<header>
  <div class="logo">
    <div class="logo-icon">&#x1F6E1;</div>
    <div>
      <h1>AI Network Anomaly Detection</h1>
      <p>deez.pkt &mdash; Internet &rarr; R1 &rarr; ASA 5505 &rarr; DS1 &rarr; VLAN10/20/30</p>
    </div>
  </div>
  <div class="status-bar">
    <div class="pulse"></div>
    <span style="font-size:12px;color:var(--green);font-weight:600">LIVE</span>
    <div class="stat-pill">Random Forest + Isolation Forest</div>
    <div class="stat-pill" id="hdr-time">--:--:--</div>
  </div>
</header>

<main>
  <!-- KPI row -->
  <div class="kpi-grid">
    <div class="kpi total"><div class="kpi-icon">&#x1F4E1;</div>
      <div><div class="kpi-val" id="k-total">0</div><div class="kpi-label">Total Flows</div></div></div>
    <div class="kpi anom"><div class="kpi-icon">&#x26A0;</div>
      <div><div class="kpi-val" id="k-anom">0</div><div class="kpi-label">Anomalies</div></div></div>
    <div class="kpi rate"><div class="kpi-icon">&#x1F4CA;</div>
      <div><div class="kpi-val" id="k-rate">0%</div><div class="kpi-label">Anomaly Rate</div></div></div>
    <div class="kpi top"><div class="kpi-icon">&#x1F3AF;</div>
      <div><div class="kpi-val" id="k-top">—</div><div class="kpi-label">Top Attack Type</div></div></div>
  </div>

  <!-- Donut chart -->
  <div class="card chart-donut">
    <div class="card-title">Traffic Distribution</div>
    <canvas id="donutChart"></canvas>
  </div>

  <!-- Bar chart -->
  <div class="card chart-bar">
    <div class="card-title">Class Counts (Live)</div>
    <canvas id="barChart"></canvas>
  </div>

  <!-- Anomaly rate line -->
  <div class="card chart-line">
    <div class="card-title">Anomaly Rate %</div>
    <canvas id="lineChart"></canvas>
  </div>

  <!-- VLAN heatmap bar -->
  <div class="card chart-hbar">
    <div class="card-title">Most Attacked Zones</div>
    <canvas id="vlanChart"></canvas>
  </div>

  <!-- Top devices -->
  <div class="card chart-device">
    <div class="card-title">Most Targeted Devices</div>
    <canvas id="deviceChart"></canvas>
  </div>

  <!-- Live feed -->
  <div class="card feed">
    <div class="card-title">Live Flow Feed &mdash; Real-Time Detection</div>
    <div class="feed-inner" id="feedEl"></div>
  </div>
</main>

<script>
const COLORS={
  Normal:'#6bcb77',DDoS:'#ff6b6b','Port Scan':'#ffd93d',
  'Brute Force':'#c77dff','Data Exfil':'#ff9f43'
};
const LABELS=['Normal','DDoS','Port Scan','Brute Force','Data Exfil'];
const ALPHA=c=>c+'33';
const BG='#0d1117',CARD='#161b22',TEXT='#e6edf3',DIM='#8b949e',ACC='#00d4ff';

Chart.defaults.color=TEXT;
Chart.defaults.borderColor='#30363d';
Chart.defaults.font.family="'Inter',sans-serif";

// ── donut ────────────────────────────────────────────────────────────
const donut=new Chart(document.getElementById('donutChart'),{
  type:'doughnut',
  data:{labels:LABELS,datasets:[{
    data:[0,0,0,0,0],
    backgroundColor:LABELS.map(l=>COLORS[l]),
    borderColor:CARD,borderWidth:3,hoverOffset:8
  }]},
  options:{responsive:true,cutout:'72%',plugins:{legend:{position:'bottom',
    labels:{boxWidth:10,font:{size:10},padding:8}}}}
});

// ── bar ──────────────────────────────────────────────────────────────
const bar=new Chart(document.getElementById('barChart'),{
  type:'bar',
  data:{labels:LABELS,datasets:[{
    label:'Flows',data:[0,0,0,0,0],
    backgroundColor:LABELS.map(l=>COLORS[l]),
    borderRadius:6,borderSkipped:false
  }]},
  options:{responsive:true,plugins:{legend:{display:false}},
    scales:{x:{grid:{color:'#21262d'}},y:{grid:{color:'#21262d'},beginAtZero:true}}}
});

// ── line (anomaly rate) ───────────────────────────────────────────────
const lineData={labels:[],datasets:[{
  label:'Anomaly %',data:[],borderColor:ACC,
  backgroundColor:'rgba(0,212,255,.08)',fill:true,
  tension:.4,pointRadius:0,borderWidth:2
}]};
const line=new Chart(document.getElementById('lineChart'),{
  type:'line',data:lineData,
  options:{responsive:true,animation:{duration:0},plugins:{legend:{display:false}},
    scales:{x:{display:false},y:{min:0,max:100,grid:{color:'#21262d'},
      ticks:{callback:v=>v+'%'}}}}
});

// ── vlan bar ──────────────────────────────────────────────────────────
const vlan=new Chart(document.getElementById('vlanChart'),{
  type:'bar',
  data:{labels:[],datasets:[{label:'Attack flows',data:[],
    backgroundColor:'rgba(255,107,107,.7)',borderRadius:6,borderSkipped:false}]},
  options:{indexAxis:'y',responsive:true,plugins:{legend:{display:false}},
    scales:{x:{grid:{color:'#21262d'},beginAtZero:true},y:{grid:{color:'#21262d'}}}}
});

// ── device bar ────────────────────────────────────────────────────────
const devChart=new Chart(document.getElementById('deviceChart'),{
  type:'bar',
  data:{labels:[],datasets:[{label:'Attack flows',data:[],
    backgroundColor:'rgba(199,125,255,.7)',borderRadius:6,borderSkipped:false}]},
  options:{indexAxis:'y',responsive:true,plugins:{legend:{display:false}},
    scales:{x:{grid:{color:'#21262d'},beginAtZero:true},y:{grid:{color:'#21262d'}}}}
});

// ── helpers ───────────────────────────────────────────────────────────
function badgeClass(lbl){
  const m={'Normal':'Normal','DDoS':'DDoS','Port Scan':'Port',
           'Brute Force':'Brute','Data Exfil':'Data'};
  return 'badge-'+(m[lbl]||lbl);
}

const rateWindow=[];

function applySnap(d){
  // KPIs
  document.getElementById('k-total').textContent=d.total.toLocaleString();
  document.getElementById('k-anom').textContent=d.anomalies.toLocaleString();
  const rate=d.total?((d.anomalies/d.total)*100).toFixed(1)+'%':'0%';
  document.getElementById('k-rate').textContent=rate;
  // top attack
  const attacks=Object.entries(d.counts).filter(([k])=>k!=='Normal');
  if(attacks.length){
    const top=attacks.sort((a,b)=>b[1]-a[1])[0][0];
    document.getElementById('k-top').textContent=top;
  }
  document.getElementById('hdr-time').textContent=new Date().toLocaleTimeString();

  // donut + bar
  const vals=LABELS.map(l=>d.counts[l]||0);
  donut.data.datasets[0].data=vals; donut.update('none');
  bar.data.datasets[0].data=vals;   bar.update('none');

  // line
  rateWindow.push(d.total?parseFloat(((d.anomalies/d.total)*100).toFixed(1)):0);
  if(rateWindow.length>60) rateWindow.shift();
  lineData.labels=rateWindow.map((_,i)=>i);
  lineData.datasets[0].data=[...rateWindow];
  line.update('none');

  // vlan
  const vk=Object.keys(d.vlan_hits), vv=Object.values(d.vlan_hits);
  vlan.data.labels=vk; vlan.data.datasets[0].data=vv; vlan.update('none');

  // devices
  const dk=Object.keys(d.device_hits), dv=Object.values(d.device_hits);
  devChart.data.labels=dk; devChart.data.datasets[0].data=dv; devChart.update('none');

  // feed
  const feed=document.getElementById('feedEl');
  d.new_flows.reverse().forEach(f=>{
    const row=document.createElement('div');
    row.className='flow-row'+(f.anomaly?' anomaly':'');
    const isoFlag=f.iso
      ?'<span class="iso-flag iso-anom">ANOMALY</span>'
      :'<span class="iso-flag iso-norm">NORMAL</span>';
    row.innerHTML=`
      <span class="flow-ts">${f.ts}</span>
      <span><span class="label-badge ${badgeClass(f.label)}">${f.label}</span></span>
      <span class="flow-src" title="${f.src_device}">${f.src_device}</span>
      <span class="flow-dst">&rarr; ${f.dst_device} <span style="color:var(--dim);font-size:10px">(${f.dst_vlan})</span></span>
      <span style="color:var(--dim)">:${f.dst_port}</span>
      ${isoFlag}`;
    feed.insertBefore(row, feed.firstChild);
    // keep max 50 rows
    while(feed.children.length>50) feed.removeChild(feed.lastChild);
  });
}

// ── SSE ───────────────────────────────────────────────────────────────
const es=new EventSource('/stream');
es.onmessage=e=>{
  try{ applySnap(JSON.parse(e.data)); }catch(err){ console.error(err); }
};
es.onerror=()=>{
  document.querySelector('.pulse').style.background='var(--red)';
};
</script>
</body>
</html>"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/stream")
def stream():
    return Response(event_stream(), mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

# ── entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    t = threading.Thread(target=run_detector, daemon=True)
    t.start()
    print("=" * 55)
    print("  Live Dashboard — deez.pkt AI Anomaly Detection")
    print("=" * 55)
    print("  Open your browser:  http://localhost:5000")
    print("  Press Ctrl+C to stop")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
