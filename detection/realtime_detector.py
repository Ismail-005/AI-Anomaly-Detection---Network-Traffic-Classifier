"""
realtime_detector.py
----------------------------------------------------------------
Simulates real-time traffic anomaly detection using the trained
AI models against the deez.pkt network topology.

Shows source/destination IPs mapped to actual device names and
VLANs, making it easy to see which devices are involved in
suspicious traffic.

Usage:
  python detection/realtime_detector.py [--speed 0.08] [--batch 1]
"""

import argparse
import time
import sys
import warnings
import threading
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

# -- paths -------------------------------------------------------------
BASE   = Path(__file__).parent.parent
MODELS = BASE / "models"
DATA   = BASE / "data" / "traffic_dataset.csv"

# -- model features (must match training) ------------------------------
FEATURES = [
    "duration", "protocol", "src_port", "dst_port",
    "fwd_packets", "bwd_packets", "fwd_bytes", "bwd_bytes",
    "pkt_len_mean", "pkt_len_std",
    "flow_iat_mean", "flow_iat_std", "fwd_iat_mean", "bwd_iat_mean",
    "fin_flag_cnt", "syn_flag_cnt", "rst_flag_cnt",
    "psh_flag_cnt", "ack_flag_cnt", "down_up_ratio",
]

LABEL_MAP = {
    0: "Normal",
    1: "DDoS",
    2: "Port Scan",
    3: "Brute Force",
    4: "Data Exfil",
}

# ======================================================================
#  TOPOLOGY MAPPING -- deez.pkt device names
# ======================================================================
DEVICE_MAP = {
    # VLAN 10 -- HR
    "192.168.10.10":  "HR-PC1",
    "192.168.10.11":  "HR-PC2",
    "192.168.10.1":   "ASA-Inside",
    # VLAN 20 -- IT
    "192.168.20.10":  "IT-PC1",
    "192.168.20.11":  "IT-PC2",
    "192.168.20.100": "NMS-Srv1",
    "192.168.20.50":  "AP1",
    "192.168.20.1":   "DS1-VLAN20",
    # VLAN 30 -- Finance
    "192.168.30.10":  "Finance-PC1",
    "192.168.30.11":  "Finance-PC2",
    "192.168.30.1":   "DS1-VLAN30",
    # Infrastructure
    "203.0.113.2":    "ASA-Outside",
    "203.0.113.1":    "R1-WAN",
    "203.0.113.100":  "Internet-Srv",
    "10.0.0.2":       "DS1-Uplink",
    # DMZ (skipped)
    "172.16.1.10":    "DNSSrv1",
    "172.16.1.20":    "WebSrv1",
    # Attackers
    "45.33.32.156":    "ATTACKER-1",
    "185.220.101.42":  "ATTACKER-2",
    "23.129.64.10":    "ATTACKER-3",
    "91.219.236.222":  "ATTACKER-4",
    "104.248.30.77":   "ATTACKER-5",
    # External
    "198.51.100.50":   "ExtHost-1",
    "198.51.100.80":   "ExtHost-2",
    "198.51.100.99":   "ExtHost-3",
}

VLAN_MAP = {
    "192.168.10": "VLAN10-HR",
    "192.168.20": "VLAN20-IT",
    "192.168.30": "VLAN30-Finance",
    "172.16.1":   "VLAN99-DMZ",
    "203.0.113":  "WAN",
    "10.0.0":     "Internal-Link",
}

def get_device_name(ip):
    """Map IP to device name."""
    return DEVICE_MAP.get(ip, ip)

def get_vlan(ip):
    """Map IP to VLAN zone."""
    for prefix, vlan in VLAN_MAP.items():
        if ip.startswith(prefix):
            return vlan
    return "External"


# -- ANSI colour codes -------------------------------------------------
RESET   = "\033[0m"
GREEN   = "\033[92m"
RED     = "\033[91m"
YELLOW  = "\033[93m"
CYAN    = "\033[96m"
BOLD    = "\033[1m"
MAGENTA = "\033[95m"
ORANGE  = "\033[38;5;208m"
DIM     = "\033[2m"

COLOUR = {
    "Normal":      GREEN,
    "DDoS":        RED,
    "Port Scan":   YELLOW,
    "Brute Force": MAGENTA,
    "Data Exfil":  ORANGE,
}

# -- load models -------------------------------------------------------
def load_models():
    required = ["random_forest.pkl", "isolation_forest.pkl", "scaler.pkl"]
    for f in required:
        if not (MODELS / f).exists():
            sys.exit(
                f"[✗] Missing model: {MODELS/f}\n"
                "    Run:  python models/train_model.py  first."
            )
    rf     = joblib.load(MODELS / "random_forest.pkl")
    iso    = joblib.load(MODELS / "isolation_forest.pkl")
    scaler = joblib.load(MODELS / "scaler.pkl")
    print(f"{GREEN}[OK] Models loaded successfully{RESET}\n")
    return rf, iso, scaler

# -- stream simulation ------------------------------------------------
def stream_samples(df, batch_size=1):
    """Yield rows one-by-one (or in batches) to simulate a live stream."""
    for i in range(0, len(df), batch_size):
        yield df.iloc[i : i + batch_size]

# -- alert system ------------------------------------------------------
alert_log = []

def make_alert(row, rf_label, iso_anomaly, confidence):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    label_name = LABEL_MAP.get(rf_label, "Unknown")
    colour     = COLOUR.get(label_name, RESET)
    iso_flag   = f"{RED}[ANOMALY]{RESET}" if iso_anomaly else f"{GREEN}[NORMAL]{RESET}"
    conf_bar   = "#" * int(confidence * 20) + "." * (20 - int(confidence * 20))
    
    # Get topology info
    src_ip  = str(row["src_ip"].iloc[0]) if "src_ip" in row.columns else "?"
    dst_ip  = str(row["dst_ip"].iloc[0]) if "dst_ip" in row.columns else "?"
    src_dev = get_device_name(src_ip)
    dst_dev = get_device_name(dst_ip)
    src_vlan = get_vlan(src_ip)
    dst_vlan = get_vlan(dst_ip)
    conf_bar   = "#" * int(confidence * 20) + "." * (20 - int(confidence * 20))

    line = (
        f"{CYAN}{ts}{RESET} | "
        f"RF: {colour}{BOLD}{label_name:<13}{RESET} "
        f"conf={YELLOW}{confidence:.0%}{RESET} [{conf_bar}] | "
        f"IsoF: {iso_flag} | "
        f"{DIM}{src_dev}{RESET} -> {DIM}{dst_dev}{RESET} "
        f"({src_vlan}->{dst_vlan}) "
        f"proto={int(row.protocol.iloc[0])} "
        f"dst_port={int(row.dst_port.iloc[0])}"
    )

    alert_log.append({
        "timestamp":    ts,
        "rf_label":     label_name,
        "confidence":   confidence,
        "iso_anomaly":  iso_anomaly,
        "src_ip":       src_ip,
        "dst_ip":       dst_ip,
        "src_device":   src_dev,
        "dst_device":   dst_dev,
        "src_vlan":     src_vlan,
        "dst_vlan":     dst_vlan,
        "protocol":     int(row.protocol.iloc[0]),
        "dst_port":     int(row.dst_port.iloc[0]),
    })
    return line, label_name

# -- statistics tracker ------------------------------------------------
class Stats:
    def __init__(self):
        self.counts    = {v: 0 for v in LABEL_MAP.values()}
        self.total     = 0
        self.anomalies = 0
        self.vlan_hits = {}
        self.lock      = threading.Lock()

    def update(self, label_name, is_anomaly, src_vlan="", dst_vlan=""):
        with self.lock:
            self.counts[label_name] = self.counts.get(label_name, 0) + 1
            self.total     += 1
            self.anomalies += int(is_anomaly)
            if is_anomaly:
                for v in [dst_vlan]:
                    self.vlan_hits[v] = self.vlan_hits.get(v, 0) + 1

    def print_summary(self):
        print(f"\n{'='*75}")
        print(f"{BOLD}  Detection Summary -- deez.pkt Network{RESET}")
        print(f"{'='*75}")
        print(f"  Total flows processed : {self.total}")
        print(f"  Anomalies detected    : {RED}{self.anomalies}{RESET} "
              f"({self.anomalies/max(self.total,1):.1%})")
        print(f"\n  {BOLD}Label Breakdown:{RESET}")
        for name, cnt in self.counts.items():
            colour = COLOUR.get(name, RESET)
            bar = "#" * int(cnt / max(self.total, 1) * 40)
            print(f"    {colour}{name:<14}{RESET}  {cnt:>5}  {bar}")
        if self.vlan_hits:
            print(f"\n  {BOLD}Most Targeted Zones:{RESET}")
            for vlan, cnt in sorted(self.vlan_hits.items(), key=lambda x: -x[1]):
                print(f"    {YELLOW}{vlan:<20}{RESET}  {cnt:>5} attack flows")
        print(f"{'='*75}\n")


# -- main detection loop ----------------------------------------------
def run_detector(speed=0.08, batch_size=1, max_flows=200):
    rf, iso, scaler = load_models()
    stats = Stats()

    if not DATA.exists():
        sys.exit(f"[✗] Dataset not found: {DATA}\n    Run generate_traffic.py first.")

    df = pd.read_csv(DATA)

    # Limit flows for demo
    if max_flows and max_flows < len(df):
        df = df.head(max_flows)

    print(f"{BOLD}{'='*75}")
    print("  [*] Real-Time Traffic Anomaly Detector")
    print(f"  [*] Network: deez.pkt -- AI-Enhanced Secure Enterprise")
    print(f"{'='*75}{RESET}")
    print(f"  Source  : {DATA.name}  ({len(df)} samples)")
    print(f"  Speed   : {speed}s per flow")
    print(f"  Models  : Random Forest + Isolation Forest")
    print(f"  Press   : Ctrl+C to stop\n")
    print(f"{'-'*75}")

    try:
        for batch in stream_samples(df, batch_size):
            X_raw = batch[FEATURES].fillna(0).values
            X_s   = scaler.transform(X_raw)

            # Random Forest (multi-class)
            rf_pred  = rf.predict(X_s)
            rf_proba = rf.predict_proba(X_s)

            # Isolation Forest (anomaly: -1 = anomaly)
            iso_pred = iso.predict(X_s)
            iso_anom = iso_pred == -1

            for i in range(len(batch)):
                label_name = LABEL_MAP.get(rf_pred[i], "Unknown")
                confidence = rf_proba[i].max()
                is_anomaly = label_name != "Normal"

                line, lname = make_alert(
                    batch.iloc[[i]], rf_pred[i], iso_anom[i], confidence
                )
                print(line)

                src_ip = str(batch.iloc[i].get("src_ip", ""))
                dst_ip = str(batch.iloc[i].get("dst_ip", ""))
                stats.update(lname, is_anomaly, get_vlan(src_ip), get_vlan(dst_ip))
                time.sleep(speed)

    except KeyboardInterrupt:
        print(f"\n{YELLOW}[!] Detection stopped by user.{RESET}")

    stats.print_summary()

    # Save alert log
    log_path = BASE / "reports" / "alert_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(alert_log).to_csv(log_path, index=False)
    print(f"[OK] Alert log saved -> {log_path}")

# -- CLI ---------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Anomaly Detector -- deez.pkt")
    parser.add_argument("--speed", type=float, default=0.02,
                        help="Seconds between each flow (default: 0.02)")
    parser.add_argument("--batch", type=int, default=1,
                        help="Flows per batch (default: 1)")
    parser.add_argument("--max", type=int, default=200,
                        help="Max flows to process (default: 200, 0=all)")
    args = parser.parse_args()
    run_detector(speed=args.speed, batch_size=args.batch,
                 max_flows=args.max if args.max > 0 else None)
