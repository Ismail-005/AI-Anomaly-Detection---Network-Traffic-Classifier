"""
pcap_to_features.py
────────────────────────────────────────────────────────────────
Bridges real Wireshark captures (.pcapng/.pcap) to the
feature format expected by the trained models.

Two modes:
  1. Parse a CICFlowMeter CSV export directly (recommended)
  2. Parse raw tshark JSON output (requires tshark in PATH)

Usage -- CICFlowMeter CSV:
  python detection/pcap_to_features.py --mode cicflow \
         --input capture_cicflow.csv \
         --output data/real_traffic.csv

Usage -- tshark JSON:
  tshark -r capture.pcapng -T json > tshark_out.json
  python detection/pcap_to_features.py --mode tshark \
         --input tshark_out.json \
         --output data/real_traffic.csv

After generating real_traffic.csv, run the detector on it:
  python detection/realtime_detector.py
  (edit DATA path in realtime_detector.py or pass --source flag)
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# ── CICFlowMeter column -> our feature mapping ──────────────────────────
# CICFlowMeter exports column names like 'Flow Duration', 'Total Fwd Packets', etc.
# Adjust keys if your CICFlowMeter version differs.
CICFLOW_MAP = {
    "Flow Duration":                   "duration",        # microseconds -> divide by 1e6
    "Protocol":                        "protocol",
    "Source Port":                     "src_port",
    "Destination Port":                "dst_port",
    "Total Fwd Packets":               "fwd_packets",
    "Total Backward Packets":          "bwd_packets",
    "Total Length of Fwd Packets":     "fwd_bytes",
    "Total Length of Bwd Packets":     "bwd_bytes",
    "Packet Length Mean":              "pkt_len_mean",
    "Packet Length Std":               "pkt_len_std",
    "Flow IAT Mean":                   "flow_iat_mean",   # microseconds -> /1e6
    "Flow IAT Std":                    "flow_iat_std",
    "Fwd IAT Mean":                    "fwd_iat_mean",
    "Bwd IAT Mean":                    "bwd_iat_mean",
    "FIN Flag Count":                  "fin_flag_cnt",
    "SYN Flag Count":                  "syn_flag_cnt",
    "RST Flag Count":                  "rst_flag_cnt",
    "PSH Flag Count":                  "psh_flag_cnt",
    "ACK Flag Count":                  "ack_flag_cnt",
    "Down/Up Ratio":                   "down_up_ratio",
}

TARGET_FEATURES = list(CICFLOW_MAP.values())

def parse_cicflow(path: Path) -> pd.DataFrame:
    """Convert CICFlowMeter CSV to model-ready feature DataFrame."""
    raw = pd.read_csv(path)
    raw.columns = raw.columns.str.strip()

    out = pd.DataFrame()
    for src_col, dst_col in CICFLOW_MAP.items():
        if src_col in raw.columns:
            out[dst_col] = pd.to_numeric(raw[src_col], errors="coerce").fillna(0)
        else:
            print(f"[!] Column '{src_col}' not found -- filling with 0")
            out[dst_col] = 0.0

    # CICFlowMeter reports IAT and duration in microseconds
    for col in ["duration", "flow_iat_mean", "flow_iat_std", "fwd_iat_mean", "bwd_iat_mean"]:
        out[col] = out[col] / 1e6   # convert to seconds

    # Carry label if present in the CICFlowMeter output
    if "Label" in raw.columns:
        out["label_name"] = raw["Label"].str.strip()
    else:
        out["label_name"] = "Unknown"
        out["label"]      = -1

    return out

# ── tshark JSON parser ─────────────────────────────────────────────────
def parse_tshark(path: Path) -> pd.DataFrame:
    """
    Parse tshark JSON (tshark -T json) to flow-level features.
    Note: tshark gives packet-level data. We do a simple per-5-tuple
    aggregation here. For production, use CICFlowMeter instead.
    """
    with open(path, "r", encoding="utf-8") as f:
        packets = json.load(f)

    rows = []
    for pkt in packets:
        try:
            layers = pkt["_source"]["layers"]
            ip     = layers.get("ip", {})
            tcp    = layers.get("tcp", {})
            udp    = layers.get("udp", {})
            transport = tcp or udp

            rows.append({
                "src_ip":   ip.get("ip.src", ""),
                "dst_ip":   ip.get("ip.dst", ""),
                "protocol": int(ip.get("ip.proto", 0)),
                "src_port": int(transport.get("tcp.srcport") or transport.get("udp.srcport") or 0),
                "dst_port": int(transport.get("tcp.dstport") or transport.get("udp.dstport") or 0),
                "pkt_len":  int(layers.get("frame", {}).get("frame.len", 0)),
                "timestamp":float(layers.get("frame", {}).get("frame.time_epoch", 0)),
                "syn":      int(tcp.get("tcp.flags.syn", 0)),
                "fin":      int(tcp.get("tcp.flags.fin", 0)),
                "rst":      int(tcp.get("tcp.flags.reset", 0)),
                "psh":      int(tcp.get("tcp.flags.push", 0)),
                "ack":      int(tcp.get("tcp.flags.ack", 0)),
            })
        except Exception:
            continue

    pkt_df = pd.DataFrame(rows)
    if pkt_df.empty:
        sys.exit("[✗] No packets parsed from tshark JSON.")

    # Group into flows by 5-tuple
    pkt_df["flow_key"] = (
        pkt_df["src_ip"] + ":" + pkt_df["src_port"].astype(str) + "->" +
        pkt_df["dst_ip"] + ":" + pkt_df["dst_port"].astype(str) + "/" +
        pkt_df["protocol"].astype(str)
    )

    def agg_flow(g):
        g = g.sort_values("timestamp")
        iats = np.diff(g["timestamp"].values)
        return pd.Series({
            "duration":      g["timestamp"].iloc[-1] - g["timestamp"].iloc[0],
            "protocol":      g["protocol"].iloc[0],
            "src_port":      g["src_port"].iloc[0],
            "dst_port":      g["dst_port"].iloc[0],
            "fwd_packets":   len(g),
            "bwd_packets":   0,
            "fwd_bytes":     g["pkt_len"].sum(),
            "bwd_bytes":     0,
            "pkt_len_mean":  g["pkt_len"].mean(),
            "pkt_len_std":   g["pkt_len"].std(ddof=0),
            "flow_iat_mean": iats.mean() if len(iats) > 0 else 0,
            "flow_iat_std":  iats.std(ddof=0) if len(iats) > 0 else 0,
            "fwd_iat_mean":  iats.mean() if len(iats) > 0 else 0,
            "bwd_iat_mean":  0,
            "fin_flag_cnt":  g["fin"].sum(),
            "syn_flag_cnt":  g["syn"].sum(),
            "rst_flag_cnt":  g["rst"].sum(),
            "psh_flag_cnt":  g["psh"].sum(),
            "ack_flag_cnt":  g["ack"].sum(),
            "down_up_ratio": 0,
            "label":         -1,
            "label_name":    "Unknown",
        })

    flow_df = pkt_df.groupby("flow_key").apply(agg_flow).reset_index(drop=True)
    return flow_df[TARGET_FEATURES + ["label", "label_name"]]

# ── main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="pcap -> feature CSV converter")
    parser.add_argument("--mode",   choices=["cicflow", "tshark"], default="cicflow")
    parser.add_argument("--input",  required=True, help="Input file path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    src  = Path(args.input)
    dst  = Path(args.output)

    if not src.exists():
        sys.exit(f"[✗] Input file not found: {src}")

    print(f"[*] Mode  : {args.mode}")
    print(f"[*] Input : {src}")

    if args.mode == "cicflow":
        df = parse_cicflow(src)
    else:
        df = parse_tshark(src)

    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    print(f"[[OK]] Saved {len(df)} flows -> {dst}")
    print(f"    Columns: {list(df.columns)}")

if __name__ == "__main__":
    main()
