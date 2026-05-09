"""
generate_traffic.py
────────────────────────────────────────────────────────────────
Generates a labelled synthetic network traffic dataset that mirrors
the real enterprise traffic from the deez.pkt Packet Tracer topology.

Topology (deez.pkt):
  Internet ─── R1 (2911) ─── ASA 5505 ─── DS1 (3560) ─── AS1/AS2/AS3
                                                ├── VLAN 10 (HR)
                                                ├── VLAN 20 (IT)
                                                ├── VLAN 30 (Finance)
                                                └── VLAN 99 (DMZ) [skipped]

Devices & IP Addressing:
  VLAN 10 (HR):      192.168.10.0/24 ── HR-PC1 (.10), HR-PC2 (.11)
  VLAN 20 (IT):      192.168.20.0/24 ── IT-PC1 (.10), IT-PC2 (.11),
                                        NMS-Srv1 (.100), AP1 (.50)
  VLAN 30 (Finance): 192.168.30.0/24 ── Finance-PC1 (.10), Finance-PC2 (.11)
  VLAN 99 (DMZ):     172.16.1.0/24   ── DNSSrv1 (.10), WebSrv1 (.20) [skipped]
  ASA Inside:        192.168.10.1  (gateway to internal)
  ASA Outside:       203.0.113.2   (WAN-facing)
  R1 WAN:            203.0.113.1
  DS1 Uplink:        10.0.0.2
  Internet:          203.0.113.100

Traffic classes:
  0 – Normal        (HTTP, DNS, SMTP, HTTPS, inter-VLAN, NMS polling)
  1 – DDoS          (UDP/ICMP flood from external)
  2 – Port Scan     (TCP SYN sweep across ports)
  3 – Brute Force   (SSH/RDP repeated login attempts)
  4 – Data Exfil    (large outbound flows to suspicious IPs)

Features mirror CICFlowMeter output so real pcap -> CSV exports
can be dropped in as a direct replacement.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng  = np.random.default_rng(SEED)
OUT  = Path(__file__).parent / "traffic_dataset.csv"

# ══════════════════════════════════════════════════════════════════════
#  NETWORK TOPOLOGY — deez.pkt
# ══════════════════════════════════════════════════════════════════════

# ── VLAN 10: HR ──
HR_HOSTS = ["192.168.10.10", "192.168.10.11"]
HR_GW    = "192.168.10.1"

# ── VLAN 20: IT ──
IT_HOSTS    = ["192.168.20.10", "192.168.20.11"]
IT_SERVERS  = ["192.168.20.100"]   # NMS-Srv1
IT_AP       = ["192.168.20.50"]    # AP1
IT_GW       = "192.168.20.1"

# ── VLAN 30: Finance ──
FIN_HOSTS = ["192.168.30.10", "192.168.30.11"]
FIN_GW    = "192.168.30.1"

# ── Infrastructure ──
ASA_INSIDE   = "192.168.10.1"
ASA_OUTSIDE  = "203.0.113.2"
R1_WAN       = "203.0.113.1"
INTERNET_SRV = "203.0.113.100"
DS1_UPLINK   = "10.0.0.2"

# ── DMZ (skipped in Packet Tracer but included for completeness) ──
DMZ_DNS = "172.16.1.10"
DMZ_WEB = "172.16.1.20"

# ── All internal hosts (flat list for convenience) ──
ALL_INTERNAL = HR_HOSTS + IT_HOSTS + IT_SERVERS + IT_AP + FIN_HOSTS
ALL_GATEWAYS = [HR_GW, IT_GW, FIN_GW, ASA_INSIDE, DS1_UPLINK]
EXTERNAL_IPS = [INTERNET_SRV, "198.51.100.50", "198.51.100.80", "198.51.100.99"]

# ── Attacker IPs (simulated external threats) ──
ATTACKER_IPS = [
    "45.33.32.156",    # nmap scanme
    "185.220.101.42",  # Tor exit node
    "23.129.64.10",    # Tor exit node
    "91.219.236.222",  # known scanner
    "104.248.30.77",   # botnet C2
]

# ── helpers ──────────────────────────────────────────────────────────
def clip(arr, lo, hi):
    return np.clip(arr, lo, hi).astype(float)

def pick_ips(src_pool, dst_pool, n):
    """Pick random src/dst IP pairs from given pools."""
    src = rng.choice(src_pool, size=n)
    dst = rng.choice(dst_pool, size=n)
    return src, dst

def ip_to_numeric(ip_str):
    """Convert dotted-quad IP to a 32-bit integer for model features."""
    parts = ip_str.split(".")
    return (int(parts[0]) << 24) + (int(parts[1]) << 16) + \
           (int(parts[2]) << 8) + int(parts[3])

def ips_to_numeric(ip_arr):
    """Vectorised IP-to-int conversion."""
    return np.array([ip_to_numeric(ip) for ip in ip_arr], dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════
#  TRAFFIC GENERATORS
# ══════════════════════════════════════════════════════════════════════

def normal_traffic(n=3000):
    """
    Normal enterprise traffic patterns observed in deez.pkt:
    - HR browsing the internet (HTTP/HTTPS)
    - IT managing network via NMS (SNMP polls, SSH)
    - Finance accessing internal apps
    - Inter-VLAN traffic (HR <-> IT for help desk)
    - DNS queries to internet
    - ICMP ping for connectivity checks
    """
    protocols = rng.choice([6, 17, 1], size=n, p=[0.6, 0.3, 0.1])  # TCP/UDP/ICMP

    # 40% internal-to-internet, 35% inter-VLAN, 25% internal-to-gateway
    n_ext     = int(n * 0.40)
    n_inter   = int(n * 0.35)
    n_gw      = n - n_ext - n_inter

    # External browsing (HR/IT/Finance → Internet)
    src_ext, _    = pick_ips(ALL_INTERNAL, EXTERNAL_IPS, n_ext)
    dst_ext       = rng.choice(EXTERNAL_IPS, size=n_ext)

    # Inter-VLAN (HR ↔ IT, IT → Finance for audits, etc.)
    src_inter     = rng.choice(HR_HOSTS + IT_HOSTS, size=n_inter)
    dst_inter     = rng.choice(IT_HOSTS + FIN_HOSTS + IT_SERVERS, size=n_inter)

    # Gateway traffic (DHCP, ARP proxy, SNMP)
    src_gw        = rng.choice(ALL_INTERNAL, size=n_gw)
    dst_gw        = rng.choice(ALL_GATEWAYS, size=n_gw)

    src_ips = np.concatenate([src_ext, src_inter, src_gw])
    dst_ips = np.concatenate([dst_ext, dst_inter, dst_gw])

    # Shuffle to mix traffic types
    idx = rng.permutation(n)
    src_ips = src_ips[idx]
    dst_ips = dst_ips[idx]

    return pd.DataFrame({
        "src_ip":           src_ips,
        "dst_ip":           dst_ips,
        "src_ip_numeric":   ips_to_numeric(src_ips),
        "dst_ip_numeric":   ips_to_numeric(dst_ips),
        "duration":         clip(rng.exponential(2.5, n),       0, 300),
        "protocol":         protocols,
        "src_port":         rng.integers(1024, 65535, n),
        "dst_port":         rng.choice([80, 443, 53, 25, 22, 161, 3389], size=n,
                                       p=[0.25, 0.30, 0.15, 0.05, 0.10, 0.10, 0.05]),
        "fwd_packets":      clip(rng.poisson(12, n),            1, 500),
        "bwd_packets":      clip(rng.poisson(10, n),            1, 500),
        "fwd_bytes":        clip(rng.lognormal(7, 1.5, n),      0, 1e6),
        "bwd_bytes":        clip(rng.lognormal(7, 1.5, n),      0, 1e6),
        "pkt_len_mean":     clip(rng.normal(512, 200, n),       20, 1500),
        "pkt_len_std":      clip(rng.normal(150, 80, n),        0, 800),
        "flow_iat_mean":    clip(rng.exponential(0.05, n),      0, 10),
        "flow_iat_std":     clip(rng.exponential(0.02, n),      0, 5),
        "fwd_iat_mean":     clip(rng.exponential(0.1, n),       0, 10),
        "bwd_iat_mean":     clip(rng.exponential(0.1, n),       0, 10),
        "fin_flag_cnt":     rng.integers(0, 3, n),
        "syn_flag_cnt":     rng.integers(0, 2, n),
        "rst_flag_cnt":     rng.integers(0, 2, n),
        "psh_flag_cnt":     rng.integers(0, 6, n),
        "ack_flag_cnt":     rng.integers(0, 12, n),
        "down_up_ratio":    clip(rng.normal(1.1, 0.4, n),       0, 20),
        "label":            0,
        "label_name":       "Normal",
    })


def ddos_traffic(n=700):
    """
    DDoS attack: External attackers flood the ASA outside interface
    and internal servers with UDP/ICMP packets.
    Targets: ASA outside (203.0.113.2), IT servers, HR PCs
    """
    protocols = rng.choice([17, 1], size=n, p=[0.6, 0.4])

    # Attackers → ASA/internal hosts
    src_ips = rng.choice(ATTACKER_IPS, size=n)
    dst_ips = rng.choice(
        [ASA_OUTSIDE, R1_WAN] + IT_SERVERS + HR_HOSTS,
        size=n,
        p=[0.30, 0.10, 0.20, 0.20, 0.20]
    )

    return pd.DataFrame({
        "src_ip":           src_ips,
        "dst_ip":           dst_ips,
        "src_ip_numeric":   ips_to_numeric(src_ips),
        "dst_ip_numeric":   ips_to_numeric(dst_ips),
        "duration":         clip(rng.exponential(0.3, n),       0, 10),
        "protocol":         protocols,
        "src_port":         rng.integers(1024, 65535, n),
        "dst_port":         rng.choice([80, 443, 53], size=n),
        "fwd_packets":      clip(rng.poisson(800, n),           100, 5000),
        "bwd_packets":      clip(rng.poisson(5, n),             0, 50),
        "fwd_bytes":        clip(rng.normal(6000, 500, n),      100, 20000),
        "bwd_bytes":        clip(rng.normal(200, 50, n),        0, 1000),
        "pkt_len_mean":     clip(rng.normal(64, 10, n),         20, 200),
        "pkt_len_std":      clip(rng.normal(5, 2, n),           0, 20),
        "flow_iat_mean":    clip(rng.exponential(0.0001, n),    0, 0.01),
        "flow_iat_std":     clip(rng.exponential(0.00005, n),   0, 0.01),
        "fwd_iat_mean":     clip(rng.exponential(0.0001, n),    0, 0.01),
        "bwd_iat_mean":     clip(rng.exponential(0.01, n),      0, 0.1),
        "fin_flag_cnt":     np.zeros(n),
        "syn_flag_cnt":     rng.integers(0, 2, n),
        "rst_flag_cnt":     np.zeros(n),
        "psh_flag_cnt":     np.zeros(n),
        "ack_flag_cnt":     np.zeros(n),
        "down_up_ratio":    clip(rng.normal(0.01, 0.005, n),    0, 0.1),
        "label":            1,
        "label_name":       "DDoS",
    })


def port_scan_traffic(n=600):
    """
    Port scan: Attacker probes internal hosts across many ports.
    Targets: IT servers, Finance PCs, NMS server
    SYN-heavy, tiny packets, rapid succession.
    """
    src_ips = rng.choice(ATTACKER_IPS[:3], size=n)
    dst_ips = rng.choice(
        IT_HOSTS + FIN_HOSTS + IT_SERVERS,
        size=n
    )

    return pd.DataFrame({
        "src_ip":           src_ips,
        "dst_ip":           dst_ips,
        "src_ip_numeric":   ips_to_numeric(src_ips),
        "dst_ip_numeric":   ips_to_numeric(dst_ips),
        "duration":         clip(rng.exponential(0.05, n),      0, 2),
        "protocol":         np.full(n, 6),   # TCP
        "src_port":         rng.integers(1024, 65535, n),
        "dst_port":         rng.integers(1, 65535, n),       # random port sweep
        "fwd_packets":      clip(rng.poisson(1.5, n),           1, 10),
        "bwd_packets":      clip(rng.poisson(0.5, n),           0, 5),
        "fwd_bytes":        clip(rng.normal(74, 10, n),         40, 200),
        "bwd_bytes":        clip(rng.normal(40, 10, n),         0, 200),
        "pkt_len_mean":     clip(rng.normal(60, 10, n),         20, 150),
        "pkt_len_std":      clip(rng.normal(5, 2, n),           0, 20),
        "flow_iat_mean":    clip(rng.exponential(0.001, n),     0, 0.1),
        "flow_iat_std":     clip(rng.exponential(0.0005, n),    0, 0.05),
        "fwd_iat_mean":     clip(rng.exponential(0.001, n),     0, 0.1),
        "bwd_iat_mean":     clip(rng.exponential(0.001, n),     0, 0.1),
        "fin_flag_cnt":     np.zeros(n),
        "syn_flag_cnt":     np.ones(n),
        "rst_flag_cnt":     rng.integers(0, 2, n),
        "psh_flag_cnt":     np.zeros(n),
        "ack_flag_cnt":     np.zeros(n),
        "down_up_ratio":    clip(rng.normal(0.3, 0.1, n),       0, 1),
        "label":            2,
        "label_name":       "Port Scan",
    })


def brute_force_traffic(n=500):
    """
    Brute force: Attacker tries repeated SSH/RDP logins.
    Targets: IT PCs (SSH), Finance PCs (RDP), NMS-Srv1 (SSH)
    Pattern: many small packets, consistent timing, high SYN+ACK counts.
    """
    dst = rng.choice([22, 3389], size=n, p=[0.6, 0.4])

    src_ips = rng.choice(ATTACKER_IPS, size=n)
    # SSH targets = IT devices; RDP targets = Finance/HR PCs
    dst_ips = np.where(
        dst == 22,
        rng.choice(IT_HOSTS + IT_SERVERS, size=n),
        rng.choice(HR_HOSTS + FIN_HOSTS, size=n)
    )

    return pd.DataFrame({
        "src_ip":           src_ips,
        "dst_ip":           dst_ips,
        "src_ip_numeric":   ips_to_numeric(src_ips),
        "dst_ip_numeric":   ips_to_numeric(dst_ips),
        "duration":         clip(rng.normal(30, 10, n),         1, 120),
        "protocol":         np.full(n, 6),
        "src_port":         rng.integers(1024, 65535, n),
        "dst_port":         dst,
        "fwd_packets":      clip(rng.poisson(80, n),            10, 500),
        "bwd_packets":      clip(rng.poisson(80, n),            10, 500),
        "fwd_bytes":        clip(rng.normal(4000, 500, n),      100, 20000),
        "bwd_bytes":        clip(rng.normal(4000, 500, n),      100, 20000),
        "pkt_len_mean":     clip(rng.normal(80, 10, n),         40, 200),
        "pkt_len_std":      clip(rng.normal(10, 3, n),          0, 50),
        "flow_iat_mean":    clip(rng.normal(0.4, 0.1, n),       0, 2),
        "flow_iat_std":     clip(rng.normal(0.05, 0.02, n),     0, 0.5),
        "fwd_iat_mean":     clip(rng.normal(0.4, 0.1, n),       0, 2),
        "bwd_iat_mean":     clip(rng.normal(0.4, 0.1, n),       0, 2),
        "fin_flag_cnt":     rng.integers(0, 2, n),
        "syn_flag_cnt":     rng.integers(5, 20, n),
        "rst_flag_cnt":     rng.integers(0, 5, n),
        "psh_flag_cnt":     rng.integers(0, 5, n),
        "ack_flag_cnt":     rng.integers(10, 80, n),
        "down_up_ratio":    clip(rng.normal(1.0, 0.1, n),       0.5, 2),
        "label":            3,
        "label_name":       "Brute Force",
    })


def data_exfil_traffic(n=400):
    """
    Data exfiltration: Compromised internal host uploads data to
    external C2 servers via unusual ports.
    Source: Finance PCs (sensitive data) or IT PCs (admin access)
    Destination: External attacker IPs on non-standard ports
    Pattern: large outbound bytes, rare dst IPs, long duration.
    """
    # Compromised hosts in Finance & IT VLANs
    src_ips = rng.choice(FIN_HOSTS + IT_HOSTS, size=n, p=[0.3, 0.2, 0.3, 0.2])
    dst_ips = rng.choice(ATTACKER_IPS, size=n)

    return pd.DataFrame({
        "src_ip":           src_ips,
        "dst_ip":           dst_ips,
        "src_ip_numeric":   ips_to_numeric(src_ips),
        "dst_ip_numeric":   ips_to_numeric(dst_ips),
        "duration":         clip(rng.normal(120, 40, n),        10, 600),
        "protocol":         np.full(n, 6),
        "src_port":         rng.integers(1024, 65535, n),
        "dst_port":         rng.choice([4444, 8080, 1337, 9001, 6667], size=n),
        "fwd_packets":      clip(rng.poisson(50, n),            5, 300),
        "bwd_packets":      clip(rng.poisson(10, n),            1, 100),
        "fwd_bytes":        clip(rng.lognormal(12, 1, n),       1e4, 5e7),
        "bwd_bytes":        clip(rng.lognormal(7, 1, n),        100, 1e5),
        "pkt_len_mean":     clip(rng.normal(1400, 100, n),      800, 1500),
        "pkt_len_std":      clip(rng.normal(50, 20, n),         0, 200),
        "flow_iat_mean":    clip(rng.normal(0.02, 0.005, n),    0, 0.1),
        "flow_iat_std":     clip(rng.normal(0.01, 0.003, n),    0, 0.05),
        "fwd_iat_mean":     clip(rng.normal(0.02, 0.005, n),    0, 0.1),
        "bwd_iat_mean":     clip(rng.normal(0.05, 0.01, n),     0, 0.2),
        "fin_flag_cnt":     rng.integers(0, 2, n),
        "syn_flag_cnt":     rng.integers(0, 2, n),
        "rst_flag_cnt":     np.zeros(n),
        "psh_flag_cnt":     rng.integers(0, 5, n),
        "ack_flag_cnt":     rng.integers(10, 60, n),
        "down_up_ratio":    clip(rng.normal(0.05, 0.02, n),     0, 0.2),
        "label":            4,
        "label_name":       "Data Exfil",
    })


# ══════════════════════════════════════════════════════════════════════
#  ASSEMBLE & SAVE
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  Traffic Generator — deez.pkt Topology")
    print("=" * 65)
    print("  Network: Internet -> R1 -> ASA 5505 -> DS1 -> AS1/AS2/AS3")
    print("  VLANs:   10 (HR) | 20 (IT) | 30 (Finance)")
    print()

    frames = [
        normal_traffic(3000),
        ddos_traffic(700),
        port_scan_traffic(600),
        brute_force_traffic(500),
        data_exfil_traffic(400),
    ]
    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"[OK] Dataset saved -> {OUT}")
    print(f"    Total samples : {len(df)}")
    print(f"    Features      : {df.shape[1] - 2}")
    print(f"\n    Class distribution:")
    print(f"{df['label_name'].value_counts().to_string()}")
    print(f"\n    Source IP distribution (top 10):")
    print(f"{df['src_ip'].value_counts().head(10).to_string()}")
    print(f"\n    Destination IP distribution (top 10):")
    print(f"{df['dst_ip'].value_counts().head(10).to_string()}")

if __name__ == "__main__":
    main()
