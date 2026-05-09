# AI-Enhanced Secure Enterprise Network

**Department:** Computer Science, GIKI  

This repository contains the complete implementation for the **AI-Enhanced Secure Enterprise Network** project. It combines a robust Cisco Packet Tracer network architecture with advanced Python-based Machine Learning models to detect and correlate cyber threats in real time.

---

## Project Architecture & Phases

The project was built systematically across four interconnected phases:

### Phase 1 — Network Infrastructure
The foundation of the project. A complete enterprise topology designed in Cisco Packet Tracer (`deez.pkt`) featuring:
- **Core, Distribution, and Access Layers** using Cisco routers and switches.
- **VLAN Segmentation** separating Corporate, Finance, IT/OPS, and Management networks.
- Inter-VLAN routing and standard OSPF/Static routing configurations.

### Phase 2 — Security Implementation (Defense-in-Depth)
Hardening the Phase 1 infrastructure to enterprise standards:
- **Cisco ASA Firewall:** 3-zone architecture (Inside, Outside, DMZ) with strict NAT and access policies.
- **VLAN Access Control Lists (ACLs):** Strict internal isolation (e.g., isolating Finance from Corporate).
- **Intrusion Detection (IDS/IPS):** Snort-style signatures for port scans, DDoS, and brute force attacks.
- **Site-to-Site IPSec VPN:** Secure branch office connectivity.
- **Device Hardening:** AAA (RADIUS) authentication, DHCP Snooping, Dynamic ARP Inspection, and SSH.

### Phase 3 — AI Anomaly Detection (Machine Learning)
A Python-based AI analytics engine that monitors the traffic flowing through the network to catch zero-day and complex threats that bypass standard firewall rules.
- **Models Used:** Random Forest (Supervised), Isolation Forest, and One-Class SVM.
- **Live Detection (`realtime_detector.py`):** Classifies network flows as Normal, DDoS, Port Scan, Brute Force, or Data Exfiltration.
- **Live Web Dashboard (`live_dashboard.py`):** A beautiful web-based GUI running on `localhost:5000` that visualizes attack vectors and anomaly rates in real time.

### Phase 4 — SNMP Monitoring & SIEM Centralized Logging
Converting the raw data into actionable security intelligence.
- **Network Observability:** Syslog, NTP, and SNMPv3 deployed across all Packet Tracer devices, forwarding events to a central Management Server.
- **SIEM Correlation Engine (`siem_dashboard.py`):** A custom Python SIEM that reads the AI ML predictions and correlates them with simulated Syslog events (e.g., matching a Firewall Deny log with an AI Port Scan detection) to assign critical threat severities.

---

## Folder Structure

```
.
├── data/
│   ├── generate_traffic.py     # Generates synthetic labelled traffic dataset
│   └── traffic_dataset.csv     # Pre-generated dataset (features + labels)
│
├── models/
│   ├── train_model.py          # Trains the Machine Learning models
│   ├── random_forest.pkl       # Saved Random Forest classifier
│   └── isolation_forest.pkl    # Saved Isolation Forest (unsupervised)
│
├── detection/
│   ├── realtime_detector.py    # Phase 3: Headless real-time traffic classifier
│   ├── live_dashboard.py       # Phase 3: Interactive web dashboard (localhost:5000)
│   └── siem_dashboard.py       # Phase 4: SIEM terminal correlating AI + Syslog
│
├── reports/
│   ├── alert_log.csv           # Ongoing log of detected anomalies
│   └── (Various graphs)        # Confusion matrices, ROC curves, feature importance
│
├── Packet Tracer/
│   └── deez.pkt                # Phase 1 & 2 network topology
│
└── requirements.txt            # Python dependencies
```

---

## Setup & Installation

**1. Install Python Dependencies:**
```bash
pip install -r requirements.txt
```

**2. Network Simulation:**
Open `Packet Tracer/deez.pkt` in Cisco Packet Tracer 8.x to view the underlying Phase 1 & 2 infrastructure.

---

## Running the Live Demonstration

To launch the full Phase 3 & 4 AI monitoring suite, open three separate terminal windows and run the following commands simultaneously:

**Terminal 1 — The AI Engine**  
Continuously analyzes network traffic and logs threats:
```bash
python detection/realtime_detector.py
```

**Terminal 2 — The Phase 4 SIEM Correlation Dashboard**  
Correlates AI detections with Network Syslog events in a live console:
```bash
python detection/siem_dashboard.py
```

**Terminal 3 — The Web UI**  
Serves the graphical interface visualizing attack trends:
```bash
python detection/live_dashboard.py
```
*(Once running, open `http://localhost:5000` in your web browser)*
