# AI Anomaly Detection — Network Traffic Classifier

**Course Project:** AI-Enhanced Secure Enterprise Network — Phase 3  
**Department:** Computer Science, GIKI  

---

## Overview

This module implements a machine-learning-based network anomaly detection system. It generates synthetic network traffic, trains classification models, and runs a real-time detection pipeline with a live dashboard — all modelled on the topology defined in the accompanying Packet Tracer file.

---

## Folder Structure

```
ai_anomaly_detection/
│
├── data/
│   ├── generate_traffic.py     # Generates synthetic labelled traffic dataset
│   └── traffic_dataset.csv     # Pre-generated dataset (features + labels)
│
├── models/
│   ├── train_model.py          # Trains Random Forest, Isolation Forest, One-Class SVM
│   ├── random_forest.pkl       # Saved Random Forest classifier
│   ├── isolation_forest.pkl    # Saved Isolation Forest (unsupervised)
│   ├── oneclasssvm.pkl         # Saved One-Class SVM
│   ├── scaler.pkl              # Feature scaler
│   └── label_encoder.pkl       # Label encoder for traffic classes
│
├── detection/
│   ├── realtime_detector.py    # Reads live/pcap traffic and classifies it
│   ├── pcap_to_features.py     # Converts .pcap files into feature vectors
│   ├── dashboard.py            # Static results dashboard
│   └── live_dashboard.py       # Real-time live monitoring dashboard
│
├── reports/
│   ├── alert_log.csv           # Log of detected anomalies
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── roc_curves.png
│   └── dashboard.png
│
├── Packet Tracer/              # Network topology (.pkt file)
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1 — Generate Training Data
```bash
python data/generate_traffic.py
```
Produces `data/traffic_dataset.csv` with labelled traffic samples (Normal, Port Scan, DDoS, etc.).

### 2 — Train Models
```bash
python models/train_model.py
```
Trains and saves Random Forest, Isolation Forest, and One-Class SVM to `models/`.

### 3 — Run Real-Time Detection
```bash
python detection/realtime_detector.py
```
Classifies incoming traffic and logs alerts to `reports/alert_log.csv`.

### 4 — Launch Live Dashboard
```bash
python detection/live_dashboard.py
```
Opens a real-time monitoring dashboard showing traffic classifications and anomaly alerts.

---

## Models

| Model | Type | Purpose |
|---|---|---|
| Random Forest | Supervised | Multi-class traffic classification |
| Isolation Forest | Unsupervised | Outlier / novel anomaly detection |
| One-Class SVM | Unsupervised | Boundary-based anomaly detection |

---

## Results

All outputs are saved to `reports/`:
- **confusion_matrix.png** — classification accuracy per traffic class
- **roc_curves.png** — ROC curves for each model
- **feature_importance.png** — top features driving the Random Forest
- **alert_log.csv** — timestamped anomaly alerts from the live detector
