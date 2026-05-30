# 🛡️ Real-Time Intrusion Detection System

BiLSTM-Powered Network Security — Detect Threats as They Happen

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Dataset](https://img.shields.io/badge/Dataset-NSL--KDD-00C851?style=for-the-badge)](https://www.unb.ca/cic/datasets/nsl.html)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

---

## 📌 Overview

> A production-grade **Intrusion Detection System (IDS)** that combines real-time packet sniffing with a **Bidirectional LSTM (BiLSTM)** deep learning model to classify live network traffic as **normal** or **malicious** — in real time.

Built on the gold-standard **NSL-KDD** dataset, this system bridges the gap between academic ML research and practical network defense. Instead of analyzing log files after the fact, it hooks directly into live traffic and makes instant predictions — making it suitable for deployment in security operations centers, research labs, or personal network monitoring.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔴 **Real-Time Detection** | Captures and analyzes live packets using `tshark` (Wireshark CLI) |
| 🧠 **BiLSTM Architecture** | Bidirectional LSTM captures temporal patterns in both directions for superior accuracy |
| 🗂️ **Multi-Class Classification** | Detects multiple attack categories — DoS, Probe, R2L, U2R, and Normal |
| ⚡ **Efficient Preprocessing** | Pickled column schema ensures fast, consistent feature alignment at inference time |
| 🔄 **Model Compatibility Tools** | Includes utilities to convert and fix model versions across TF/Keras releases |
| 📝 **Alert Logging** | Detected threats are logged in `ids_alerts.jsonl` for audit and analysis |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      NETWORK INTERFACE                          │
└─────────────────────────┬───────────────────────────────────────┘
                          │  Live Packets
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   tshark / Wireshark                            │
│              (Packet Capture & Feature Extraction)              │
└─────────────────────────┬───────────────────────────────────────┘
                          │  Raw Features
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Preprocessing Pipeline                         │
│         Encoding  →  Normalization  →  Column Alignment         │
│                    (columns.pkl schema)                         │
└─────────────────────────┬───────────────────────────────────────┘
                          │  Feature Vector
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│               BiLSTM Deep Learning Model                        │
│                                                                 │
│   Input → [BiLSTM Layer] → [BiLSTM Layer] → [Dense] → Softmax  │
│                ↑ Forward pass                                   │
│                ↓ Backward pass                                  │
└─────────────────────────┬───────────────────────────────────────┘
                          │  Prediction
                          ▼
            ┌─────────────────────────┐
            │  Normal  /  Attack Type  │
            │   + Alert Logging        │
            └─────────────────────────┘
```

---

## 📁 Project Structure

```
real-time-intrusion-detection-system/
│
├── 📄 realtime_ids.py             # 🚀 Main script — runs real-time IDS
├── 📄 convert_model.py            # 🔧 Convert model between formats
├── 📄 fix_model_compatibility.py  # 🛠️ Fix Keras version compatibility issues
├── 📄 fix_columns_pkl.py          # 🛠️ Repair/rebuild the columns schema
│
├── 🤖 bilstm_ids.h5               # Trained BiLSTM model (HDF5)
├── 🤖 bilstm.weights.h5           # Model weights only
├── 🤖 fixed_model.keras           # Keras-native format (compatibility fixed)
│
├── 📦 columns.pkl                 # Feature column schema for preprocessing
├── 📋 ids_alerts.jsonl            # Alert log (written at runtime)
│
└── 📖 README.md                   # You are here
```

---

## 🧠 Model Details

### Bidirectional LSTM (BiLSTM)

A standard LSTM processes sequences in one direction (past → future). A **BiLSTM** runs two LSTMs in parallel:

- ➡️ **Forward LSTM** — reads the sequence left to right
- ⬅️ **Backward LSTM** — reads the sequence right to left

By combining both directions, the model gains richer context about each time step, which is particularly powerful for network traffic where attack signatures may span packets in complex, non-linear ways.

### Training Dataset — NSL-KDD

The **NSL-KDD** dataset is an improved version of the classic KDD Cup 1999 dataset, correcting for duplicate records and class imbalance. It contains labeled network connection records across:

| Label | Attack Category | Examples |
|---|---|---|
| `Normal` | Benign traffic | Regular HTTP, DNS, SSH sessions |
| `DoS` | Denial of Service | Neptune, Smurf, Pod, Teardrop |
| `Probe` | Reconnaissance | Satan, IPsweep, Portsweep, Nmap |
| `R2L` | Remote to Local | Guess_passwd, FTP_write, Imap |
| `U2R` | User to Root | Buffer_overflow, Rootkit, Perl |

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- [Wireshark / tshark](https://www.wireshark.org/download.html) installed and accessible in `PATH`
- Sufficient permissions to capture network packets (may require `sudo` on Linux/macOS)

### 1. Clone the Repository

```bash
git clone https://github.com/harshithhraya/real-time-intrusion-detection-system.git
cd real-time-intrusion-detection-system
```

### 2. Install Python Dependencies

```bash
pip install tensorflow numpy pandas scikit-learn
```

### 3. Verify tshark

```bash
tshark --version
```

> **Linux users:** You may need `sudo` or add your user to the `wireshark` group:
> ```bash
> sudo usermod -aG wireshark $USER
> ```

### 4. Fix Model Compatibility (if needed)

If you encounter Keras/TensorFlow version mismatch errors:

```bash
python fix_model_compatibility.py
python fix_columns_pkl.py
```

---

## 🚀 Usage

### Run the Real-Time IDS

```bash
python realtime_ids.py
```

The script will:
1. Start capturing live packets from your default network interface via `tshark`
2. Preprocess each packet's features using the saved `columns.pkl` schema
3. Run the BiLSTM model to classify traffic
4. Print predictions to the console and log alerts to `ids_alerts.jsonl`

### Convert Model Format

```bash
python convert_model.py
```

---

## 📊 Results

| Metric | Value |
|---|---|
| **Dataset** | NSL-KDD (125,973 train / 22,544 test records) |
| **Model** | 1D-CNN + BiLSTM × 2 |
| **Overall Accuracy** | 99.16% |
| **Misclassification Rate** | 0.84% |
| **False Positive Rate** | 0.83% |
| **DoS Detection Rate** | 99.84% |
| **Probe Detection Rate** | 98.98% |
| **R2L Detection Rate** | 91.73% |
| **U2R Detection Rate** | 36.84% *(limited by only 52 training examples in dataset)* |
| **Inference Mode** | Real-time (live packet capture via TShark) |

> 💡 **Tip:** Run `python realtime_ids.py` on a network with known test traffic (e.g., using tools like `hping3` or `nmap` in a lab environment) to validate real-time detection performance.

---

## 🔭 Roadmap & Future Improvements

- [ ] 📊 **Web Dashboard** — Real-time attack visualization with charts and live alert feed
- [ ] 🎯 **Higher Accuracy** — Experiment with attention mechanisms and hybrid CNN-BiLSTM architectures
- [ ] ⚡ **Lower Latency** — Optimize the preprocessing pipeline for sub-millisecond inference
- [ ] 🐳 **Docker Support** — Containerize the entire system for easy deployment
- [ ] 📡 **PCAP Replay Mode** — Test the model against pre-recorded `.pcap` files
- [ ] 🔔 **Alerting Integrations** — Push alerts to Slack, email, or SIEM systems

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the model, add features, or fix bugs:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

---

## 👥 About

This project was developed by students of the **Department of Information Science & Engineering, BMS College of Engineering (BMSCE), Bengaluru** as part of an academic research initiative in the domain of network security and deep learning.

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute it.

---
