# Real-Time Intrusion Detection System using BiLSTM

## Overview

This project implements a real-time intrusion detection system using a BiLSTM deep learning model trained on the NSL-KDD dataset. It analyzes live network traffic and classifies it as normal or malicious.

## Features

* Real-time network traffic capture using Wireshark/tshark
* Deep learning-based attack detection using BiLSTM
* Classification of multiple attack categories
* Efficient preprocessing and feature handling

## Tech Stack

* Python
* TensorFlow / Keras
* Wireshark (tshark)
* NSL-KDD Dataset

## Project Structure

* `realtime_ids.py` → Main real-time detection script
* `convert_model.py` → Model conversion utilities
* `fix_model_compatibility.py` → Compatibility fixes
* `columns.pkl` → Feature columns used for prediction
* `bilstm_ids.h5` → Trained model

## How It Works

1. Capture live network traffic using tshark
2. Extract and preprocess features
3. Feed processed data into BiLSTM model
4. Predict whether traffic is normal or an attack

## How to Run

1. Install dependencies:

```bash
pip install tensorflow numpy pandas
```

2. Run the project:

```bash
python realtime_ids.py
```
## Results

- Achieved ~XX% accuracy on NSL-KDD dataset  
- Successfully detected multiple attack types in real-time traffic  

## Future Improvements

* Improve model accuracy
* Deploy as a web-based dashboard
* Optimize for lower latency

## Author

Harshith H
