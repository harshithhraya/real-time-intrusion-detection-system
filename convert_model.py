import subprocess
import numpy as np
import joblib
import tensorflow as tf
import logging
from datetime import datetime

# ─── CONFIG ────────────────────────────────────────────────────
MODEL_PATH   = "fixed_model.keras"
COLUMNS_PATH = "columns.pkl"
INTERFACE    = "5"          # Change to your tshark interface number/name
WINDOW_SIZE  = 122
LOG_FILE     = "ids_log.txt"
# ───────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.critical(f"Failed to load model: {e}")
    raise

# Load feature columns (for reference / future alignment)
try:
    columns = joblib.load(COLUMNS_PATH)
    logging.info(f"Feature columns loaded: {len(columns)} features")
except Exception as e:
    logging.warning(f"Could not load columns.pkl: {e}")
    columns = None

LABELS = ["DOS", "NORMAL", "PROBE", "R2L", "U2R"]

# tshark command — captures MULTIPLE fields for better accuracy
cmd = [
    r"C:\Program Files\Wireshark\tshark.exe",
    "-i", INTERFACE,
    "-T", "fields",
    "-e", "frame.len",          # packet length
    "-e", "ip.proto",           # IP protocol
    "-e", "tcp.flags",          # TCP flags (SYN, ACK, FIN, RST...)
    "-e", "tcp.srcport",        # source port
    "-e", "tcp.dstport",        # destination port
    "-E", "separator=,",        # comma-separated output
    "-E", "occurrence=f"        # first occurrence of each field
]

try:
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logging.info("Real-time IDS started. Listening on interface %s...\n", INTERFACE)
except FileNotFoundError:
    logging.critical("tshark not found. Check the path in CONFIG.")
    raise

buffer = []

while True:
    line = process.stdout.readline()

    if not line:
        # Check if tshark died
        if process.poll() is not None:
            logging.error("tshark process terminated unexpectedly.")
            break
        continue

    try:
        raw = line.decode().strip()
        if not raw:
            continue

        # Parse comma-separated fields; fill missing with 0
        parts = raw.split(",")
        values = []
        for p in parts:
            try:
                values.append(float(p) if p.strip() else 0.0)
            except ValueError:
                values.append(0.0)

        # Pad or trim to exactly 1 value per packet for the rolling window
        # (use frame.len as primary; extend logic here if model expects multi-feature)
        frame_len = values[0] if values else 0.0
        buffer.append(frame_len)

        # Sliding window — overlap by 50% so no attack is missed at boundaries
        if len(buffer) >= WINDOW_SIZE:
            window = buffer[-WINDOW_SIZE:]          # take last 122 samples
            X = np.array(window, dtype=np.float32).reshape(1, WINDOW_SIZE, 1)

            pred   = model.predict(X, verbose=0)
            conf   = float(np.max(pred) * 100)
            label  = LABELS[int(np.argmax(pred))]

            msg = f"Prediction: {label:8s}  (confidence: {conf:.1f}%)"
            logging.info(msg)

            if label != "NORMAL":
                logging.warning(f"⚠  ATTACK DETECTED: {label}  at {datetime.now().isoformat()}")

            # Slide by half a window instead of clearing completely
            buffer = buffer[WINDOW_SIZE // 2:]

    except KeyboardInterrupt:
        logging.info("IDS stopped by user.")
        break
    except Exception as e:
        logging.error(f"Error processing packet: {e}")
        continue