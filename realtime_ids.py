"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          Real-Time Intrusion Detection System — BiLSTM / NSL-KDD            ║
║  Model   : Conv1D → MaxPool → BatchNorm → BiLSTM × 2 → Dense(5)            ║
║  Classes : normal | dos | probe | r2l | u2r                                 ║
║  Input   : Live packets via tshark → 122-feature NSL-KDD vector             ║
╚══════════════════════════════════════════════════════════════════════════════╝

REQUIREMENTS
  pip install tensorflow joblib numpy scapy colorama

USAGE
  python realtime_ids.py                         # uses defaults in CONFIG
  python realtime_ids.py --iface 5               # Windows interface index
  python realtime_ids.py --iface eth0            # Linux interface name
  python realtime_ids.py --model my_model.keras  # override model path
  python realtime_ids.py --list-ifaces           # show available interfaces
"""

import argparse
import subprocess
import sys
import os
import time
import logging
import json
from collections  import defaultdict, deque
from datetime     import datetime
from pathlib      import Path

import numpy  as np
import joblib

# ── Optional colorama (graceful degradation on plain terminals) ──────────────
try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init(autoreset=True)
    _HAS_COLOR = True
except ImportError:
    _HAS_COLOR = False
    class _Stub:
        def __getattr__(self, _): return ""
    Fore = Style = _Stub()

# ── TensorFlow (loaded lazily so --list-ifaces works without GPU) ────────────
tf = None

# ════════════════════════════════════════════════════════════════════════════
#  CONFIG  — edit these defaults; CLI flags override them
# ════════════════════════════════════════════════════════════════════════════
CONFIG = dict(
    model_path   = "bilstm_ids.h5",      # .h5 or .keras
    columns_path = "columns.pkl",
    interface    = "5",                   # tshark interface index or name
    log_file     = "ids_log.txt",
    alert_file   = "ids_alerts.jsonl",   # one JSON object per alert line
    confidence_threshold = 0.55,         # below this → shown as LOW-CONF
    window_slide = 30,                   # slide window by N packets (overlap)
    tshark_path  = r"C:\Program Files\Wireshark\tshark.exe",   # Windows
    # tshark_path = "tshark",            # Linux / macOS
)

LABELS = ["DOS", "NORMAL", "PROBE", "R2L", "U2R"]

LABEL_COLOR = {
    "NORMAL" : Fore.GREEN,
    "DOS"    : Fore.RED,
    "PROBE"  : Fore.YELLOW,
    "R2L"    : Fore.MAGENTA,
    "U2R"    : Fore.CYAN,
}

# ════════════════════════════════════════════════════════════════════════════
#  NSL-KDD SERVICE MAP  — maps well-known ports → NSL-KDD service label
# ════════════════════════════════════════════════════════════════════════════
PORT_TO_SERVICE = {
    21  : "ftp",        22  : "ssh",       23  : "telnet",
    25  : "smtp",       37  : "time",      43  : "whois",
    53  : "domain",     69  : "tftp_u",    70  : "gopher",
    79  : "finger",     80  : "http",      88  : "klogin",
    109 : "pop_2",      110 : "pop_3",     111 : "sunrpc",
    113 : "auth",       119 : "nntp",      123 : "ntp_u",
    137 : "netbios_ns", 138 : "netbios_dgm", 139: "netbios_ssn",
    143 : "imap4",      161 : "other",     179 : "bgp",
    194 : "IRC",        389 : "ldap",      443 : "http_443",
    445 : "netbios_ssn",512 : "exec",      513 : "login",
    514 : "shell",      515 : "printer",   540 : "uucp",
    543 : "klogin",     544 : "kshell",    587 : "smtp",
    631 : "printer",    636 : "ldap",      993 : "imap4",
    995 : "pop_3",      1080: "other",     3306: "sql_net",
    5190: "aol",        8001: "http_8001", 8080: "http_2784",
}

# NSL-KDD TCP flag labels (hex → KDD flag name)
TCP_FLAGS_MAP = {
    0x002: "S0",    # SYN only
    0x012: "S1",    # SYN + ACK  (no FIN yet from client)
    0x011: "SF",    # FIN + ACK  → completed
    0x001: "SH",    # FIN only
    0x004: "RSTO",  # RST from originator
    0x014: "RSTR",  # RST from responder
    0x010: "SF",    # ACK only → treat as established/SF
    0x018: "SF",    # PSH + ACK
    0x000: "OTH",
}

def tcp_flags_to_kdd(raw_flags: str) -> str:
    """Convert tshark hex TCP flags string to NSL-KDD flag name."""
    try:
        val = int(raw_flags, 16) & 0x03F   # mask lower 6 bits
        # Check RST
        if val & 0x004:
            if val & 0x010: return "RSTR"
            return "RSTO"
        # Check FIN
        if val & 0x001:
            if val & 0x010: return "SF"
            return "SH"
        # Check SYN
        if val & 0x002:
            if val & 0x010: return "S1"
            return "S0"
        if val & 0x010: return "SF"   # ACK / PSH+ACK
        return "OTH"
    except (ValueError, TypeError):
        return "OTH"

def proto_to_kdd(ip_proto: str) -> str:
    mapping = {"6": "tcp", "17": "udp", "1": "icmp"}
    return mapping.get(str(ip_proto).strip(), "tcp")

def port_to_service(port: str, proto: str) -> str:
    try:
        p = int(port)
        if proto == "udp" and p == 53:  return "domain_u"
        if proto == "udp" and p == 69:  return "tftp_u"
        return PORT_TO_SERVICE.get(p, "other")
    except (ValueError, TypeError):
        return "other"

# ════════════════════════════════════════════════════════════════════════════
#  FEATURE VECTOR BUILDER
# ════════════════════════════════════════════════════════════════════════════

class FeatureBuilder:
    """
    Converts a parsed tshark line into an NSL-KDD-aligned 122-element vector.

    NSL-KDD columns breakdown:
      [0-18]   : numeric / binary features
      [19-37]  : traffic statistics (computed over time window)
      [38-40]  : protocol_type one-hot  (icmp, tcp, udp)
      [41-110] : service one-hot        (70 service categories)
      [111-121]: flag one-hot           (11 TCP flag states)
    """

    # All 70 service names (columns 41-110 in the pkl)
    ALL_SERVICES = [
        "IRC","X11","Z39_50","aol","auth","bgp","courier","csnet_ns","ctf",
        "daytime","discard","domain","domain_u","echo","eco_i","ecr_i","efs",
        "exec","finger","ftp","ftp_data","gopher","harvest","hostnames","http",
        "http_2784","http_443","http_8001","imap4","iso_tsap","klogin","kshell",
        "ldap","link","login","mtp","name","netbios_dgm","netbios_ns",
        "netbios_ssn","netstat","nnsp","nntp","ntp_u","other","pm_dump",
        "pop_2","pop_3","printer","private","red_i","remote_job","rje","shell",
        "smtp","sql_net","ssh","sunrpc","supdup","systat","telnet","tftp_u",
        "tim_i","time","urh_i","urp_i","uucp","uucp_path","vmnet","whois",
    ]

    # All 11 flag names (columns 111-121)
    ALL_FLAGS = ["OTH","REJ","RSTO","RSTOS0","RSTR","S0","S1","S2","S3","SF","SH"]

    WINDOW   = 100   # rolling window for rate stats
    DST_WIN  = 100   # dst-host window

    def __init__(self, columns: list):
        self.columns = list(columns)
        assert len(self.columns) == 122, f"Expected 122 columns, got {len(self.columns)}"

        # Rolling buffers for traffic-based statistics
        self._recent       = deque(maxlen=self.WINDOW)   # (proto, service, flag, dst_ip, src_port, dst_port, has_err)
        self._dst_recent   = defaultdict(lambda: deque(maxlen=self.DST_WIN))

    def _compute_stats(self, proto, service, flag, dst_ip, src_port):
        """
        Compute NSL-KDD traffic statistics from recent packet history.
        Returns dict with all 19 stat features (columns 19-37).
        """
        hist = list(self._recent)
        n = len(hist) or 1

        # count  : connections to same dst in last window
        # srv_count : connections to same service in last window
        same_dst  = sum(1 for r in hist if r["dst_ip"] == dst_ip)
        same_srv  = sum(1 for r in hist if r["service"] == service)
        count     = max(same_dst, 1)
        srv_count = max(same_srv, 1)

        # Error rates (S0 / REJ flags = SYN errors; RSTR/RSTO = RST errors)
        syn_err_flags = {"S0", "S1", "S2", "S3"}
        rst_err_flags = {"RSTO", "RSTOS0", "RSTR"}

        serr_dst = sum(1 for r in hist if r["dst_ip"]==dst_ip and r["flag"] in syn_err_flags)
        rerr_dst = sum(1 for r in hist if r["dst_ip"]==dst_ip and r["flag"] in rst_err_flags)
        serr_srv = sum(1 for r in hist if r["service"]==service and r["flag"] in syn_err_flags)
        rerr_srv = sum(1 for r in hist if r["service"]==service and r["flag"] in rst_err_flags)
        diff_srv = len({r["service"] for r in hist if r["dst_ip"]==dst_ip})
        same_port= sum(1 for r in hist if r["dst_ip"]==dst_ip and r["src_port"]==src_port)
        diff_host_srv = len({r["dst_ip"] for r in hist if r["service"]==service})

        # dst_host window (per destination ip)
        dh = list(self._dst_recent[dst_ip])
        dh_n = len(dh) or 1
        dh_same_srv  = sum(1 for r in dh if r["service"]==service)
        dh_diff_srv  = len({r["service"] for r in dh})
        dh_same_port = sum(1 for r in dh if r["src_port"]==src_port)
        dh_diff_host = len({r["dst_ip"] for r in dh})   # always 1 — placeholder
        dh_serr      = sum(1 for r in dh if r["flag"] in syn_err_flags)
        dh_rerr      = sum(1 for r in dh if r["flag"] in rst_err_flags)
        dh_serr_srv  = sum(1 for r in dh if r["flag"] in syn_err_flags and r["service"]==service)
        dh_rerr_srv  = sum(1 for r in dh if r["flag"] in rst_err_flags and r["service"]==service)

        return {
            "count"                     : float(count),
            "srv_count"                 : float(srv_count),
            "serror_rate"               : serr_dst / count,
            "srv_serror_rate"           : serr_srv / srv_count,
            "rerror_rate"               : rerr_dst / count,
            "srv_rerror_rate"           : rerr_srv / srv_count,
            "same_srv_rate"             : same_srv  / count,
            "diff_srv_rate"             : diff_srv  / count,
            "srv_diff_host_rate"        : diff_host_srv / srv_count,
            "dst_host_count"            : float(dh_n),
            "dst_host_srv_count"        : float(dh_same_srv),
            "dst_host_same_srv_rate"    : dh_same_srv  / dh_n,
            "dst_host_diff_srv_rate"    : dh_diff_srv  / dh_n,
            "dst_host_same_src_port_rate": dh_same_port / dh_n,
            "dst_host_srv_diff_host_rate": dh_diff_host / dh_n,
            "dst_host_serror_rate"      : dh_serr     / dh_n,
            "dst_host_srv_serror_rate"  : dh_serr_srv / dh_n,
            "dst_host_rerror_rate"      : dh_rerr     / dh_n,
            "dst_host_srv_rerror_rate"  : dh_rerr_srv / dh_n,
        }

    def packet_to_vector(self, raw_fields: dict) -> np.ndarray:
        """
        raw_fields keys: frame_len, ip_proto, tcp_flags, src_port, dst_port,
                         dst_ip, src_ip, ip_ttl, tcp_stream, udp_len
        Returns numpy array of shape (122,) aligned to columns.pkl order.
        """
        # ── Parse raw fields ─────────────────────────────────────────────
        def flt(k, default=0.0):
            v = raw_fields.get(k, "")
            try:    return float(v) if v else default
            except: return default

        frame_len  = flt("frame_len")
        ip_proto_s = str(raw_fields.get("ip_proto","6")).strip()
        tcp_flags_s= str(raw_fields.get("tcp_flags","0x010")).strip()
        src_port_s = str(raw_fields.get("src_port","0")).strip()
        dst_port_s = str(raw_fields.get("dst_port","0")).strip()
        dst_ip     = str(raw_fields.get("dst_ip","0.0.0.0")).strip()
        src_ip     = str(raw_fields.get("src_ip","0.0.0.0")).strip()

        proto   = proto_to_kdd(ip_proto_s)
        service = port_to_service(dst_port_s, proto)
        flag    = tcp_flags_to_kdd(tcp_flags_s) if proto == "tcp" else "SF"

        try:  src_port = int(src_port_s)
        except: src_port = 0

        # ── Base numeric features (columns 0-18) ────────────────────────
        land          = 1 if (src_ip == dst_ip and src_port_s == dst_port_s) else 0
        logged_in     = 1 if flag == "SF" else 0
        syn_err_flags = {"S0","S1","S2","S3"}
        wrong_fragment= 1 if flag in {"RSTOS0","RSTO"} else 0

        numeric = {
            "duration"          : 0.0,          # unknown in packet-level capture
            "src_bytes"         : frame_len,
            "dst_bytes"         : 0.0,           # can't know from single packet
            "land"              : float(land),
            "wrong_fragment"    : float(wrong_fragment),
            "urgent"            : 0.0,
            "hot"               : 0.0,
            "num_failed_logins" : 0.0,
            "logged_in"         : float(logged_in),
            "num_compromised"   : 0.0,
            "root_shell"        : 0.0,
            "su_attempted"      : 0.0,
            "num_root"          : 0.0,
            "num_file_creations": 0.0,
            "num_shells"        : 0.0,
            "num_access_files"  : 0.0,
            "num_outbound_cmds" : 0.0,
            "is_host_login"     : 0.0,
            "is_guest_login"    : 0.0,
        }

        # ── Traffic statistics (columns 19-37) ──────────────────────────
        stats = self._compute_stats(proto, service, flag, dst_ip, src_port)

        # ── Update rolling history ───────────────────────────────────────
        record = dict(proto=proto, service=service, flag=flag,
                      dst_ip=dst_ip, src_port=src_port)
        self._recent.append(record)
        self._dst_recent[dst_ip].append(record)

        # ── One-hot: protocol_type (cols 38-40) ─────────────────────────
        proto_oh = {
            "protocol_type_icmp": 1.0 if proto=="icmp" else 0.0,
            "protocol_type_tcp" : 1.0 if proto=="tcp"  else 0.0,
            "protocol_type_udp" : 1.0 if proto=="udp"  else 0.0,
        }

        # ── One-hot: service (cols 41-110) ──────────────────────────────
        svc_key = service.replace("-", "_")
        service_oh = {f"service_{s}": (1.0 if s==svc_key else 0.0)
                      for s in self.ALL_SERVICES}

        # ── One-hot: flag (cols 111-121) ────────────────────────────────
        flag_oh = {f"flag_{f}": (1.0 if f==flag else 0.0)
                   for f in self.ALL_FLAGS}

        # ── Assemble full feature dict ───────────────────────────────────
        feat = {}
        feat.update(numeric)
        feat.update(stats)
        feat.update(proto_oh)
        feat.update(service_oh)
        feat.update(flag_oh)

        # ── Build vector in exact columns.pkl order ──────────────────────
        vec = np.array([feat.get(col, 0.0) for col in self.columns], dtype=np.float32)
        return vec   # shape: (122,)


# ════════════════════════════════════════════════════════════════════════════
#  TSHARK CAPTURE
# ════════════════════════════════════════════════════════════════════════════

TSHARK_FIELDS = [
    "frame.len",      # 0
    "ip.proto",       # 1
    "tcp.flags",      # 2
    "tcp.srcport",    # 3
    "tcp.dstport",    # 4
    "ip.src",         # 5
    "ip.dst",         # 6
    "ip.ttl",         # 7
    "tcp.stream",     # 8
    "udp.length",     # 9
    "udp.srcport",    # 10
    "udp.dstport",    # 11
]

FIELD_KEYS = [
    "frame_len","ip_proto","tcp_flags","src_port","dst_port",
    "src_ip","dst_ip","ip_ttl","tcp_stream","udp_len",
    "udp_src_port","udp_dst_port",
]

def build_tshark_cmd(interface: str, tshark_path: str) -> list:
    cmd = [tshark_path, "-i", interface, "-l",
           "-T", "fields",
           "-E", "separator=|",
           "-E", "occurrence=f",
           "-E", "quote=n"]
    for f in TSHARK_FIELDS:
        cmd += ["-e", f]
    return cmd

def parse_tshark_line(line: str) -> dict:
    parts = line.strip().split("|")
    raw = {}
    for i, key in enumerate(FIELD_KEYS):
        raw[key] = parts[i].strip() if i < len(parts) else ""

    # Unify TCP/UDP ports
    if not raw.get("src_port") and raw.get("udp_src_port"):
        raw["src_port"] = raw["udp_src_port"]
    if not raw.get("dst_port") and raw.get("udp_dst_port"):
        raw["dst_port"] = raw["udp_dst_port"]
    return raw

def list_interfaces(tshark_path: str):
    """Print available tshark interfaces and exit."""
    try:
        result = subprocess.run([tshark_path, "-D"],
                                capture_output=True, text=True, timeout=10)
        print("\nAvailable interfaces:\n")
        print(result.stdout or result.stderr)
    except FileNotFoundError:
        print(f"[ERROR] tshark not found at: {tshark_path}")
    sys.exit(0)


# ════════════════════════════════════════════════════════════════════════════
#  LOGGING SETUP
# ════════════════════════════════════════════════════════════════════════════

def setup_logging(log_file: str):
    logger = logging.getLogger("IDS")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def write_alert(alert_file: str, payload: dict):
    """Append one JSON alert to the JSONL alert file."""
    try:
        with open(alert_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ════════════════════════════════════════════════════════════════════════════

def print_banner():
    print(Fore.CYAN + r"""
  ╔══════════════════════════════════════════════════════════╗
  ║    BiLSTM Real-Time Intrusion Detection System          ║
  ║    NSL-KDD • Conv1D + BiLSTM × 2 • 5-Class             ║
  ╚══════════════════════════════════════════════════════════╝
""" + Style.RESET_ALL)

STATS = defaultdict(int)

def print_prediction(label: str, conf: float, pkt_count: int,
                     low_conf: bool, logger):
    color  = LABEL_COLOR.get(label, "")
    badge  = f"[{label:6s}]"
    conf_s = f"{conf*100:5.1f}%"
    lc     = " ⚠ LOW-CONF" if low_conf else ""
    line   = f"  Pkts:{pkt_count:>6}  {color}{badge}{Style.RESET_ALL}  conf={conf_s}{lc}"
    print(line)

    if label != "NORMAL":
        alert_color = LABEL_COLOR.get(label, Fore.RED)
        print(alert_color +
              f"  ╔══ ⚠  ALERT: {label} DETECTED  (conf={conf_s}) ══╗" +
              Style.RESET_ALL)
    STATS[label] += 1

def print_stats():
    print(Fore.WHITE + "\n─── Session Summary ───────────────────────────────")
    for label in LABELS:
        cnt = STATS[label]
        bar = "█" * min(cnt, 40)
        c   = LABEL_COLOR.get(label, "")
        print(f"  {c}{label:8s}{Style.RESET_ALL}  {bar}  {cnt}")
    print("───────────────────────────────────────────────────\n")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN IDS LOOP
# ════════════════════════════════════════════════════════════════════════════

def run_ids(cfg: dict):
    global tf
    import tensorflow as _tf
    tf = _tf

    logger = setup_logging(cfg["log_file"])
    print_banner()

    # ── Load model ───────────────────────────────────────────────────────
    logger.info(f"Loading model: {cfg['model_path']}")
    try:
        model = tf.keras.models.load_model(cfg["model_path"])
        logger.info("Model loaded OK — input shape: %s", model.input_shape)
    except Exception as e:
        logger.critical("Failed to load model: %s", e)
        sys.exit(1)

    # ── Load columns ─────────────────────────────────────────────────────
    logger.info(f"Loading columns: {cfg['columns_path']}")
    try:
        columns = joblib.load(cfg["columns_path"])
        assert len(columns) == 122
        logger.info("Columns loaded: %d features", len(columns))
    except Exception as e:
        logger.critical("Failed to load columns.pkl: %s", e)
        sys.exit(1)

    builder     = FeatureBuilder(list(columns))
    WINDOW_SIZE = 122
    SLIDE       = cfg["window_slide"]
    THRESH      = cfg["confidence_threshold"]

    # ── Start tshark ─────────────────────────────────────────────────────
    cmd = build_tshark_cmd(cfg["interface"], cfg["tshark_path"])
    logger.info("Starting tshark on interface '%s'", cfg["interface"])
    logger.info("Command: %s", " ".join(cmd))

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
    except FileNotFoundError:
        logger.critical("tshark not found at: %s", cfg["tshark_path"])
        logger.critical("Run with --list-ifaces to verify path or set tshark_path in CONFIG.")
        sys.exit(1)

    logger.info("Listening... (Ctrl-C to stop)\n")

    buffer    = []
    pkt_count = 0

    try:
        while True:
            line = proc.stdout.readline()

            if not line:
                if proc.poll() is not None:
                    logger.error("tshark exited unexpectedly (code %d)", proc.returncode)
                    stderr_out = proc.stderr.read().decode(errors="replace")
                    if stderr_out:
                        logger.error("tshark stderr: %s", stderr_out[:500])
                    break
                time.sleep(0.01)
                continue

            raw_text = line.decode("utf-8", errors="replace").strip()
            if not raw_text:
                continue

            # ── Extract features for this packet ─────────────────────
            try:
                raw_fields = parse_tshark_line(raw_text)
                vec        = builder.packet_to_vector(raw_fields)   # (122,)
                buffer.append(vec)
                pkt_count += 1
            except Exception as e:
                logger.debug("Feature extraction error: %s | raw=%s", e, raw_text[:80])
                continue

                       # ── Predict when buffer is full ───────────────────────────
            if len(buffer) >= WINDOW_SIZE:
                window = np.stack(buffer[-WINDOW_SIZE:], axis=0)
                window = window.mean(axis=1)
                X = window.reshape(1, WINDOW_SIZE, 1)      # (1, 122, 1)

                # Each row = one packet's 122-feature vector;
                # the model sees a time-series of 122 packet-feature scalars
                # (identical to training where 122 values form the time axis)

                try:
                    preds    = model.predict(X, verbose=0)[0]      # (5,)
                    conf     = float(np.max(preds))
                    label    = LABELS[int(np.argmax(preds))]
                    low_conf = conf < THRESH
                    if conf < 0.40:
                        label = "NORMAL"

                    print_prediction(label, conf, pkt_count, low_conf, logger)
                    logger.debug("Probs: %s", dict(zip(LABELS, preds.round(3).tolist())))
                    

                    # ── Write alert for non-normal events ────────────
                    if label != "NORMAL":
                        payload = {
                            "ts"        : datetime.now().isoformat(),
                            "label"     : label,
                            "confidence": round(conf, 4),
                            "low_conf"  : low_conf,
                            "pkt_count" : pkt_count,
                            "probs"     : dict(zip(LABELS, preds.round(4).tolist())),
                        }
                        write_alert(cfg["alert_file"], payload)
                        logger.warning("ALERT %s conf=%.2f%%", label, conf*100)

                except Exception as e:
                    logger.error("Prediction error: %s", e)

                # ── Slide window ──────────────────────────────────────
                buffer = buffer[SLIDE:]

    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        proc.terminate()
        print_stats()


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Real-Time BiLSTM Intrusion Detection System (NSL-KDD)")
    ap.add_argument("--iface",       default=CONFIG["interface"],
                    help="tshark interface index or name  [default: %(default)s]")
    ap.add_argument("--model",       default=CONFIG["model_path"],
                    help="Path to .h5 or .keras model      [default: %(default)s]")
    ap.add_argument("--columns",     default=CONFIG["columns_path"],
                    help="Path to columns.pkl              [default: %(default)s]")
    ap.add_argument("--log",         default=CONFIG["log_file"],
                    help="Log file path                    [default: %(default)s]")
    ap.add_argument("--alerts",      default=CONFIG["alert_file"],
                    help="JSONL alert output file          [default: %(default)s]")
    ap.add_argument("--threshold",   type=float, default=CONFIG["confidence_threshold"],
                    help="Confidence threshold for LOW-CONF[default: %(default)s]")
    ap.add_argument("--slide",       type=int,   default=CONFIG["window_slide"],
                    help="Window slide amount (overlap)    [default: %(default)s]")
    ap.add_argument("--tshark",      default=CONFIG["tshark_path"],
                    help="Full path to tshark executable   [default: %(default)s]")
    ap.add_argument("--list-ifaces", action="store_true",
                    help="List available tshark interfaces and exit")
    args = ap.parse_args()

    if args.list_ifaces:
        list_interfaces(args.tshark)

    cfg = dict(
        interface            = args.iface,
        model_path           = args.model,
        columns_path         = args.columns,
        log_file             = args.log,
        alert_file           = args.alerts,
        confidence_threshold = args.threshold,
        window_slide         = args.slide,
        tshark_path          = args.tshark,
    )
    run_ids(cfg)

if __name__ == "__main__":
    main()