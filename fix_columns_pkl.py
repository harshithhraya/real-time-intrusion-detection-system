"""
fix_columns_pkl.py
──────────────────
Regenerates columns.pkl from the known NSL-KDD 122-feature list.
Fixes: "No module named 'numpy._core'" (NumPy 2.x pickle vs NumPy 1.x)

Run once:
    py -3.10 fix_columns_pkl.py

Produces: columns.pkl  (compatible with your installed NumPy)
"""

import joblib
import numpy as np

COLUMNS = [
    # ── Numeric / binary features (0-18) ──────────────────────────────────
    "duration", "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login",

    # ── Traffic statistics (19-37) ─────────────────────────────────────────
    "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",

    # ── protocol_type one-hot (38-40) ──────────────────────────────────────
    "protocol_type_icmp", "protocol_type_tcp", "protocol_type_udp",

    # ── service one-hot (41-110) ───────────────────────────────────────────
    "service_IRC", "service_X11", "service_Z39_50", "service_aol",
    "service_auth", "service_bgp", "service_courier", "service_csnet_ns",
    "service_ctf", "service_daytime", "service_discard", "service_domain",
    "service_domain_u", "service_echo", "service_eco_i", "service_ecr_i",
    "service_efs", "service_exec", "service_finger", "service_ftp",
    "service_ftp_data", "service_gopher", "service_harvest",
    "service_hostnames", "service_http", "service_http_2784",
    "service_http_443", "service_http_8001", "service_imap4",
    "service_iso_tsap", "service_klogin", "service_kshell", "service_ldap",
    "service_link", "service_login", "service_mtp", "service_name",
    "service_netbios_dgm", "service_netbios_ns", "service_netbios_ssn",
    "service_netstat", "service_nnsp", "service_nntp", "service_ntp_u",
    "service_other", "service_pm_dump", "service_pop_2", "service_pop_3",
    "service_printer", "service_private", "service_red_i",
    "service_remote_job", "service_rje", "service_shell", "service_smtp",
    "service_sql_net", "service_ssh", "service_sunrpc", "service_supdup",
    "service_systat", "service_telnet", "service_tftp_u", "service_tim_i",
    "service_time", "service_urh_i", "service_urp_i", "service_uucp",
    "service_uucp_path", "service_vmnet", "service_whois",

    # ── flag one-hot (111-121) ─────────────────────────────────────────────
    "flag_OTH", "flag_REJ", "flag_RSTO", "flag_RSTOS0", "flag_RSTR",
    "flag_S0", "flag_S1", "flag_S2", "flag_S3", "flag_SF", "flag_SH",
]

assert len(COLUMNS) == 122, f"Expected 122, got {len(COLUMNS)}"

# Save as a numpy array (same format as original columns.pkl)
columns_array = np.array(COLUMNS)
joblib.dump(columns_array, "columns.pkl")

print(f"✓ columns.pkl regenerated — {len(COLUMNS)} features")
print(f"  NumPy version : {np.__version__}")
print(f"  First 5 cols  : {COLUMNS[:5]}")
print(f"  Last  5 cols  : {COLUMNS[-5:]}")
print("\nNow run:")
print("  py -3.10 realtime_ids.py --iface 5 --model fixed_model.keras")