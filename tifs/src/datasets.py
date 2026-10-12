"""
Dataset loaders for the PCSP journal experiments.

Each loader returns a dict with
    Xtr, ytr, Xca, yca           training / calibration (exchangeable)
    Xte, yte                     i.i.d. test fold
    Xsh, ysh (optional)          shifted deployment stream, in time order
    raw_bytes_te / raw_bytes_sh  per-record native evidence size (bytes)
    classes                      class names
and `evid_te` / `evid_sh`: per-record canonical evidence byte strings
(used for commitments and audits).

NSL-KDD: official KDDTrain+ (125,973 rows) and KDDTest+ (22,544 rows).
The i.i.d. folds are drawn from KDDTrain+; KDDTest+ is the shifted
deployment stream (it contains attack types absent from training).

CIC-IDS-2017: by default the corrected release of Liu, Engelen et al.
(IEEE CNS 2022; CICIDS2017_improved.zip, five day files, 2,099,976 rows,
regenerated with the fixed CICFlowMeter and relabeled).  Flows labeled
"X - Attempted" carry no attack behavior and are relabeled benign, as the
dataset authors recommend; "Infiltration - Portscan" (scans launched from
the infiltrated host) maps to portscan.  Flows are put in timestamp order.
Set PCSP_CIC_RELEASE=original for the eight MachineLearningCVE day files
of the original release.  Web attacks, Infiltration, and Botnet form one
rare-attack class in the corrected release (six classes).  The i.i.d.
folds are a random sample per seed;
the temporal stream is Thursday + Friday traffic with the model trained
and calibrated on Monday--Wednesday.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "sim" / "data"
CACHE = DATA / "cache"

# ---------------------------------------------------------------------------
# NSL-KDD
# ---------------------------------------------------------------------------

NSL_COLS = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "attack", "difficulty",
]
NSL_CAT = ["protocol_type", "service", "flag"]
NSL_GROUPS = {
    "dos": ["back", "land", "neptune", "pod", "smurf", "teardrop",
            "apache2", "udpstorm", "processtable", "mailbomb", "worm"],
    "probe": ["ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"],
    "r2l": ["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy",
            "warezclient", "warezmaster", "sendmail", "named",
            "snmpgetattack", "snmpguess", "xlock", "xsnoop", "httptunnel"],
    "u2r": ["buffer_overflow", "loadmodule", "perl", "rootkit", "ps",
            "sqlattack", "xterm"],
}
NSL_CLASSES = ["normal", "dos", "probe", "r2l", "u2r"]
_NSL_MAP = {a: g for g, v in NSL_GROUPS.items() for a in v}
_NSL_MAP["normal"] = "normal"


def _nsl_read(name):
    df = pd.read_csv(DATA / "nslkdd_official" / name, header=None,
                     names=NSL_COLS)
    unk = set(df.attack) - set(_NSL_MAP)
    if unk:
        raise ValueError(f"unmapped NSL-KDD labels: {sorted(unk)}")
    df["y"] = df.attack.map(_NSL_MAP).map(NSL_CLASSES.index)
    return df


def _nsl_native(df):
    """Native evidence record: the 41 feature fields as the dataset's own
    comma-separated text (label and difficulty removed)."""
    feats = df[NSL_COLS[:41]].astype(str)
    return [",".join(r).encode() for r in feats.itertuples(index=False)]


def nslkdd(seed: int = 0, frac=(0.6, 0.2, 0.2)):
    tr = _nsl_read("KDDTrain+.txt")
    te = _nsl_read("KDDTest+.txt")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(tr))
    a, b = int(frac[0] * len(tr)), int((frac[0] + frac[1]) * len(tr))
    p_tr, p_ca, p_te = tr.iloc[idx[:a]], tr.iloc[idx[a:b]], tr.iloc[idx[b:]]
    cats = pd.get_dummies(p_tr[NSL_CAT], columns=NSL_CAT).columns
    num = NSL_COLS[:41]
    num = [c for c in num if c not in NSL_CAT]

    def feat(d):
        c = pd.get_dummies(d[NSL_CAT], columns=NSL_CAT).reindex(
            columns=cats, fill_value=0)
        x = np.concatenate([d[num].to_numpy(np.float64),
                            c.to_numpy(np.float64)], axis=1)
        return x.astype(np.float32)

    out = dict(Xtr=feat(p_tr), ytr=p_tr.y.to_numpy(),
               Xca=feat(p_ca), yca=p_ca.y.to_numpy(),
               Xte=feat(p_te), yte=p_te.y.to_numpy(),
               Xsh=feat(te), ysh=te.y.to_numpy(),
               classes=NSL_CLASSES, name="NSL-KDD")
    out["evid_te"] = _nsl_native(p_te)
    out["evid_sh"] = _nsl_native(te)
    out["raw_bytes_te"] = np.array([len(e) for e in out["evid_te"]])
    out["raw_bytes_sh"] = np.array([len(e) for e in out["evid_sh"]])
    out["n_features"] = len(num) + len(cats)
    return out


# ---------------------------------------------------------------------------
# CIC-IDS-2017
# ---------------------------------------------------------------------------

CIC_CLASSES = ["benign", "dos", "ddos", "portscan", "brute_force",
               "web_attack", "bot_infil"]
_CIC_MAP = {
    "benign": "benign", "dos hulk": "dos", "dos goldeneye": "dos",
    "dos slowloris": "dos", "dos slowhttptest": "dos", "heartbleed": "dos",
    "ddos": "ddos", "portscan": "portscan", "ftp-patator": "brute_force",
    "ssh-patator": "brute_force", "web attack brute force": "web_attack",
    "web attack xss": "web_attack", "web attack sql injection": "web_attack",
    "bot": "bot_infil", "infiltration": "bot_infil",
    # corrected release (CNS 2022) label names
    "botnet": "bot_infil", "infiltration portscan": "portscan",
}
CIC_RELEASE = os.environ.get("PCSP_CIC_RELEASE", "improved")
assert CIC_RELEASE in ("improved", "original"), CIC_RELEASE
if CIC_RELEASE == "improved":
    # Without the attempted flows only 104 Web-attack flows remain (36
    # Infiltration, 736 Botnet), too few for a class of their own in a
    # 300,000-flow sample: they form one rare-attack class.
    CIC_CLASSES = ["benign", "dos", "ddos", "portscan", "brute_force",
                   "web_bot_infil"]
    _CIC_MAP = {k: ("web_bot_infil" if v in ("web_attack", "bot_infil")
                    else v) for k, v in _CIC_MAP.items()}
CIC_TAG = "imp" if CIC_RELEASE == "improved" else ""   # cache-key suffix
CIC_IMP_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday"]
# identifiers and bookkeeping columns of the corrected release that are not
# flow features (the original release likewise has no addresses, source
# port, or timestamp)
_CIC_IMP_DROP = ["id", "Flow ID", "Src IP", "Dst IP", "Src Port",
                 "Timestamp", "Label", "Attempted Category"]
_META = ("y", "day", "row", "ts", "fam")
CIC_DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday-WorkingHours-"
                 "Morning", "Thursday-WorkingHours-Afternoon",
                 "Friday-WorkingHours-Morning", "Friday-WorkingHours-"
                 "Afternoon-PortScan", "Friday-WorkingHours-Afternoon-DDos"]


def _norm(s):
    s = "".join(ch if ord(ch) < 128 else " " for ch in str(s).strip().lower())
    s = s.replace(" - ", " ")
    return " ".join(s.split())


def _cic_frame_improved():
    """Corrected release: five day files with timestamps, in time order."""
    f = CACHE / "cicids2017_improved.parquet"
    if f.exists():
        return pd.read_parquet(f)
    d = DATA / "cicids2017_improved"
    frames = []
    for i, key in enumerate(CIC_IMP_DAYS):
        df = pd.read_csv(d / f"{key}.csv", low_memory=False)
        df["day"] = i
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    att = df["Attempted Category"].to_numpy() != -1
    lab = df["Label"].map(_norm)
    # consistency of the two attempted markers
    assert (lab.str.endswith(" attempted") == att).all()
    lab = lab.where(~att, "benign")
    unk = set(lab) - set(_CIC_MAP)
    if unk:
        raise ValueError(f"unmapped CIC labels: {sorted(unk)}")
    fam = df["Label"].where(~att, "BENIGN")
    ts = (pd.to_datetime(df["Timestamp"]).astype("datetime64[us]")
          .astype("int64").to_numpy())             # microseconds since epoch
    feats = [c for c in df.columns if c not in _CIC_IMP_DROP + ["day"]]
    x = df[feats].to_numpy(np.float64)
    ok = np.isfinite(x).all(axis=1)
    out = df.loc[:, feats].astype(np.float32)
    out["y"] = lab.map(_CIC_MAP).map(CIC_CLASSES.index).astype(np.int8)
    out["day"] = df["day"].astype(np.int8)
    out["row"] = df["id"].to_numpy()
    out["ts"] = ts
    out["fam"] = fam.to_numpy()
    out = out.loc[ok]
    # time order (timestamps are not sorted in the files); ties by file id
    out = out.sort_values(["ts", "day", "row"], kind="mergesort")
    out = out.reset_index(drop=True)
    out.to_parquet(f)
    return out


def _cic_frame():
    CACHE.mkdir(parents=True, exist_ok=True)
    if CIC_RELEASE == "improved":
        return _cic_frame_improved()
    f = CACHE / "cicids2017.parquet"
    if f.exists():
        return pd.read_parquet(f)
    d = DATA / "cicids2017" / "MachineLearningCVE"
    frames = []
    for i, key in enumerate(CIC_DAY_ORDER):
        (path,) = [p for p in d.glob("*.csv") if p.name.startswith(key)]
        df = pd.read_csv(path, encoding="latin-1", low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        df["day"] = i
        df["row"] = np.arange(len(df))
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.loc[:, ~df.columns.duplicated()]      # 'Fwd Header Length.1'
    lab = df.pop("Label").map(_norm)
    unk = set(lab) - set(_CIC_MAP)
    if unk:
        raise ValueError(f"unmapped CIC labels: {sorted(unk)}")
    df["y"] = lab.map(_CIC_MAP).map(CIC_CLASSES.index).astype(np.int8)
    feats = [c for c in df.columns if c not in ("y", "day", "row")]
    x = df[feats].to_numpy(np.float64)
    ok = np.isfinite(x).all(axis=1)
    df = df.loc[ok].reset_index(drop=True)
    df[feats] = df[feats].astype(np.float32)
    df.to_parquet(f)
    return df


def _cic_feats(df):
    return [c for c in df.columns if c not in _META]


def _cic_wire_bytes(df):
    """Per-flow on-wire evidence size: L4 payload plus L3/L4 headers in
    both directions, plus 14 B Ethernet framing per packet, i.e. the
    size of the flow's packet capture without pcap record headers."""
    if "Total Fwd Packets" in df:                       # original release
        pkts = df["Total Fwd Packets"] + df["Total Backward Packets"]
        pay = (df["Total Length of Fwd Packets"]
               + df["Total Length of Bwd Packets"])
    else:                                               # corrected release
        pkts = df["Total Fwd Packet"] + df["Total Bwd packets"]
        pay = df["Total Length of Fwd Packet"] + df["Total Length of Bwd Packet"]
    hdr = (df["Fwd Header Length"].clip(lower=0)
           + df["Bwd Header Length"].clip(lower=0))
    return (pay + hdr + 14 * pkts).to_numpy(np.float64)


def _cic_evid(df, feats):
    x = df[feats].to_numpy(np.float32)
    return [r.tobytes() for r in x]


def cicids(seed: int = 0, n_iid: int = 300_000, frac=(0.6, 0.2, 0.2),
           temporal: bool = True, n_shift: int = 200_000):
    df = _cic_frame()
    feats = _cic_feats(df)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(df), size=min(n_iid, len(df)), replace=False)
    a, b = int(frac[0] * len(pick)), int((frac[0] + frac[1]) * len(pick))
    p_tr, p_ca, p_te = (df.iloc[pick[:a]], df.iloc[pick[a:b]],
                        df.iloc[pick[b:]])

    def X(d):
        return d[feats].to_numpy(np.float32)

    out = dict(Xtr=X(p_tr), ytr=p_tr.y.to_numpy(), Xca=X(p_ca),
               yca=p_ca.y.to_numpy(), Xte=X(p_te), yte=p_te.y.to_numpy(),
               classes=CIC_CLASSES, name="CIC-IDS-2017",
               n_features=len(feats))
    out["evid_te"] = _cic_evid(p_te, feats)
    out["raw_bytes_te"] = _cic_wire_bytes(p_te)
    out["feat_bytes"] = 4 * len(feats)
    if temporal:
        early = df[df.day <= 2]
        late = df[df.day >= 3]
        ie = rng.permutation(len(early))
        a = int(0.75 * len(ie))
        e_tr, e_ca = early.iloc[ie[:a]], early.iloc[ie[a:]]
        # deployment stream: a time-ordered subsample of Thu-Fri
        il = np.sort(rng.choice(len(late), size=min(n_shift, len(late)),
                                replace=False))
        l_sh = late.iloc[il]
        out.update(Xtr_t=X(e_tr), ytr_t=e_tr.y.to_numpy(),
                   Xca_t=X(e_ca), yca_t=e_ca.y.to_numpy(),
                   Xsh=X(l_sh), ysh=l_sh.y.to_numpy())
        out["raw_bytes_sh"] = _cic_wire_bytes(l_sh)
        out["day_sh"] = l_sh.day.to_numpy()     # day-file index per claim
        if "fam" in l_sh:
            out["fam_sh"] = l_sh.fam.to_numpy()  # original label per claim
            out["ts_sh"] = l_sh.ts.to_numpy()
    return out


if __name__ == "__main__":
    d = nslkdd(0)
    print(d["name"], d["Xtr"].shape, np.bincount(d["ytr"]),
          np.bincount(d["ysh"], minlength=5),
          "native bytes mean", d["raw_bytes_te"].mean())
    c = cicids(0)
    print(c["name"], c["Xtr"].shape, np.bincount(c["ytr"], minlength=8),
          "wire bytes median", np.median(c["raw_bytes_te"]),
          "mean", c["raw_bytes_te"].mean())
    print("temporal", c["Xtr_t"].shape, np.bincount(c["ytr_t"], minlength=8),
          np.bincount(c["ysh"], minlength=8))
