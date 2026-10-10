"""
PCSP v2: proof-carrying semantic packets, journal version.

Wire format (all integers big-endian)
-------------------------------------
Header, 53 B:
    sid(2) kid(1) seq(4) cnt(4) t(4) N(2) mode(1) K(1) alpha_code(1)
    bundle(1) R_evid(32)
`seq` is the per-sender batch sequence number (monotone across key
rotations), `cnt` the number of claims the sender emitted before this
batch (so the receiver can count dropped claims), `t` the emission epoch.
Claims, N * B_sem B:
    SET  : z(1) || ctx(1) || setmask(ceil(K/8))
    SCORE: z(1) || ctx(1) || qprob(K)          (8-bit quantized p)
Signature: |sigma| B (scheme-determined; Falcon padded to its maximum).

Signed message (rebuilt by the receiver, never transmitted):
    M = "PCSP-v2-hdr" || header-without-R_evid (21 B) || H(theta) || H(D)
        || R_claim || R_evid
H(theta), H(D) come from the approved bundle Pi[bundle]; the bundle
index itself is signed, and Pi[bundle] fixes (mode, K, alpha_code), which
the receiver checks against the header.

Trees: RFC 6962 (leaf 0x00, node 0x01, largest-power-of-two split).
Claim leaves: "PCSP-v2-claim" || i(2) || claim body.
Evidence leaves (salted): "PCSP-v2-evid" || sid || seq || i || r_i || H(X_i),
r_i = HMAC-SHA256(k_salt, seq || i)[:16], revealed only in audit openings.

Fragment-independent verification (FIV, N a power of two): header frames
carry header || 2^j depth-j claim-tree roots || sigma; each claim fragment
carries seq(4) || offset(2) || 2^k claims || (log2 N - j - k) sibling
hashes up to its depth-j root.

Author: Liang Dong.
"""

from __future__ import annotations

import hashlib
import hmac
import struct
from dataclasses import dataclass, field
from typing import Sequence

HASH_LEN = 32
HDR_TAG = b"PCSP-v2-hdr"
CLAIM_TAG = b"PCSP-v2-claim"
EVID_TAG = b"PCSP-v2-evid"

MODE_SET = 0
MODE_SCORE = 1

_HDR_FMT = ">HBIIIHBBBB"     # sid kid seq cnt t N mode K alpha bundle
HDR_FIXED = struct.calcsize(_HDR_FMT)          # 21 B
HDR_BYTES = HDR_FIXED + HASH_LEN               # + R_evid = 53 B
FRAG_HDR = 6                                   # seq(4) + offset(2)
SALT_LEN = 16


def H(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


# ---------------------------------------------------------------------------
# RFC 6962 Merkle tree
# ---------------------------------------------------------------------------

def _leaf(d: bytes) -> bytes:
    return H(b"\x00" + d)


def _node(l: bytes, r: bytes) -> bytes:
    return H(b"\x01" + l + r)


def _split(n: int) -> int:
    """Largest power of two strictly smaller than n (n >= 2)."""
    k = 1
    while k << 1 < n:
        k <<= 1
    return k


def mth(leaf_data: Sequence[bytes]) -> bytes:
    """Merkle tree hash over already-encoded leaf inputs."""
    n = len(leaf_data)
    if n == 0:
        return H(b"")
    if n == 1:
        return _leaf(leaf_data[0])
    k = _split(n)
    return _node(mth(leaf_data[:k]), mth(leaf_data[k:]))


def mth_from_hashes(hs: Sequence[bytes]) -> bytes:
    """Same tree, but inputs are leaf hashes (already prefixed)."""
    n = len(hs)
    if n == 1:
        return hs[0]
    k = _split(n)
    return _node(mth_from_hashes(hs[:k]), mth_from_hashes(hs[k:]))


def audit_path(leaf_data: Sequence[bytes], m: int) -> list[bytes]:
    """RFC 6962 PATH(m, D[n])."""
    n = len(leaf_data)
    if n <= 1:
        return []
    k = _split(n)
    if m < k:
        return audit_path(leaf_data[:k], m) + [mth(leaf_data[k:])]
    return audit_path(leaf_data[k:], m - k) + [mth(leaf_data[:k])]


def verify_path(leaf_d: bytes, m: int, n: int, path: list[bytes],
                root: bytes) -> bool:
    """Verify an RFC 6962 inclusion proof (RFC 9162 sec. 2.1.3.2)."""
    if m >= n:
        return False
    fn, sn = m, n - 1
    r = _leaf(leaf_d)
    for p in path:
        if sn == 0:
            return False
        if fn & 1 or fn == sn:
            r = _node(p, r)
            if not fn & 1:
                while fn & 1 == 0 and fn != 0:
                    fn >>= 1
                    sn >>= 1
        else:
            r = _node(r, p)
        fn >>= 1
        sn >>= 1
    return sn == 0 and r == root


def subtree_proof(hs: Sequence[bytes], lo: int, hi: int) -> list[bytes]:
    """Sibling hashes needed to recompute the root from the aligned
    subtree hs[lo:hi] (hi-lo a power of two, lo a multiple of it, N a
    power of two).  Used by fragment-independent verification."""
    n = len(hs)
    assert n & (n - 1) == 0 and (hi - lo) & (hi - lo - 1) == 0
    size = hi - lo
    proof = []
    level = list(hs)
    idx = lo // size
    # collapse the leaf level up to the subtree size
    s = 1
    while s < size:
        level = [_node(level[i], level[i + 1])
                 for i in range(0, len(level), 2)]
        s <<= 1
    while len(level) > 1:
        proof.append(level[idx ^ 1])
        level = [_node(level[i], level[i + 1])
                 for i in range(0, len(level), 2)]
        idx >>= 1
    return proof


def root_from_subtree(sub_hashes: Sequence[bytes], lo: int,
                      proof: list[bytes]) -> bytes:
    size = len(sub_hashes)
    h = mth_from_hashes(sub_hashes)
    idx = lo // size
    for p in proof:
        h = _node(p, h) if idx & 1 else _node(h, p)
        idx >>= 1
    return h


# ---------------------------------------------------------------------------
# Claims and batches
# ---------------------------------------------------------------------------

def setmask(classes: Sequence[int], K: int) -> bytes:
    m = 0
    for c in classes:
        m |= 1 << int(c)
    return m.to_bytes((K + 7) // 8, "big")


def unmask(b: bytes, K: int) -> tuple[int, ...]:
    m = int.from_bytes(b, "big")
    return tuple(c for c in range(K) if m >> c & 1)


def claim_bytes(mode: int, K: int) -> int:
    return 2 + ((K + 7) // 8 if mode == MODE_SET else K)


@dataclass(frozen=True)
class Claim:
    z: int
    ctx: int
    cert: bytes             # set mask (SET) or quantized probs (SCORE)

    def body(self) -> bytes:
        return bytes([self.z & 0xFF, self.ctx & 0xFF]) + self.cert


def claim_leaf_input(i: int, c: Claim) -> bytes:
    return CLAIM_TAG + struct.pack(">H", i) + c.body()


def evid_leaf_input(sid: int, seq: int, i: int, salt: bytes,
                    hx: bytes) -> bytes:
    return EVID_TAG + struct.pack(">HIH", sid, seq, i) + salt + hx


def salt_for(k_salt: bytes, seq: int, i: int) -> bytes:
    return hmac.new(k_salt, struct.pack(">IH", seq, i),
                    hashlib.sha256).digest()[:SALT_LEN]


@dataclass
class Header:
    sid: int
    kid: int
    seq: int
    cnt: int
    t: int
    N: int
    mode: int
    K: int
    alpha_code: int
    bundle: int
    R_evid: bytes

    def fixed(self) -> bytes:
        return struct.pack(_HDR_FMT, self.sid, self.kid, self.seq, self.cnt,
                           self.t, self.N, self.mode, self.K,
                           self.alpha_code, self.bundle)

    def pack(self) -> bytes:
        return self.fixed() + self.R_evid

    @staticmethod
    def unpack(b: bytes) -> "Header":
        f = struct.unpack(_HDR_FMT, b[:HDR_FIXED])
        return Header(*f, R_evid=bytes(b[HDR_FIXED:HDR_FIXED + HASH_LEN]))


def signed_message(h: Header, H_theta: bytes, H_D: bytes,
                   R_claim: bytes) -> bytes:
    return HDR_TAG + h.fixed() + H_theta + H_D + R_claim + h.R_evid


@dataclass
class Batch:
    hdr: Header
    claims: list[Claim]
    sigma: bytes
    # sender-side secrets kept for audits (never transmitted)
    evidence: list[bytes] = field(default_factory=list, repr=False)
    salts: list[bytes] = field(default_factory=list, repr=False)

    def to_bytes(self) -> bytes:
        return (self.hdr.pack() + b"".join(c.body() for c in self.claims)
                + self.sigma)

    def wire_bytes(self) -> int:
        return len(self.to_bytes())


class ParseError(ValueError):
    pass


def parse_batch(data: bytes, sig_len: int | tuple[int, int]) -> Batch:
    """Gate G4: parse a serialized batch.  Rejects unless the total length
    equals header + N claims of the size fixed by (mode, K) + a signature
    of admissible length (exact, or a (min, max) range for Falcon)."""
    if len(data) < HDR_BYTES:
        raise ParseError("short header")
    h = Header.unpack(data)
    if h.mode not in (MODE_SET, MODE_SCORE) or not 2 <= h.K <= 64 or h.N < 1:
        raise ParseError("bad header field")
    cb = claim_bytes(h.mode, h.K)
    body_end = HDR_BYTES + h.N * cb
    slen = len(data) - body_end
    lo, hi = (sig_len, sig_len) if isinstance(sig_len, int) else sig_len
    if not lo <= slen <= hi:
        raise ParseError("length")
    claims = []
    for i in range(h.N):
        o = HDR_BYTES + i * cb
        claims.append(Claim(data[o], data[o + 1], bytes(data[o + 2:o + cb])))
    return Batch(h, claims, bytes(data[body_end:]))


@dataclass
class Bundle:
    """Approved bundle Pi[beta]: model hash, calibration hash, and the
    certificate semantics the sender must declare in its header."""
    H_theta: bytes
    H_D: bytes
    mode: int = MODE_SET
    K: int = 5
    alpha_code: int = 1


class Transmitter:
    def __init__(self, sig, sk, sid: int, kid: int, bundle_idx: int,
                 bundle: Bundle, k_salt: bytes, mode: int | None = None,
                 K: int | None = None, alpha_code: int | None = None):
        self.sig, self.sk = sig, sk
        self.sid, self.kid = sid, kid
        self.bundle_idx, self.bundle = bundle_idx, bundle
        self.k_salt = k_salt
        self.mode = bundle.mode if mode is None else mode
        self.K = bundle.K if K is None else K
        self.alpha_code = bundle.alpha_code if alpha_code is None else alpha_code
        self.seq = 0
        self.cnt = 0

    def encode(self, claims: list[Claim], evidence: list[bytes],
               t: int) -> Batch:
        N = len(claims)
        seq, cnt = self.seq, self.cnt
        self.seq += 1
        self.cnt += N
        salts = [salt_for(self.k_salt, seq, i) for i in range(N)]
        ev = [evid_leaf_input(self.sid, seq, i, salts[i], H(x))
              for i, x in enumerate(evidence)]
        R_evid = mth(ev)
        R_claim = mth([claim_leaf_input(i, c) for i, c in enumerate(claims)])
        hdr = Header(self.sid, self.kid, seq, cnt, t, N, self.mode, self.K,
                     self.alpha_code, self.bundle_idx, R_evid)
        sigma = self.sig.sign(self.sk, signed_message(
            hdr, self.bundle.H_theta, self.bundle.H_D, R_claim))
        return Batch(hdr, list(claims), sigma, list(evidence), salts)

    def open(self, batch: Batch, i: int):
        """Audit opening: (X_i, salt_i, inclusion path in evidence tree)."""
        ev = [evid_leaf_input(self.sid, batch.hdr.seq, j, batch.salts[j],
                              H(x)) for j, x in enumerate(batch.evidence)]
        return batch.evidence[i], batch.salts[i], audit_path(ev, i)

    def open_bytes(self, batch: Batch, i: int) -> bytes:
        x, s, path = self.open(batch, i)
        return x + s + b"".join(path)


def audit_response_bytes(B_x: float, N: int) -> float:
    """Upper bound on the opening size: evidence, 16-B salt, and at most
    ceil(log2 N) path hashes (exact when N is a power of two)."""
    depth = max(0, (N - 1).bit_length())
    return B_x + SALT_LEN + HASH_LEN * depth


AUDIT_CHALLENGE_BYTES = 2 + 4 + 2          # sid, seq, index (reverse link)


# ---------------------------------------------------------------------------
# Receiver
# ---------------------------------------------------------------------------

class ReplayWindow:
    """Per-sender anti-replay state in the style of RFC 4303 / RFC 6479:
    highest accepted sequence number plus a W-bit bitmap of the most
    recent W sequence numbers.  Accept seq iff it is newer than the
    highest seen, or within the window and not yet marked."""

    def __init__(self, W: int = 64):
        self.W = W
        self.top = -1
        self.bits = 0

    def check(self, seq: int) -> bool:
        if seq > self.top:
            return True
        off = self.top - seq
        if off >= self.W:
            return False
        return not (self.bits >> off) & 1

    def commit(self, seq: int):
        if seq > self.top:
            shift = seq - self.top
            self.bits = ((self.bits << shift) | 1) & ((1 << self.W) - 1) \
                if shift < self.W else 1
            self.top = seq
        else:
            self.bits |= 1 << (self.top - seq)


G_STRUCT, G_BUNDLE, G_SIG, G_FRESH, G_REPLAY = (
    "G4_struct", "G2_bundle", "G1_sig", "G3_fresh", "G3_replay")


@dataclass
class ReceiverPolicy:
    check_sig: bool = True
    check_bundle: bool = True
    check_fresh: bool = True
    check_replay: bool = True
    tau: int = 2                    # freshness window (epochs)


class Receiver:
    def __init__(self, sig, keys: dict, bundles: dict,
                 policy: ReceiverPolicy | None = None, W: int = 64,
                 sig_len=None):
        """keys: {(sid, kid): pk}; bundles: {bundle_idx: Bundle}."""
        self.sig = sig
        self.keys = keys
        self.bundles = bundles
        self.policy = policy or ReceiverPolicy()
        self.W = W
        self.sig_len = sig_len if sig_len is not None else sig.sig_bytes
        self.win: dict[int, ReplayWindow] = {}
        self.n_emitted: dict[int, int] = {}   # authenticated claim counter
        self.n_accepted: dict[int, int] = {}
        self.fiv_state: dict = {}             # (sid, seq) -> registered FIV batch

    def _window(self, sid):
        if sid not in self.win:
            self.win[sid] = ReplayWindow(self.W)
        return self.win[sid]

    def verify_bytes(self, data: bytes, t_now: int) -> tuple[bool, str, Batch | None]:
        try:
            b = parse_batch(data, self.sig_len)
        except ParseError:
            return False, G_STRUCT, None
        ok, why = self.verify(b, t_now)
        return ok, why, b

    def verify(self, batch: Batch, t_now: int) -> tuple[bool, str]:
        """Return (accepted, reason); all-or-nothing per monolithic batch."""
        p = self.policy
        h = batch.hdr
        if h.N != len(batch.claims) or any(
                len(c.cert) != claim_bytes(h.mode, h.K) - 2
                for c in batch.claims):
            return False, G_STRUCT
        b = self.bundles.get(h.bundle)
        if p.check_bundle and (b is None or (b.mode, b.K, b.alpha_code) !=
                               (h.mode, h.K, h.alpha_code)):
            return False, G_BUNDLE
        if not p.check_bundle:
            # ablation: trust model/calibration hashes carried in the packet
            ex = getattr(batch, "explicit", None)
            if ex is not None:
                b = Bundle(*ex)
            elif b is None:
                b = Bundle(H(b"unknown"), H(b"unknown"))
        if p.check_sig:
            pk = self.keys.get((h.sid, h.kid))
            if pk is None:
                return False, G_SIG
            R_claim = mth([claim_leaf_input(i, c)
                           for i, c in enumerate(batch.claims)])
            msg = signed_message(h, b.H_theta, b.H_D, R_claim)
            if not self.sig.verify(pk, msg, batch.sigma):
                return False, G_SIG
        if p.check_fresh and abs(t_now - h.t) > p.tau:
            return False, G_FRESH
        if p.check_replay:
            w = self._window(h.sid)
            if not w.check(h.seq):
                return False, G_REPLAY
            w.commit(h.seq)
        self.n_emitted[h.sid] = max(self.n_emitted.get(h.sid, 0), h.cnt + h.N)
        self.n_accepted[h.sid] = self.n_accepted.get(h.sid, 0) + h.N
        return True, "ok"

    def check_audit(self, batch: Batch, i: int, x: bytes, salt: bytes,
                    path: list[bytes]) -> bool:
        h = batch.hdr
        leaf = evid_leaf_input(h.sid, h.seq, i, salt, H(x))
        return verify_path(leaf, i, h.N, path, h.R_evid)


# ---------------------------------------------------------------------------
# Fragment-independent verification (FIV)
# ---------------------------------------------------------------------------

def _level(hs, depth_from_top, N):
    """Hashes of the 2^j subtrees at depth j of a power-of-two tree."""
    size = N >> depth_from_top
    return [mth_from_hashes(hs[o:o + size]) for o in range(0, N, size)]


def fiv_encode(batch: Batch, j: int, k: int):
    """Return (header_stream, claim_fragments) for a power-of-two batch.
    header_stream = header || 2^j depth-j roots || sigma (split into frames
    by the link); claim fragment f = seq || offset || 2^k claims || proof."""
    N = batch.hdr.N
    n = N.bit_length() - 1
    assert N == 1 << n and 0 <= j and j + k <= n
    hs = [_leaf(claim_leaf_input(i, c)) for i, c in enumerate(batch.claims)]
    roots = _level(hs, j, N)
    hdr_stream = batch.hdr.pack() + b"".join(roots) + batch.sigma
    P, sub = 1 << k, N >> j
    frags = []
    for off in range(0, N, P):
        base = (off // sub) * sub
        proof = subtree_proof(hs[base:base + sub], off - base, off - base + P)
        frags.append(struct.pack(">IH", batch.hdr.seq, off)
                     + b"".join(batch.claims[i].body()
                                for i in range(off, off + P))
                     + b"".join(proof))
    return hdr_stream, frags


def fiv_verify_header(rx: Receiver, hdr_stream: bytes, j: int, t_now: int):
    """Verify an FIV header stream (G4 length, G2, G1, G3) and register the
    batch so that its claim fragments can be accepted.  Returns
    (ok, header, depth-j roots)."""
    if len(hdr_stream) < HDR_BYTES:
        return False, None, None
    h = Header.unpack(hdr_stream)
    nroots = 1 << j
    slen = len(hdr_stream) - HDR_BYTES - HASH_LEN * nroots
    lo, hi = (rx.sig_len, rx.sig_len) if isinstance(rx.sig_len, int) \
        else rx.sig_len
    n = h.N.bit_length() - 1
    if h.N != 1 << n or j > n or not lo <= slen <= hi:          # G4
        return False, h, None
    roots = [hdr_stream[HDR_BYTES + HASH_LEN * r:HDR_BYTES + HASH_LEN * (r + 1)]
             for r in range(nroots)]
    sigma = hdr_stream[HDR_BYTES + HASH_LEN * nroots:]
    b = rx.bundles.get(h.bundle)                                 # G2
    if b is None or (b.mode, b.K, b.alpha_code) != (h.mode, h.K, h.alpha_code):
        return False, h, None
    pk = rx.keys.get((h.sid, h.kid))                             # G1
    if pk is None or not rx.sig.verify(pk, signed_message(
            h, b.H_theta, b.H_D, mth_from_hashes(roots)), sigma):
        return False, h, None
    if abs(t_now - h.t) > rx.policy.tau:                         # G3
        return False, h, None
    w = rx._window(h.sid)
    if not w.check(h.seq):
        return False, h, None
    w.commit(h.seq)
    rx.n_emitted[h.sid] = max(rx.n_emitted.get(h.sid, 0), h.cnt + h.N)
    rx.fiv_state[(h.sid, h.seq)] = dict(h=h, roots=roots, j=j, seen=set())
    return True, h, roots


def fiv_accept_fragment(rx: Receiver, sid: int, frag: bytes, k: int,
                        t_now: int):
    """Accept the claims of one fragment of a registered FIV batch.  Rejects
    fragments of unknown or expired batches (G3 freshness, state dropped
    after tau epochs), fragments already accepted (per-batch offset set),
    and fragments that fail G4 length or the subtree proof."""
    if len(frag) < FRAG_HDR:
        return None
    seq, off = struct.unpack(">IH", frag[:FRAG_HDR])
    st = rx.fiv_state.get((sid, seq))
    if st is None:
        return None
    h = st["h"]
    if abs(t_now - h.t) > rx.policy.tau:
        del rx.fiv_state[(sid, seq)]
        return None
    if off in st["seen"]:
        return None
    claims = fiv_verify_fragment(h, st["roots"], frag, st["j"], k)
    if claims is None:
        return None
    st["seen"].add(off)
    rx.n_accepted[sid] = rx.n_accepted.get(sid, 0) + len(claims)
    return claims


def fiv_verify_fragment(h: Header, roots, frag: bytes, j: int, k: int):
    """Stateless check of one claim fragment against verified roots.
    Returns its claims, or None."""
    seq, off = struct.unpack(">IH", frag[:FRAG_HDR])
    N, P = h.N, 1 << k
    cb = claim_bytes(h.mode, h.K)
    n = N.bit_length() - 1
    plen = HASH_LEN * (n - j - k)
    if seq != h.seq or off % P or off >= N or \
            len(frag) != FRAG_HDR + P * cb + plen:
        return None
    body = frag[FRAG_HDR:FRAG_HDR + P * cb]
    claims = [Claim(body[i * cb], body[i * cb + 1],
                    bytes(body[i * cb + 2:(i + 1) * cb])) for i in range(P)]
    proof = [frag[FRAG_HDR + P * cb + HASH_LEN * r:
                  FRAG_HDR + P * cb + HASH_LEN * (r + 1)]
             for r in range(n - j - k)]
    sub = N >> j
    base = (off // sub) * sub
    hs = [_leaf(claim_leaf_input(off + i, c)) for i, c in enumerate(claims)]
    if root_from_subtree(hs, off - base, proof) != roots[off // sub]:
        return None
    return claims


# ---------------------------------------------------------------------------
# Byte accounting (Proposition 3)
# ---------------------------------------------------------------------------

def batch_overhead(sig_bytes: int) -> int:
    """Per-batch fixed cost B_o = header + signature."""
    return HDR_BYTES + sig_bytes


def per_claim_bytes(N: int, mode: int, K: int, sig_bytes: int,
                    q: float = 0.0, B_x: float = 0.0) -> float:
    return (claim_bytes(mode, K) + batch_overhead(sig_bytes) / N
            + q * audit_response_bytes(B_x, N))
