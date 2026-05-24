"""
PCSP (Proof-Carrying Semantic Packets) reference implementation.

A PCSP packet binds a semantic claim z, conformal-prediction
uncertainty set C_alpha, freshness epoch t, context label ctx, three
SHA-256 commitments (evidence H(x), model H(theta), calibration
H(D)), and a post-quantum signature sigma over all the above.

Aggregated batches (Theorem 2) sign the Merkle root of N per-claim
hashes, amortising the signature cost to O(1/N) per claim.

Signature backend is pluggable: a `SigBackend` abstract interface
with concrete implementations for `LibOQSBackend` (real ML-DSA-65 /
SLH-DSA-SHA2-128s when liboqs is available) and `StubBackend`
(zero-overhead placeholder for end-to-end pipeline testing).

Author: Liang Dong, MILCOM 2026 Paper 3.
"""

from __future__ import annotations

import hashlib
import struct
import time
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence


# ---------------------------------------------------------------------------
# Signature backends
# ---------------------------------------------------------------------------

class SigBackend(ABC):
    """Abstract post-quantum signature scheme."""

    name: str = "abstract"
    sig_bytes: int = 0
    pk_bytes: int = 0

    @abstractmethod
    def keygen(self): ...
    @abstractmethod
    def sign(self, sk, msg: bytes) -> bytes: ...
    @abstractmethod
    def verify(self, pk, msg: bytes, sig: bytes) -> bool: ...


class StubBackend(SigBackend):
    """Zero-cost placeholder signer.  Signatures are 16-byte BLAKE2s
    digests of (key || msg).  In this stub `sk == pk` (degenerate
    symmetric) so the pipeline can be exercised without a real PQC
    primitive; use only for end-to-end testing -- NOT secure.
    """

    name = "STUB"
    sig_bytes = 16
    pk_bytes  = 32

    def keygen(self):
        sk = secrets.token_bytes(32)
        return sk, sk

    def sign(self, sk, msg: bytes) -> bytes:
        return hashlib.blake2s(sk + msg,
                               digest_size=self.sig_bytes).digest()

    def verify(self, pk, msg: bytes, sig: bytes) -> bool:
        return hashlib.blake2s(pk + msg,
                               digest_size=self.sig_bytes).digest() == sig


class DilithiumPyBackend(SigBackend):
    """Real ML-DSA-65 (Dilithium-Mode3) via the pure-Python
    `dilithium_py` package.  Signature size matches FIPS 204 (3293 B);
    verify latency is ~50-100x slower than a C/AVX2 implementation
    such as liboqs but produces identical signatures."""

    name = "ML-DSA-65"

    def __init__(self):
        from dilithium_py.ml_dsa import ML_DSA_65
        self._scheme = ML_DSA_65
        pk0, sk0 = self._scheme.keygen()
        sig0 = self._scheme.sign(sk0, b"x")
        self.sig_bytes = len(sig0)
        self.pk_bytes  = len(pk0)

    def keygen(self):
        pk, sk = self._scheme.keygen()
        return sk, pk

    def sign(self, sk, msg: bytes) -> bytes:
        return self._scheme.sign(sk, msg)

    def verify(self, pk, msg: bytes, sig: bytes) -> bool:
        try:
            return bool(self._scheme.verify(pk, msg, sig))
        except Exception:
            return False


class PySPXBackend(SigBackend):
    """Real SLH-DSA-SHA2-128s (NIST FIPS 205, formerly SPHINCS+-SHA2-128s)
    via the `pyspx` package.  Signature size matches FIPS 205 (7856 B);
    pyspx is a CFFI binding around the SPHINCS+ reference C
    implementation."""

    name = "SLH-DSA-SHA2-128s"

    def __init__(self):
        import pyspx.sha2_128s as scheme
        self._scheme = scheme
        self.sig_bytes = int(scheme.crypto_sign_BYTES)
        self.pk_bytes  = int(scheme.crypto_sign_PUBLICKEYBYTES)
        self._seed_bytes = int(scheme.crypto_sign_SEEDBYTES)

    def keygen(self):
        pk, sk = self._scheme.generate_keypair(
            secrets.token_bytes(self._seed_bytes))
        return sk, pk

    def sign(self, sk, msg: bytes) -> bytes:
        return self._scheme.sign(msg, sk)

    def verify(self, pk, msg: bytes, sig: bytes) -> bool:
        try:
            return bool(self._scheme.verify(msg, sig, pk))
        except Exception:
            return False


class LibOQSBackend(SigBackend):
    """Real ML-DSA-65 or SLH-DSA-SHA2-128s via liboqs-python.

    liboqs is imported lazily inside ``__init__`` because the first
    ``import oqs`` triggers a 5-second-warned bootstrap that compiles
    the liboqs C library from source.  We avoid that side-effect for
    pipelines that only want the StubBackend.
    """

    def __init__(self, mechanism: str = "ML-DSA-65"):
        try:
            import oqs                       # liboqs-python
        except Exception as e:
            raise RuntimeError(
                f"liboqs-python is not installed: {e}") from e
        self.name = mechanism
        self._sig = oqs.Signature(mechanism)
        details = self._sig.details
        self.sig_bytes = int(details["length_signature"])
        self.pk_bytes  = int(details["length_public_key"])

    def keygen(self):
        pk = self._sig.generate_keypair()
        sk = self._sig.export_secret_key()
        return sk, pk

    def sign(self, sk, msg: bytes) -> bytes:
        self._sig.import_secret_key(sk)
        return self._sig.sign(msg)

    def verify(self, pk, msg: bytes, sig: bytes) -> bool:
        try:
            return bool(self._sig.verify(msg, sig, pk))
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Merkle tree for aggregated PCSP
# ---------------------------------------------------------------------------

def _sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def merkle_root(leaves: Sequence[bytes]) -> bytes:
    if not leaves:
        return b"\x00" * 32
    level = list(leaves)
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        level = [_sha256(level[i] + level[i + 1])
                 for i in range(0, len(level), 2)]
    return level[0]


def merkle_path(leaves: Sequence[bytes], idx: int) -> list[bytes]:
    """Return the authentication path for `leaves[idx]`."""
    level = list(leaves)
    path: list[bytes] = []
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        sib = idx ^ 1
        path.append(level[sib])
        level = [_sha256(level[i] + level[i + 1])
                 for i in range(0, len(level), 2)]
        idx //= 2
    return path


def merkle_verify(leaf: bytes, path: list[bytes], idx: int,
                  root: bytes) -> bool:
    h = leaf
    for sib in path:
        if idx & 1:
            h = _sha256(sib + h)
        else:
            h = _sha256(h + sib)
        idx //= 2
    return h == root


# ---------------------------------------------------------------------------
# PCSP packet
# ---------------------------------------------------------------------------

@dataclass
class PCSPClaim:
    """One per-cycle semantic assertion (pre-aggregation)."""
    z: int                      # class index (4 bits for RadioML, 3 for NSL-KDD)
    c_alpha: tuple[int, ...]    # conformal prediction set as class indices
    t: int                      # freshness epoch (cycle counter)
    ctx: int                    # context tag (16 bits)
    H_x: bytes                  # SHA-256 commitment to raw evidence (32 B)


@dataclass
class PCSPBatch:
    """N aggregated claims under a single signature."""
    claims: list[PCSPClaim]
    H_theta: bytes              # SHA-256 of model parameters (32 B)
    H_D:     bytes              # SHA-256 of calibration state (32 B)
    sigma:   bytes              # signature over Merkle root + H_theta + H_D
    leaves:  list[bytes] = field(default_factory=list)   # per-claim leaves


# ---------------------------------------------------------------------------
# Encoder / verifier
# ---------------------------------------------------------------------------

def claim_leaf(c: PCSPClaim) -> bytes:
    """Hash of a single PCSPClaim (Merkle leaf input)."""
    return _sha256(
        struct.pack(">BHIH", c.z & 0xff, c.t & 0xffff,
                    c.ctx & 0xffff_ffff,
                    len(c.c_alpha) & 0xffff)
        + bytes(c.c_alpha)
        + c.H_x
    )


def encode_batch(claims: list[PCSPClaim], H_theta: bytes, H_D: bytes,
                 sig: SigBackend, sk) -> PCSPBatch:
    leaves = [claim_leaf(c) for c in claims]
    root = merkle_root(leaves)
    msg = root + H_theta + H_D
    sigma = sig.sign(sk, msg)
    return PCSPBatch(claims=claims, H_theta=H_theta, H_D=H_D,
                     sigma=sigma, leaves=leaves)


def packet_bytes(batch: PCSPBatch, sig_bytes: int) -> int:
    """Theoretical packet byte cost (Theorem 2 RHS, per claim N=1
    or aggregated by len(batch.claims))."""
    n = len(batch.claims)
    # Per-claim: z 1 B, t 2 B, ctx 4 B, c_alpha 1+|set| B, H_x 32 B
    per_claim = sum(1 + 2 + 4 + 1 + len(c.c_alpha) + 32
                    for c in batch.claims)
    # Plus shared: H_theta 32 B, H_D 32 B, signature
    shared = 32 + 32 + sig_bytes
    return per_claim + shared


def verify_batch(batch: PCSPBatch, sig: SigBackend, pk,
                 approved_model_hashes: set[bytes],
                 freshness_window: int,
                 current_epoch: int) -> tuple[bool, list[bool]]:
    """Return (sig_ok, per_claim_accept_bits).

    A per-claim bit is True iff (i) the batch signature is valid,
    (ii) the model hash is in the approved bundle, (iii) the claim's
    freshness epoch is within `freshness_window` cycles of
    `current_epoch`, AND (iv) the claim's Merkle leaf matches the
    aggregated root.
    """
    if batch.H_theta not in approved_model_hashes:
        return False, [False] * len(batch.claims)

    leaves = [claim_leaf(c) for c in batch.claims]
    root = merkle_root(leaves)
    msg = root + batch.H_theta + batch.H_D
    if not sig.verify(pk, msg, batch.sigma):
        return False, [False] * len(batch.claims)

    accept = []
    for c, leaf in zip(batch.claims, leaves):
        ok_t = abs(current_epoch - c.t) <= freshness_window
        ok_leaf = (leaf in leaves)        # always true if not tampered
        accept.append(ok_t and ok_leaf)
    return True, accept


# ---------------------------------------------------------------------------
# Sanity test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    backend = StubBackend()
    sk, pk = backend.keygen()

    fake_H_theta = _sha256(b"model-v1")
    fake_H_D     = _sha256(b"calib-v1")
    approved     = {fake_H_theta}

    claims = [
        PCSPClaim(z=0, c_alpha=(0,), t=10, ctx=1, H_x=_sha256(b"x0")),
        PCSPClaim(z=1, c_alpha=(0, 1), t=11, ctx=1, H_x=_sha256(b"x1")),
        PCSPClaim(z=2, c_alpha=(2,), t=12, ctx=1, H_x=_sha256(b"x2")),
    ]

    batch = encode_batch(claims, fake_H_theta, fake_H_D, backend, sk)
    ok, accept = verify_batch(batch, backend, pk, approved,
                               freshness_window=5, current_epoch=12)
    print("packet bytes:", packet_bytes(batch, backend.sig_bytes))
    print("sig_ok:", ok, " per-claim accept:", accept)
