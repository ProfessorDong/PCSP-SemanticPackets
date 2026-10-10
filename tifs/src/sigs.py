"""Signature backends for PCSP v2 (same interface as sim/src/pcsp.py)."""

from __future__ import annotations

import hashlib
import os
import secrets


class StubBackend:
    """Insecure keyed-hash placeholder (sk == pk); tests only."""
    name, sig_bytes, pk_bytes = "STUB", 16, 32

    def keygen(self):
        k = secrets.token_bytes(32)
        return k, k

    def sign(self, sk, msg):
        return hashlib.blake2s(sk + msg, digest_size=16).digest()

    def verify(self, pk, msg, sig):
        return hashlib.blake2s(pk + msg, digest_size=16).digest() == sig


class DilithiumPyBackend:
    """Pure-Python ML-DSA-65 (dilithium_py); FIPS 204 byte format."""
    name = "ML-DSA-65"

    def __init__(self):
        from dilithium_py.ml_dsa import ML_DSA_65
        self._s = ML_DSA_65
        self.sig_bytes, self.pk_bytes = 3309, 1952

    def keygen(self):
        pk, sk = self._s.keygen()
        return sk, pk

    def sign(self, sk, msg):
        return self._s.sign(sk, msg)

    def verify(self, pk, msg, sig):
        try:
            return bool(self._s.verify(pk, msg, sig))
        except Exception:
            return False


class PySPXBackend:
    """SLH-DSA-SHA2-128s via pyspx (SPHINCS+ reference C)."""
    name = "SLH-DSA-SHA2-128s"

    def __init__(self):
        import pyspx.sha2_128s as s
        self._s = s
        self.sig_bytes = int(s.crypto_sign_BYTES)
        self.pk_bytes = int(s.crypto_sign_PUBLICKEYBYTES)
        self._seed = int(s.crypto_sign_SEEDBYTES)

    def keygen(self):
        pk, sk = self._s.generate_keypair(secrets.token_bytes(self._seed))
        return sk, pk

    def sign(self, sk, msg):
        return self._s.sign(msg, sk)

    def verify(self, pk, msg, sig):
        try:
            return bool(self._s.verify(msg, sig, pk))
        except Exception:
            return False


class LibOQSBackend:
    """liboqs C implementation (set OQS_INSTALL_PATH)."""

    def __init__(self, mechanism="ML-DSA-65"):
        import oqs
        self._oqs = oqs
        self.name = mechanism
        s = oqs.Signature(mechanism)
        self.sig_bytes = int(s.details["length_signature"])
        self.pk_bytes = int(s.details["length_public_key"])

    def keygen(self):
        s = self._oqs.Signature(self.name)
        pk = s.generate_keypair()
        return s.export_secret_key(), pk

    def sign(self, sk, msg):
        s = self._oqs.Signature(self.name, sk)
        return s.sign(msg)

    def verify(self, pk, msg, sig):
        try:
            return bool(self._oqs.Signature(self.name).verify(msg, sig, pk))
        except Exception:
            return False


def make(name: str):
    n = name.lower()
    if n == "stub":
        return StubBackend()
    if n == "mldsa":
        return DilithiumPyBackend()
    if n == "slhdsa":
        return PySPXBackend()
    if n.startswith("oqs:"):
        return LibOQSBackend(name[4:])
    raise ValueError(name)


SIG_BYTES = {"ML-DSA-44": 2420, "ML-DSA-65": 3309, "ML-DSA-87": 4627,
             "SLH-DSA-SHA2-128s": 7856, "SLH-DSA-SHA2-128f": 17088,
             "SLH-DSA-SHA2-192s": 16224, "Falcon-512": 752}
