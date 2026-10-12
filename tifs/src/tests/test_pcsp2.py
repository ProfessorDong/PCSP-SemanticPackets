import os, sys
from dataclasses import replace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pcsp2 as P
from sigs import StubBackend


def test_merkle_paths_all_sizes():
    for n in range(1, 40):
        d = [bytes([i]) * 3 for i in range(n)]
        r = P.mth(d)
        for m in range(n):
            assert P.verify_path(d[m], m, n, P.audit_path(d, m), r)
            assert not P.verify_path(d[m] + b"x", m, n, P.audit_path(d, m), r)
            if n > 1:
                assert not P.verify_path(d[m], (m + 1) % n, n,
                                         P.audit_path(d, m), r)


def test_no_duplicate_leaf_mutation():
    d = [bytes([i]) for i in range(3)]
    assert P.mth(d) != P.mth(d + [d[-1]])


def test_subtree_proofs():
    for N in [2, 8, 64]:
        hs = [P._leaf(bytes([i])) for i in range(N)]
        root = P.mth_from_hashes(hs)
        size = 1
        while size <= N:
            for lo in range(0, N, size):
                pr = P.subtree_proof(hs, lo, lo + size)
                assert P.root_from_subtree(hs[lo:lo + size], lo, pr) == root
            size <<= 1


def test_replay_window():
    w = P.ReplayWindow(8)
    for s in [5, 3, 7, 6]:
        assert w.check(s); w.commit(s)
    for s in [5, 3, 7, 6]:
        assert not w.check(s)
    assert w.check(4)
    assert not w.check(-1)            # older than the window


def _setup(N=8, K=5):
    sig = StubBackend(); sk, pk = sig.keygen()
    b = P.Bundle(P.H(b"theta"), P.H(b"D"), P.MODE_SET, K, 10)
    b_same_hash = P.Bundle(b.H_theta, b.H_D, P.MODE_SET, K, 50)
    tx = P.Transmitter(sig, sk, 1, 0, 0, b, b"k" * 32)
    rx = P.Receiver(sig, {(1, 0): pk}, {0: b, 1: b_same_hash})
    cl = [P.Claim(i % K, 0, P.setmask([i % K], K)) for i in range(N)]
    ev = [bytes([i]) * 50 for i in range(N)]
    return sig, tx, rx, cl, ev


def test_accept_and_attacks():
    sig, tx, rx, cl, ev = _setup()
    b0 = tx.encode(cl, ev, t=10)
    assert rx.verify(b0, 10) == (True, "ok")
    assert rx.verify(b0, 10)[1] == P.G_REPLAY          # in-window replay
    b1 = tx.encode(cl, ev, t=11)
    bad = P.Batch(b1.hdr, [P.Claim(4, 0, cl[0].cert)] + cl[1:], b1.sigma)
    assert rx.verify(bad, 11)[1] == P.G_SIG             # claim tamper
    bad = P.Batch(replace(b1.hdr, N=7), cl[:7], b1.sigma)
    assert rx.verify(bad, 11)[1] == P.G_SIG             # truncation
    bad = P.Batch(replace(b1.hdr, t=1), cl, b1.sigma)
    assert rx.verify(bad, 11)[1] == P.G_SIG             # epoch rewrite
    # bundle-index rewrite to an entry with the same hashes but a
    # different alpha: G2 (header/bundle mismatch) or G1 (index is signed)
    bad = P.Batch(replace(b1.hdr, bundle=1), cl, b1.sigma)
    assert rx.verify(bad, 11)[1] in (P.G_BUNDLE, P.G_SIG)
    bad = P.Batch(replace(b1.hdr, bundle=1, alpha_code=50), cl, b1.sigma)
    assert rx.verify(bad, 11)[1] == P.G_SIG
    assert rx.verify(b1, 20)[1] == P.G_FRESH            # stale
    assert rx.verify(b1, 11) == (True, "ok")
    x, s, path = tx.open(b1, 3)
    assert rx.check_audit(b1, 3, x, s, path)
    assert not rx.check_audit(b1, 3, x + b"!", s, path)
    assert rx.n_emitted[1] == 16 and rx.n_accepted[1] == 16


def test_serialization_and_g4():
    sig, tx, rx, cl, ev = _setup()
    b = tx.encode(cl, ev, t=5)
    data = b.to_bytes()
    assert len(data) == P.HDR_BYTES + 8 * P.claim_bytes(P.MODE_SET, 5) + 16
    ok, why, pb = rx.verify_bytes(data, 5)
    assert ok and pb.claims == cl
    assert rx.verify_bytes(data[:-1], 5)[1] == P.G_STRUCT     # short
    assert rx.verify_bytes(data + b"\0", 5)[1] == P.G_STRUCT  # long
    x, s, path = tx.open(b, 2)
    ob = tx.open_bytes(b, 2)
    assert len(ob) == P.audit_response_bytes(len(x), 8)
    sid, seq, i, x2, s2, p2 = P.parse_opening(ob)
    assert (sid, seq, i) == (1, b.hdr.seq, 2) and rx.check_audit(b, i, x2, s2, p2)
    # out-of-range fields are rejected by G4
    d = bytearray(data); d[P.HDR_BYTES] = 5                    # z >= K
    assert rx.verify_bytes(bytes(d), 5)[1] == P.G_STRUCT
    d = bytearray(data); d[P.HDR_BYTES + 2] = 0x80             # padding bit
    assert rx.verify_bytes(bytes(d), 5)[1] == P.G_STRUCT


def test_challenge_authentication():
    k = b"r" * 32
    c = P.make_challenge(k, 1, 7, 3)
    assert len(c) == P.AUDIT_CHALLENGE_BYTES
    assert P.check_challenge(k, c) == (1, 7, 3)
    assert P.check_challenge(b"x" * 32, c) is None
    assert P.check_challenge(k, c[:-1] + bytes([c[-1] ^ 1])) is None


def test_discrepancy_predicate():
    wrong_but_honest = bytes([255, 0])           # model itself says benign
    assert not P.discrepant(wrong_but_honest, bytes([255, 0]))
    assert P.discrepant(bytes([128, 127]), bytes([127, 128]))   # arg max flips
    assert P.discrepant(bytes([255, 0]), bytes([3, 252]))      # overwrite
    assert not P.discrepant(bytes([200, 55]), bytes([201, 54]))  # 1 LSB


def test_counter_overflow():
    sig, tx, rx, cl, ev = _setup()
    tx.cnt = P.MAX_U32 - 4
    try:
        tx.encode(cl, ev, t=1)
        assert False
    except OverflowError:
        pass


def test_fiv():
    sig, tx, rx, cl, ev = _setup(N=64)
    b = tx.encode(cl, ev, t=5)
    for j, k in [(0, 2), (1, 3), (2, 0), (0, 6)]:
        rx2 = P.Receiver(sig, rx.keys, rx.bundles)
        hs, frags = P.fiv_encode(b, j, k)
        ok, h, roots = P.fiv_verify_header(rx2, hs, j, 5)
        assert ok and rx2.n_emitted[1] == 64
        got = []
        for f in frags[::2]:                        # half the frames lost
            c = P.fiv_accept_fragment(rx2, 1, f, k, 5)
            assert c is not None
            got += c
        assert len(got) == (1 << k) * len(frags[::2])
        assert P.fiv_accept_fragment(rx2, 1, frags[0], k, 5) is None   # replay
        f = bytearray(frags[-1]); f[P.FRAG_HDR] ^= 1                   # tamper
        assert P.fiv_accept_fragment(rx2, 1, bytes(f), k, 5) is None
        if len(frags) > 1:                          # late fragment: expired
            assert P.fiv_accept_fragment(rx2, 1, frags[1], k, 9) is None
            assert P.fiv_accept_fragment(rx2, 1, frags[1], k, 5) is None
        assert not P.fiv_verify_header(rx2, hs, j, 5)[0]               # replay
        assert not P.fiv_verify_header(rx2, hs[:-1], j, 5)[0]          # G4


def test_fiv_overlap_and_revocation():
    sig, tx, rx, cl, ev = _setup(N=4)
    b = tx.encode(cl, ev, t=5)
    hs, f2 = P.fiv_encode(b, 0, 1)          # fragments of 2 claims
    _, f1 = P.fiv_encode(b, 0, 0)           # fragments of 1 claim
    rx2 = P.Receiver(sig, dict(rx.keys), rx.bundles)
    assert P.fiv_verify_header(rx2, hs, 0, 5)[0]
    assert P.fiv_accept_fragment(rx2, 1, f2[0], 1, 5) is not None  # claims 0,1
    assert P.fiv_accept_fragment(rx2, 1, f1[1], 0, 5) is None      # claim 1 again
    assert P.fiv_accept_fragment(rx2, 1, f1[2], 0, 5) is None      # other layout
    rx2.keys.pop((1, 0))                                           # revoke
    assert P.fiv_accept_fragment(rx2, 1, f2[1], 1, 5) is None
    assert P.fiv_accept_fragment(rx2, 1, f2[1], 7, 5) is None      # bad k


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("pass", k)
