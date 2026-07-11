"""Tahap 5 — uji sirkularitas via order-preservation (menggantikan containment lama).

Fragrantica menampilkan main accords terurut menurun menurut intensitas. Kalau
anotator menyalin daftar accord parfum target ke produk, URUTAN ikut tersalin.

Untuk tiap pasangan berlabel (p, q):
  contained = set(A(p)) ⊆ set(A(q))
  subseq    = A(p) subsequence terurut dari A(q)
  prefix    = A(q)[:len(A(p))] == A(p)

Null: permutasi acak urutan A(q) (set dipertahankan), 1000 permutasi seed 0.
p-value empiris satu sisi untuk subseq dan prefix.

Kontrol wajib:
  K1 konvensi sorting (alfabetis / freq desc / idf asc). Jika ada > 50% -> Gate G4.
  K2 kontrol panjang (distribusi len(A(p))).
  K3 kontrol negatif (q, produk acak non-label) -> harus turun ke null.
  K4 tidak relevan (data bersih).

Verdict (§5.6): DERIVATIF(sebagian) | TIDAK TERBUKTI DERIVATIF | TIDAK DAPAT DIUJI.
Strata & eval sets ALL / NON_OP (§5.5).

Output: 02_circularity.md, 02_order_preservation.csv, 02_null.csv,
        02_sorting_convention.csv, 02_suspected_mislinked.csv, 02_strata.csv
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from ..data import load_dataset

OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "v3"
N_PERM = 1000
SEED = 0
K1_THRESHOLD = 0.50


def is_subsequence(a, b) -> bool:
    it = iter(b)
    return all(tok in it for tok in a)


def is_prefix(a, b) -> bool:
    return len(a) <= len(b) and list(b[:len(a)]) == list(a)


def monotone(seq, key, nonincreasing=False) -> bool:
    """True if seq is monotone in key (adjacency-based; lenient with ties)."""
    ks = [key(x) for x in seq]
    if len(ks) < 2:
        return True
    if nonincreasing:
        return all(ks[i] >= ks[i + 1] for i in range(len(ks) - 1))
    return all(ks[i] <= ks[i + 1] for i in range(len(ks) - 1))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset()
    rng = np.random.default_rng(SEED)

    # corpus df / idf over products + queries
    df = Counter()
    for fr in [p.accords for p in ds.products] + [q.accords for q in ds.queries]:
        for a in set(fr):
            df[a] += 1
    n_docs = len(ds.products) + len(ds.queries)
    idf = {a: np.log((1 + n_docs) / (1 + df[a])) + 1.0 for a in df}
    alpha = {a: a for a in df}

    # ---- labeled pairs ----
    pairs = []  # (p_idx, q_idx, A(p), A(q))
    for q in ds.queries:
        for pi in sorted(q.relevant):
            pairs.append((pi, q.idx, ds.products[pi].accords, q.accords))

    # ---- observed contained/subseq/prefix ----
    rows = []
    obs_sub = obs_pre = obs_con = 0
    for pi, qi, ap, aq in pairs:
        con = set(ap) <= set(aq)
        sub = is_subsequence(ap, aq)
        pre = is_prefix(ap, aq)
        obs_con += con; obs_sub += sub; obs_pre += pre
        rows.append([pi, ds.products[pi].name, qi, ds.queries[qi].name,
                     len(ap), len(aq), int(con), int(sub), int(pre)])
    n_pairs = len(pairs)

    with open(OUT_DIR / "02_order_preservation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["p_idx", "product", "q_idx", "query", "len_Ap", "len_Aq",
                    "contained", "subseq", "prefix"])
        w.writerows(rows)

    # ---- null: permute A(q) order 1000x ----
    null_sub = np.zeros(N_PERM, dtype=int)
    null_pre = np.zeros(N_PERM, dtype=int)
    for t in range(N_PERM):
        cs = cp = 0
        for pi, qi, ap, aq in pairs:
            perm = list(aq)
            rng.shuffle(perm)
            if is_subsequence(ap, perm):
                cs += 1
            if is_prefix(ap, perm):
                cp += 1
        null_sub[t] = cs
        null_pre[t] = cp
    p_sub = (1 + int(np.sum(null_sub >= obs_sub))) / (1 + N_PERM)
    p_pre = (1 + int(np.sum(null_pre >= obs_pre))) / (1 + N_PERM)

    with open(OUT_DIR / "02_null.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["statistic", "observed", "null_mean", "null_p95", "p_empirical"])
        w.writerow(["subseq", obs_sub, f"{null_sub.mean():.2f}",
                    f"{np.percentile(null_sub, 95):.1f}", f"{p_sub:.4f}"])
        w.writerow(["prefix", obs_pre, f"{null_pre.mean():.2f}",
                    f"{np.percentile(null_pre, 95):.1f}", f"{p_pre:.4f}"])

    # ---- K1 sorting convention ----
    def prop(frs, key, nonincr):
        frs2 = [f for f in frs if len(f) >= 2]
        if not frs2:
            return 0.0
        return sum(monotone(f, key, nonincr) for f in frs2) / len(frs2)

    Aq = [q.accords for q in ds.queries]
    Ap = [ds.products[pi].accords for pi, _, _, _ in pairs]
    k1 = {
        "Aq_alpha_asc": prop(Aq, lambda a: alpha[a], False),
        "Aq_freq_desc": prop(Aq, lambda a: df[a], True),
        "Aq_idf_asc": prop(Aq, lambda a: idf[a], False),
        "Ap_alpha_asc": prop(Ap, lambda a: alpha[a], False),
        "Ap_freq_desc": prop(Ap, lambda a: df[a], True),
        "Ap_idf_asc": prop(Ap, lambda a: idf[a], False),
    }
    k1_fail = any(v > K1_THRESHOLD for v in k1.values())

    with open(OUT_DIR / "02_sorting_convention.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["convention", "proportion", "exceeds_0.50"])
        for k, v in k1.items():
            w.writerow([k, f"{v:.4f}", int(v > K1_THRESHOLD)])

    # ---- K2 length ----
    lenhist = dict(sorted(Counter(len(ap) for _, _, ap, _ in pairs).items()))

    # ---- K3 negative control (q, random non-relevant product) ----
    neg_sub = neg_pre = 0
    all_pidx = np.arange(ds.n_pool)
    for q in ds.queries:
        choices = np.setdiff1d(all_pidx, list(q.relevant), assume_unique=False)
        pj = int(rng.choice(choices))
        ap = ds.products[pj].accords
        neg_sub += is_subsequence(ap, q.accords)
        neg_pre += is_prefix(ap, q.accords)

    # ---- §5.4 suspected mislinked ----
    mislinked = []
    for pi, qi, ap, aq in pairs:
        if set(ap) <= set(aq):
            continue  # only non-contained
        cands = []
        for q2 in ds.queries:
            if q2.idx == qi:
                continue
            if set(ap) <= set(q2.accords) or is_prefix(ap, q2.accords):
                cands.append(q2.name)
        if cands:
            mislinked.append([pi, ds.products[pi].name, ds.queries[qi].name,
                              "; ".join(cands[:5])])
    with open(OUT_DIR / "02_suspected_mislinked.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["p_idx", "product", "linked_query", "alt_candidates"])
        w.writerows(mislinked)

    # ---- §5.5 strata per query (highest among dupes) ----
    order = {"prefix": 3, "subseq": 2, "contained": 1, "partial": 0}
    inv = {v: k for k, v in order.items()}
    strata_rows = []
    non_op = []
    for q in ds.queries:
        best = 0
        for pi in q.relevant:
            ap = ds.products[pi].accords
            if is_prefix(ap, q.accords):
                lv = 3
            elif is_subsequence(ap, q.accords):
                lv = 2
            elif set(ap) <= set(q.accords):
                lv = 1
            else:
                lv = 0
            best = max(best, lv)
        stratum = inv[best]
        is_nonop = best < 2   # no prefix/subseq dupe
        non_op.append(is_nonop)
        strata_rows.append([q.idx, q.name, len(q.relevant), stratum, int(is_nonop)])
    with open(OUT_DIR / "02_strata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["q_idx", "query", "n_dupes", "stratum", "is_NON_OP"])
        w.writerows(strata_rows)
    strat_counts = Counter(r[3] for r in strata_rows)
    n_nonop = sum(non_op)

    # ---- verdict ----
    if k1_fail:
        verdict = "TIDAK DAPAT DIUJI"
    elif p_pre < 0.01 or p_sub < 0.01:
        verdict = "DERIVATIF (sebagian)"
    else:
        verdict = "TIDAK TERBUKTI DERIVATIF"

    # ---- gates ----
    gates = []
    if k1_fail:
        gates.append("G4")
    if len(mislinked) > 15:
        gates.append("G8")
    if verdict.startswith("DERIVATIF") and n_nonop < 100:
        gates.append("G7")

    # ---- report ----
    L = []
    L.append("# 02 — Uji Sirkularitas (order-preservation)\n")
    L.append(f"n pasangan berlabel = {n_pairs}\n")
    L.append("## Observasi (§5.2)\n")
    L.append("| statistik | observed | null mean | null p95 | p empiris |")
    L.append("|---|---|---|---|---|")
    L.append(f"| contained | {obs_con} | (set-based, permutation-invariant) | | |")
    L.append(f"| subseq | {obs_sub} | {null_sub.mean():.2f} | {np.percentile(null_sub,95):.1f} | {p_sub:.4f} |")
    L.append(f"| prefix | {obs_pre} | {null_pre.mean():.2f} | {np.percentile(null_pre,95):.1f} | {p_pre:.4f} |")
    L.append("")
    L.append("## K1 — konvensi sorting (Gate G4 jika ada > 0.50)\n")
    L.append("| konvensi | proporsi |")
    L.append("|---|---|")
    for k, v in k1.items():
        flag = " **>0.50**" if v > K1_THRESHOLD else ""
        L.append(f"| {k} | {v:.4f}{flag} |")
    L.append(f"\nK1 fail = {k1_fail}")
    L.append("")
    L.append("## K2 — kontrol panjang len(A(p))\n")
    L.append(f"{lenhist}")
    if len(lenhist) == 1 or max(lenhist.values()) / n_pairs > 0.8:
        L.append("\n> len(A(p)) terkonsentrasi → konvensi 'ambil top-k'; dilaporkan.")
    L.append("")
    L.append("## K3 — kontrol negatif (q, produk acak non-label)\n")
    L.append(f"- subseq = {neg_sub}, prefix = {neg_pre} (harus ~ null mean: "
             f"{null_sub.mean():.1f}/{null_pre.mean():.1f})")
    L.append("")
    L.append("## §5.4 — suspected mislinked (non-contained dgn kandidat q' lain)\n")
    L.append(f"- jumlah = {len(mislinked)} (Gate G8 jika > 15). Lihat "
             "`02_suspected_mislinked.csv`. **Hanya untuk review manusia — jangan auto-fix.**")
    L.append("")
    L.append("## §5.5 — strata & eval set\n")
    L.append(f"- distribusi stratum query: {dict(strat_counts)}")
    L.append(f"- NON_OP (query tanpa dupe prefix/subseq) = {n_nonop} dari {len(ds.queries)}")
    L.append("- **NON_OP bukan 'subset bersih'**: ia subset *tanpa bukti transkripsi "
             "berurutan*. Anotator yang menyalin lalu mengacak urutan tak terdeteksi.")
    L.append("")
    L.append("## VERDICT (§5.6)\n")
    L.append(f"**{verdict}**")
    if gates:
        L.append(f"\n**GATE: {', '.join(gates)}**")
        for gname in gates:
            (OUT_DIR / f"GATE_{gname}.md").write_text(
                f"# GATE {gname}\n\nDipicu di Tahap 5 (circularity). Verdict={verdict}, "
                f"k1_fail={k1_fail}, n_nonop={n_nonop}, mislinked={len(mislinked)}.\n",
                encoding="utf-8")
    (OUT_DIR / "02_circularity.md").write_text("\n".join(L), encoding="utf-8")

    print(f"verdict = {verdict}")
    print(f"  obs subseq={obs_sub} (p={p_sub:.4f}), prefix={obs_pre} (p={p_pre:.4f}), "
          f"contained={obs_con}/{n_pairs}")
    print(f"  K1 fail={k1_fail}  strata={dict(strat_counts)}  NON_OP={n_nonop}")
    print(f"  mislinked={len(mislinked)}  gates={gates}")
    return verdict


if __name__ == "__main__":
    main()
