"""Digest of survey_M*/summary.jsonl -> survey_summary.txt.
Per sigma: q_min(2j_max) sequence, trend class (by comparing the last value with the minimum over 2j_max<=8),
and the global minimiser per 2j_max with its spins, compared with Brunnemann-Rideout II Table 5.
Usage: python summarize_survey.py survey_M22
"""
import sys
import json
import math
from collections import defaultdict
from chain_check import chain_couplings_sq, chain_spectrum

# BR 0706.0382 Table 5 (doubled spins of the overall minimiser, sigma_min = (4,2,2,0)), 2j5 = 2..22
BR_T5 = {2: (1, 1, 2, 2, 2), 3: (1, 2, 3, 3, 3), 4: (1, 3, 4, 4, 4), 5: (1, 4, 5, 5, 5), 6: (1, 5, 6, 6, 6),
         7: (1, 6, 7, 7, 7), 8: (1, 7, 8, 8, 8), 9: (1, 8, 9, 9, 9), 10: (3, 8, 9, 10, 10),
         11: (3, 9, 10, 11, 11), 12: (3, 10, 11, 12, 12), 13: (3, 11, 12, 13, 13), 14: (3, 12, 13, 14, 14),
         15: (3, 13, 14, 15, 15), 16: (5, 13, 14, 16, 16), 17: (5, 14, 15, 17, 17), 18: (5, 15, 16, 18, 18),
         19: (5, 16, 17, 19, 19), 20: (5, 17, 18, 20, 20), 21: (5, 18, 19, 21, 21), 22: (5, 19, 20, 22, 22)}


def main(d):
    rows = [json.loads(l) for l in open(f"{d}/summary.jsonl")]
    by_sig = defaultdict(dict)
    for r in rows:
        if r["q_min"] is not None:
            by_sig[tuple(r["sigma"])][r["2jmax"]] = (r["q_min"], tuple(r["argmin"]), r["max_zero_over_qmin_any_tuple"])
    M = max(r["2jmax"] for r in rows)
    out = []
    out.append(f"Survey {d}: 85 sigma classes (BR sorted labelling), all sorted 5-valent tuples with 2j_max <= {M}.")
    out.append("q_min = smallest nonzero |eigenvalue| of i*sum sigma qhat (BR normalisation); lambda_V = sqrt(q_min).")
    cls = defaultdict(list)
    amb_max = 0.0
    for s, seq in sorted(by_sig.items()):
        early = min(seq[k][0] for k in seq if k <= 8)
        last = seq[M][0]
        trend = "decreasing" if last < 0.5 * early else ("increasing" if last > 1.5 * early else "constant")
        cls[trend].append(s)
        amb_max = max(amb_max, max(v[2] for v in seq.values()))
    out.append("Trend classes (q_min at 2j_max=M vs min over 2j_max<=8; factor 2): " +
               ", ".join(f"{k}: {len(v)}" for k, v in cls.items()))
    for k in ("decreasing", "constant", "increasing"):
        out.append(f"  {k}: " + " ".join(str(s) for s in cls.get(k, [])))
    has4 = {k: sum(1 for s in v if 4 in map(abs, s)) for k, v in cls.items()}
    out.append("  number of classes containing a +-4 entry: " + str(has4))
    out.append(f"Largest (largest numerically-zero singular value / q_min) over all records: {amb_max:.2e}")
    out.append("")
    out.append("Global minimiser per 2j_max (over all sigma and tuples):")
    out.append(" 2jmax  q_min            lambda_V        sigma           spins(2j)            BR Table 5        family(1,N-1,N,N,N) q")
    for J5 in range(2, M + 1):
        best = min(((seq[J5][0], s, seq[J5][1]) for s, seq in by_sig.items() if J5 in seq))
        _, c2 = chain_couplings_sq((1, J5 - 1, J5, J5, J5), 4, 2)
        qf = chain_spectrum(c2).min()
        out.append(f" {J5:4d}  {best[0]:.6e}  {math.sqrt(best[0]):.6e}  {str(best[1]):15s} {str(best[2]):20s} "
                   f"{str(BR_T5.get(J5)):18s} {qf:.6e}")
    txt = "\n".join(out)
    open("survey_summary.txt", "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "survey_M22")
