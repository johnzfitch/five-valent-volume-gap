"""Summarise survey_N200/summary.jsonl -> survey_summary.txt."""
import json, os
here = os.path.dirname(os.path.abspath(__file__))
R = [json.loads(l) for l in open(os.path.join(here, "survey_N200", "summary.jsonl"))]
o = []
P = lambda *a: o.append(" ".join(str(x) for x in a))
P("tuples surveyed (sorted, admissible, 2j_max <= 200):", sum(r["n_tuples"] for r in R), "; with a nonzero eigenvalue (d>=2):", sum(r["n_nonzero"] for r in R))
for k in ["thm_violations", "eighth_violations", "violations_lamS_lt_1overC", "violations_S_bound", "violations_lam_lt_jp1_over_27", "BR_violations"]:
    P(f"  count of {k}:", sum(r[k] for r in R))
g = min(R, key=lambda r: r["lam_min"]); P("global minimum of the smallest positive eigenvalue:", g["lam_min"], "at 2j =", g["lam_min_tuple"])
P("per-j_max minimiser is (1/2,1/2,j_max,j_max) for every j_max:", all(r["argmin_is_half_half_j_j"] for r in R),
  "; max |lam_min(j_max) - sqrt(j_max(j_max+1))/2| =", max(abs(r["lam_min_equals_sqrt_j_jp1_over_2"]) for r in R))
m = min(R, key=lambda r: r["lam_over_jmax_min"]); P("min lam/j_max:", m["lam_over_jmax_min"], "at", m["BR_ratio_min_tuple"])
d3 = [r for r in R if "d3_lam_over_sqrtjj1_min" in r]; m = min(d3, key=lambda r: r["d3_lam_over_sqrtjj1_min"])
P("d>=3: min lam/sqrt(j_max(j_max+1)):", m["d3_lam_over_sqrtjj1_min"], "at", m["d3_lam_over_sqrtjj1_tuple"])
P("min lam*S over d>=3:", min(r["lamS_min_d3"] for r in R if r["J4"] > 1), "; max lam*S even d:", max(r["lamS_max_even"] for r in R if r["lamS_max_even"]),
  "; max lam*S odd d:", max(r["lamS_max_odd"] for r in R if r["lamS_max_odd"]))
P("max S*(j_max+1):", max(r["max_S_times_jmax_plus_1"] for r in R), "(Lemma 4 bound: theta_3*sqrt(96) = 11.827)")
w = [r for r in R if r["wkb_ratio_min"] is not None]
P("lam / Bohr-Sommerfeld prediction (non-flat shape spaces m<1): min", min(r["wkb_ratio_min"] for r in w), "max", max(r["wkb_ratio_max"] for r in w))
P("")
P("J4=2j_max  lam_min  argmin  | d>=3 min lam/sqrt(j(j+1)) argmin | max lam*S (even d) | lam/WKB range")
for r in R:
    if r["J4"] in (1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 60, 80, 100, 126, 150, 175, 200):
        P(r["J4"], round(r["lam_min"], 9), r["lam_min_tuple"], "|", r.get("d3_lam_over_sqrtjj1_min"), r.get("d3_lam_over_sqrtjj1_tuple"), "|", r["lamS_max_even"], "|", r["wkb_ratio_min"], r["wkb_ratio_max"])
open(os.path.join(here, "survey_summary.txt"), "w").write("\n".join(o) + "\n")
print("\n".join(o))
