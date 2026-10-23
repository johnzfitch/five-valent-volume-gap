# Path fixes in the shipped certificate scripts (22 October 2026)

Some copied scripts read their inputs from the research folders they were written in
(`../../<research folder>/...`, or an absolute `R:\quantum-tetrahedra\research\...` path), so
they could not run from this tree. Only the path strings were changed; no computation was
touched. Every change is listed below. The stored outputs in each folder were produced by
the scripts before the change (with the old path strings, in the research tree). The
research-folder names map to the shipped subfolders as in MANIFEST.txt
(volume_gap_higher_valence_20261021 -> hv, dichotomy_large_spin_20261021 -> ls,
dichotomy_finite_cert_20261021 -> fc, dichotomy_open_regions_20261021 -> or,
dichotomy_rigor_recheck_20261021 -> rr, dichotomy_fixed_j1_20261021 -> fj,
volume_gap_4valent_20261021 -> vg4).

## Changes (file, line, old string, new string)

| file | line | old | new |
| --- | --- | --- | --- |
| `gd/scripts/classes_table.py` | 12 | `"..", "volume_gap_higher_valence_20261021"` | `"..", "hv"` |
| `lj/scripts/lj_check_core.py` | 6 | `'..', '..', 'dichotomy_large_spin_20261021', 'scripts'` | `'..', '..', 'ls', 'scripts'` |
| `lj/scripts/lj_fc_minima.py` | 4 | `'..', '..', 'dichotomy_finite_cert_20261021', 'outputs'` | `'..', '..', 'fc', 'outputs'` |
| `ls/scripts/ls_validate.py` | 10 | `'..', '..', 'dichotomy_open_regions_20261021', 'scripts'` | `'..', '..', 'or', 'scripts'` |
| `refA/scripts/cert_ldl_sample.py` | 23 | `"../../dichotomy_finite_cert_20261021/outputs/cert_M%03d.jsonl"` | `"../../fc/outputs/cert_M%03d.jsonl"` |
| `refA/scripts/cert_ldl_sample.py` | 24 | `"../../dichotomy_rigor_recheck_20261021/outputs/cert52_M%03d.jsonl"` | `"../../rr/outputs/cert52_M%03d.jsonl"` |
| `rr/scripts/summary52.py` | 9 | `open("../../dichotomy_fixed_j1_20261021/outputs/stagecert_s52_39_70.txt")` | `open("../../fj/outputs/stagecert_s52_39_70.txt")` |
| `rs/scripts/rs_core.py` | 9 | `sys.path.insert(0, r"R:\quantum-tetrahedra\research\volume_gap_higher_valence_20261021")` | `import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'hv'))` |
| `v6/scripts/ex1.py` | 2 | `sys.path.insert(0, r"R:\quantum-tetrahedra\research\volume_gap_higher_valence_20261021")` | `import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'hv'))` |
| `v6/scripts/ex2.py` | 3 | `sys.path.insert(0, r"R:\quantum-tetrahedra\research\volume_gap_higher_valence_20261021")` | `import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'hv'))` |
| `v6/scripts/ex3.py` | 2 | `sys.path.insert(0, r"R:\quantum-tetrahedra\research\volume_gap_higher_valence_20261021")` | `import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'hv'))` |
| `v6/scripts/ex4.py` | 2 | `sys.path.insert(0, r"R:\quantum-tetrahedra\research\volume_gap_higher_valence_20261021")` | `import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'hv'))` |
| `v6/scripts/ex5.py` | 2 | `sys.path.insert(0, r"R:\quantum-tetrahedra\research\volume_gap_higher_valence_20261021")` | `import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'hv'))` |
| `v6/scripts/v6_validate.py` | 13 | `sys.path.insert(0, r"R:\quantum-tetrahedra\research\volume_gap_higher_valence_20261021")` | `import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'hv'))` |
| `vg4/referee/scripts/r6_survey.py` | 63 | `np.load(r"R:/quantum-tetrahedra/research/volume_gap_4valent_20261021/survey_N200/pertuple_J4le60.npz", allow_pickle=False)` | `np.load(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', '..', 'survey_N200', 'pertuple_J4le60.npz'), allow_pickle=False)` |
| `z32/scripts/z32_crosscheck_OR.py` | 9 | `'..', '..', 'dichotomy_open_regions_20261021', 'scripts'` | `'..', '..', 'or', 'scripts'` |
| `z32/scripts/z32_Z32_literal.py` | 8 | `'..', '..', 'dichotomy_open_regions_20261021', 'scripts'` | `'..', '..', 'or', 'scripts'` |
| `z32/scripts/zk_crosscheck.py` | 8 | `'..', '..', 'dichotomy_open_regions_20261021', 'scripts'` | `'..', '..', 'or', 'scripts'` |

Docstrings and comments that name the research folders were left as they are; they are
provenance, not paths.

## Per-folder checksum lists refreshed

The hash of each changed script was replaced in its folder's list (first 16 hex digits
shown). `vg4/referee/scripts/r6_survey.py` is in no per-folder list; it is covered by
../SHA256SUMS.txt.

| file | list | old hash | new hash |
| --- | --- | --- | --- |
| `gd/scripts/classes_table.py` | `gd/SOURCE_SHA256SUMS.txt` | `be4552aa43d20027...` | `3191da65e7804670...` |
| `lj/scripts/lj_check_core.py` | `lj/SOURCE_SHA256SUMS.txt` | `c0ded7248d6ed016...` | `a79d5abc0052fba8...` |
| `lj/scripts/lj_fc_minima.py` | `lj/SOURCE_SHA256SUMS.txt` | `2efcb71502126c6d...` | `0e348e1358ee4344...` |
| `ls/scripts/ls_validate.py` | `ls/SOURCE_SHA256SUMS.txt` | `0fdf9a1c0fb22448...` | `60a829f0dd21d6f4...` |
| `refA/scripts/cert_ldl_sample.py` | `refA/SOURCE_SHA256SUMS.txt` | `077b6e615e3e82e0...` | `70efc36316e07244...` |
| `rr/scripts/summary52.py` | `rr/SOURCE_SHA256SUMS.txt` | `957833bee9038e8c...` | `37f96c10eb358880...` |
| `rs/scripts/rs_core.py` | `rs/SOURCE_SHA256SUMS.txt` | `319e2810892e1e98...` | `c8422a0188d326ff...` |
| `v6/scripts/ex1.py` | `v6/SOURCE_SHA256SUMS.txt` | `77a0f810abb8a16a...` | `e1d9a0b6ee4327e1...` |
| `v6/scripts/ex2.py` | `v6/SOURCE_SHA256SUMS.txt` | `43489642aafad8e9...` | `9ea1f4e774a79278...` |
| `v6/scripts/ex3.py` | `v6/SOURCE_SHA256SUMS.txt` | `34bbc2bc8b7044e3...` | `b0e09a4a72ce5426...` |
| `v6/scripts/ex4.py` | `v6/SOURCE_SHA256SUMS.txt` | `58661841522a55c3...` | `efc877ef7542666c...` |
| `v6/scripts/ex5.py` | `v6/SOURCE_SHA256SUMS.txt` | `1a833b2f0fc77cc4...` | `9fc6ffe6dda5690a...` |
| `v6/scripts/v6_validate.py` | `v6/SOURCE_SHA256SUMS.txt` | `771ea1f1142bff1b...` | `eb3eae930a110775...` |
| `vg4/referee/scripts/r6_survey.py` | (not in a per-folder list) | | `4258b4ab68b83de7...` |
| `z32/scripts/z32_crosscheck_OR.py` | `z32/SOURCE_SHA256SUMS.txt` | `7441e8c5089bc739...` | `1349f907f19bcde3...` |
| `z32/scripts/z32_Z32_literal.py` | `z32/SOURCE_SHA256SUMS.txt` | `c055eaa895f68b1f...` | `17b840b19979eed7...` |
| `z32/scripts/zk_crosscheck.py` | `z32/SOURCE_SHA256SUMS.txt` | `91fbdb7f1cd5da6f...` | `5d1fa4e86a86d126...` |

## Inputs added so that the fixed paths resolve

Copied from the research folders on 22 October 2026; each verifies against the source
checksum list already shipped in its folder.

- `hv/survey_M22/summary.jsonl`, `hv/survey_summary.txt` (listed in hv/SHA256SUMS.txt), read by `gd/scripts/classes_table.py`.
- `vg4/survey_N200/pertuple_J4le60.npz` (listed in vg4/SHA256SUMS.txt), read by `vg4/referee/scripts/r6_survey.py`.
- `rr/outputs/cert52_M062.jsonl` ... `cert52_M070.jsonl` (23 MB; with the 23 already shipped, all 32 files of rr/LARGE_OUTPUTS_SHA256.txt verify), read by `rr/scripts/summary52.py` and `refA/scripts/cert_ldl_sample.py`.

Not added: the fc per-tuple outputs for levels 21 to 38 (about 195 MB). `lj/scripts/lj_fc_minima.py`,
`refA/scripts/cert_ldl_sample.py` and `fc/scripts/summarize.py` need them; regenerate them first with
`python run_cert.py 21 38 8` in fc/scripts. `lj/scripts/lj_limit_finite.py` does not read them.

## Replays after the change

Run on a scratch copy of this tree, each from its own directory, output compared line by line with
the stored output (timings ignored):

| command | result |
| --- | --- |
| `rr/scripts: python summary52.py` | identical to outputs/summary52.txt |
| `gd/scripts: python classes_table.py <out>` | identical to outputs/classes_table.txt |
| `v6/scripts: python v6_validate.py` | identical to outputs/v6_validate_output.txt |
| `ls/scripts: python ls_validate.py` | identical to outputs/validate_out.txt |
| `vg4/referee/scripts: python r6_survey.py 80 <out>` | identical to outputs/r6_out.txt |
| `z32/scripts: python z32_crosscheck_OR.py 16 <out>` | identical to outputs/crosscheck_OR_M16.txt |
| `z32/scripts: python zk_crosscheck.py 3 10 <out>` | identical to outputs/zk_crosscheck_s32_M10.txt |
| `z32/scripts: python z32_Z32_literal.py <out>` | identical to outputs/Z32_literal_families.txt |
| `rs/scripts: python rs_verify.py 40` | identical to outputs/rs_verify_M40.txt |

Not replayed: `v6/scripts/ex1.py` to `ex5.py` (exploration scripts with no stored output),
`lj/scripts/lj_check_core.py` (a module), `lj/scripts/lj_fc_minima.py` and
`refA/scripts/cert_ldl_sample.py` (need the fc outputs above).
