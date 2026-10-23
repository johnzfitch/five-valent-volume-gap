# The Volume Gap at Five-Valent Vertices

John Zachary Fitch (zack@internetuniverse.org) · ORCID [0009-0007-7953-1531](https://orcid.org/0009-0007-7953-1531)

This repository holds the paper, its LaTeX sources, the code written for it, and the certificates behind its computer-assisted proofs. The paper is [`lqg_volume_gap.pdf`](lqg_volume_gap.pdf), 80 pages, version of 22 October 2026.

## What the paper proves

Whether the volume operator of loop quantum gravity has a spectral gap was known at valence five and higher only from numerics. Brunnemann and Rideout observed that some embeddings produce eigenvalues accumulating at zero and others do not, and that the decaying ones are exactly those whose reduced sign factors contain ±4. The paper explains this rule for the Ashtekar–Lewandowski operator and proves most of it.

Up to relabelling, every five-valent vertex operator whose edge tangents are in general position (no three coplanar) is zero or −4i[α J₁·(J₂×J₃) + β J₁·(J₄×J₅)] with (α, β) one of (2, 0), (2, 2) and (4, 2). "Contains ±4" means t = β/α = ½. For every |t| other than 0 and 1 the gap closes, exponentially in the spin. At t = 0 the gap is exactly 2√3. At t = 1 the gap is proved on several infinite classes and certified for all spins up to 19; the rest of t = 1 is stated as a conjecture.

Along the way the paper proves the sharp four-valent law λ² ≥ j_max(j_max + 1)/4, through a factorisation into a hydrogen Stark matrix. It also proves a factorial gap closing at a six-valent vertex, and a uniform gap for the Rovelli–Smolin operator on every five-valent vertex with an edge of spin ½.

## Layout

The LaTeX sources are `lqg_volume_gap.tex`, `sec*.tex`, `app*.tex` and `refs.tex`.

`code/` holds the code written for this paper (Section 10.3). `su2.py` builds the vertex operators twice, from the generators and by Racah recoupling. `grid.py` builds the normal-form operator. `orbits.py` and `nongeneric.py` classify the sign vectors. `validate.py`, `checks.py`, `reproduce.py` and `b2_constants.py` check the operator, the constants quoted in the text, the published numbers of Brunnemann–Rideout and Bianchi–Haggard, and the two constants of Theorem 6.10. `figdata.py` writes the data for the figures. The outputs are in `outputs/`.

`certificates/` holds the computer-assisted steps on which the proofs rest (Section 10.2), one subfolder for each. [`certificates/MANIFEST.txt`](certificates/MANIFEST.txt) says which statement each folder proves, in which arithmetic, how to run it and how long it takes. Most folders were copied from the research folders where the computations were first run, and keep the checksum list of their source as `SOURCE_SHA256SUMS.txt`. [`certificates/PATH_FIXES.md`](certificates/PATH_FIXES.md) lists the 17 scripts whose file paths were changed so that they run inside this tree.

`SHA256SUMS.txt` lists a SHA-256 checksum for every file of the paper, its code and its certificates. Verify with `sha256sum -c SHA256SUMS.txt`. Git stores the files byte for byte (`.gitattributes`), so the checksums hold on every platform.

## Large outputs

Two sets of per-tuple outputs are too large for the repository and are attached to the [v1.0 release](https://github.com/johnzfitch/five-valent-volume-gap/releases/tag/v1.0).

| Archive | Contents | Unzip into |
|---|---|---|
| `fc-outputs-M021-M038.zip` | the first certificate of Theorem 6.1(h), levels 21 to 38 (195 MB unpacked) | `certificates/fc/outputs/` |
| `rfloat_sqrt-outputs-large.zip` | the cited certificate of Theorem 6.1(h), levels 21 to 38, and the j₁ = 5/2 run (245 MB unpacked) | `certificates/rfloat_sqrt/outputs/` |

Each folder's `LARGE_OUTPUTS_SHA256.txt` lists their checksums; run `sha256sum -c LARGE_OUTPUTS_SHA256.txt` from the `outputs/` folder after unzipping. The proofs do not need these files to be read, since the scripts regenerate them, but the summaries and two cross-checks read them.

## Running the checks

Install the requirements with `python -m pip install -r requirements.txt`. The scripts need Python 3, NumPy, SciPy, SymPy, mpmath and python-flint. The GPU benchmarks and the interval box for j₁ = 3/2 also use PyTorch with CUDA. From the repository root, these reproduce the outputs of the paper's own code. On 22 October 2026 each run below wrote a file identical to the stored one, apart from the elapsed times that `r3` prints.

```
python code/orbits.py                                        # under 1 s
python code/validate.py                                      # about 3 min
python code/checks.py                                        # about 30 s
python code/nongeneric.py                                    # about 20 s
python code/b2_constants.py outputs/b2_constants_out.txt     # about 1 s
python code/reproduce.py r1 outputs/repro_r1.txt              # about 1 s
python code/reproduce.py r2 outputs/repro_r2.txt              # under 1 s
python code/reproduce.py r5 outputs/repro_r5.txt              # about 5 s
python code/reproduce.py r3 outputs/repro_r3_small.txt 14     # about 2 s
python code/reproduce.py r4 outputs/repro_r4_small.txt 5      # about 10 s
```

The full runs `r3 … 30` and `r4 … 9` write `repro_r3.txt` and `repro_r4.txt` and take longer. The certificates are run from their own `scripts/` folders as `certificates/MANIFEST.txt` describes; some of them take hours.

## Not included

- **The typesetting kit.** The sources use the author's etch-and-sketch style files (`etch-math`, `etch-plot`), which are not part of this repository, so the LaTeX does not compile as is. The compiled PDF is provided.
- **The review files.** The internal referee reports and the external reviews of drafts, with their triage, are not published. Section 12 and Appendix J of the paper describe how they were used.

## Companion paper

[Bohr–Sommerfeld Asymptotics for Limit-Circle Jacobi Matrices](https://github.com/johnzfitch/limit-circle-jacobi-asymptotics) comes from the same operator with su(1,1) labels.

## Licence

The paper (the LaTeX sources and the PDF) and the stored outputs are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The code (all `.py` files) is licensed under the MIT License. See [`LICENSE`](LICENSE).
