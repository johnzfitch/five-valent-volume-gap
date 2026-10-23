#!/bin/bash
# full interval box: two-parameter regimes to n1 <= 4000, one-parameter families to n1 <= 1e5 (as in the original)
cd "$(dirname "$0")"
for r in R1 R2+ R2- R4; do python ibox.py $r 4000 cuda ../outputs/ibox_${r}_N4000.json; done
for r in R1c R4c Tp Tm; do python ibox.py $r 100000 cuda ../outputs/ibox_${r}_N100000.json; done
