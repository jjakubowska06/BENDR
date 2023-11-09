#!/bin/bash

mkdir -p results

# Train LO/MSO from scratch
# python3 downstream.py linear --random-init --results-filename "results/linear_random_init-polid.xlsx"
# python3 downstream.py BENDR --random-init --results-filename "results/BENDR_random_init-polid.xlsx"

# # Train LO/MSO from checkpoint
# python3 downstream.py linear --results-filename "results/linear-polid.xlsx"
python3 downstream.py BENDR --results-filename "results/BENDR-polid.xlsx" --model-path "weights/BENDR-polid.pt"

# Train LO/MSO from checkpoint with frozen encoder
# python3 downstream.py linear --freeze-encoder --results-filename "results/linear_freeze_encoder-polid.xlsx"
# python3 downstream.py BENDR --freeze-encoder --results-filename "results/BENDR_freeze_encoder-polid.xlsx"