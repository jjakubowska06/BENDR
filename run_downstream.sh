#!/bin/bash

mkdir -p results

# Train LO/MSO from scratch
# python3 downstream.py linear --random-init --results-filename "results/linear_random_init-polid.xlsx"
# python3 downstream.py BENDR --random-init --results-filename "results/BENDR_random_init-polid.xlsx"

# Train LO/MSO from checkpoint
# sleep-edf
python3 downstream.py linear --model-path "weights/linear-sleepedf-AASM-pool1-all_data-no-positioning.pt" --logs-tracker-train "results/train-linear-sleepedf-AASM-pool1-all_data-no-positioning.csv" --logs-tracker-valid "results/valid-linear-sleepedf-AASM-pool1-all_data-no-positioning.csv"
# python3 downstream.py BENDR --model-path "weights/BENDR-sleepedf-AASM-pool1-temazepan.pt" --logs-tracker-train "results/train-BENDR-sleepedf-AASM-pool1-temazepan.csv" --logs-tracker-valid "results/valid-BENDR-sleepedf-AASM-pool1-temazepan.csv"

# polid
# python3 downstream.py linear --results-filename "results/linear-polid-AASM-eeg1_emg-pool1-cam-cropped.xlsx" --model-path "weights/linear-polid-AASM-eeg1_emg-pool1-cam-cropped.pt" --logs-tracker-train "results/train-linear-AASM-eeg1_emg-pool1-cam-cropped.csv" --logs-tracker-valid "results/valid-linear-AASM-eeg1_emg-pool1-cam-cropped.csv"
# python3 downstream.py BENDR --results-filename "results/BENDR-polid-AASM-eeg1_emg.xlsx" --model-path "weights/BENDR-polid-AASM-eeg1_emg.pt" --logs-tracker-train "results/train-logs_BENDR-polid-AASM-eeg1_emg.csv" --logs-tracker-valid "results/valid-logs_BENDR-polid-AASM-eeg1_emg.csv"

# Train LO/MSO from checkpoint with frozen encoder
# python3 downstream.py linear --freeze-encoder --results-filename "results/linear_freeze_encoder-polid.xlsx"
# python3 downstream.py BENDR --freeze-encoder --results-filename "results/BENDR_freeze_encoder-polid.xlsx"S