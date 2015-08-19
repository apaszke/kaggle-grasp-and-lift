#!/bin/bash

if [[ ! $1 ]]; then
  echo "No subject specified!"
  exit 1
fi

echo "Creating model for subject $1"

# Run setup
echo "Running setup..."
sh setup.sh $1

# clear old files
rm -f data/torch/*.t7
rm -f cv/*.t7

th train_lstm.lua -seq_length 100 -batch_size 10 -print_every 10 -eval_val_every 100 -max_epochs 1

# Find the best model
best_checkpoint=$(ls -1 cv | sort | head -1)

echo "Sampling validation set"
th sample.lua cv/$best_checkpoint

echo "Sampling test set"
th sample.lua cv/$best_checkpoint -submission
