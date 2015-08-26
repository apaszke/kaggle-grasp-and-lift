#!/bin/bash

function echoHeader() {
  local message=$1
  printf "\033[0;34m"
  echo "================================================================================"
  echo "= $message"
  echo "================================================================================"
  printf "\033[0m"
}

if [[ ! $1 ]]; then
  echo "No subject specified!"
  exit 1
fi

# Run setup
echoHeader "Running setup..."
sh setup.sh $1

clear old files
rm -f data/torch/*.t7
rm -f cv/*.t7

echoHeader "Training LSTM..."
th train_lstm.lua -seq_length 800 -batch_size 5 -print_every 10 -eval_val_every 100 -max_epochs 20 -rnn_size 100 -dropout 0.5

# Find the best model
best_checkpoint=$(ls -1 cv | sort | head -1)

echoHeader "Sampling validation set"
th sample.lua cv/$best_checkpoint
