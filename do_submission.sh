#!/bin/bash

function echoHeader() {
  local message=$1
  printf "\033[0;34m"
  echo "================================================================================"
  echo "= $message"
  echo "================================================================================"
  printf "\033[0m"
}

mkdir tmp/submission_files
mkdir tmp/validation_files
mkdir tmp/model_files
rm tmp/best_losses
touch tmp/best_losses

num_subjects=12
for ((i=1; i <= $num_subjects; i++)); do
  echoHeader "Subject: $i"

  sh setup.sh $i

  echoHeader "clear old files"
  rm -f data/torch/*.t7
  rm -f cv/*.t7

  echoHeader "Training LSTM..."
  th train_lstm.lua -seq_length 800 -batch_size 5 -print_every 10 -eval_val_every 100 -max_epochs 7 -rnn_size 280 -dropout 0.3
  mkdir -p tmp/model_files/$i
  cp cv/* tmp/model_files/$i/.

  # Find the best model
  best_checkpoint=$(ls -1 cv | sort | head -1)

  echoHeader "Best model: $best_checkpoint"
  echo "$i: $best_checkpoint" >> best_losses

  echoHeader "Sampling validation set"
  th sample.lua cv/$best_checkpoint
  cp sampled_files/* tmp/validation_files/.

  echoHeader "Sampling test set"
  th sample.lua cv/$best_checkpoint -submission
  cp sampled_files/* tmp/submission_files/.
done
