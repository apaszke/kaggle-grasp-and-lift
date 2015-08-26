#!/bin/bash
spacer="========================================"

mkdir -p data/train
mkdir -p data/test
mkdir -p data/preprocessed
mkdir -p data/torch
mkdir -p cv
mkdir -p tmp
mkdir -p tmp/sampled_files
mkdir -p tmp/validation_files
mkdir -p tmp/submission_files

while [ ! "$(ls -A data/train)" ]; do
    echo "Please copy the data into:"
    echo "* data/train"
    echo "* data/test"
    echo "and press RETURN"
    read _
done

if [ ! -e "data/mean_std.pickle" ]; then
    echo "Calculating mean and std"
    python3 python_utils/calc_mean_std.py
    echo "Done"
fi

echo "Filtering and processing the data"
echo $spacer
if [ "$1" ]; then
  python3 python_utils/modify_data.py -num_val 2 -subsample 3 -subject $1
else
  python3 python_utils/modify_data.py -num_val 2 -subsample 3
fi

echo "Setup done!"
