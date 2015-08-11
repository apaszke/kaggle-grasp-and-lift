#!/bin/bash
spacer="========================================"

mkdir -p data/train
mkdir -p data/test
mkdir -p data/filtered
mkdir -p data/preprocessed
mkdir -p cv

while [ ! "$(ls -A data/train)" ]; do
    echo "Please copy the data into:"
    echo "* data/train"
    echo "* data/test"
    echo "and press RETURN"
    read _
done

if [ ! -e "data/mean_std.pickle" ]; then
    echo "Calculating mean and std"
    python3 calc_mean_std.py
    echo "Done"
fi

echo "Filtering the data"
echo $spacer
python3 python_utils/modify_data.py -c -v 2 -s 5

echo "Setup done!"
