Kaggle Grasp-and-Lift Detection
===============================================

Code in this repository can be used to train and sample both LSTM and CNN (quite experimental) models on Kaggle Grasp-and-Lift EEG Detection competition data.

I have no idea what it's leaderboard score is because I forgot about the entry deadline...

Notes
-----

These models probably aren't performing really well. I had little knowledge in signal processing and EEG domain and I've not spent enough time on this competition to get satisfying results. Anyway, it was a great opportunity to learn how LSTMs work and how to use python for data processing.

Setup
-----

After cloning the repo you should run `setup.sh`, which will prepare the directory structure and preprocess the data. It should be ready to work afterwards.

Main scripts
------------

There are two other scripts attached:
* `do_subject.sh num` - trains an LSTM model for subject number `num` and evaluates it on validation set.
* `do_submission.sh` - trains an LSTM model for each subject separately, and generates both validation and submission files

Pipeline
--------

First python scripts are used for preprocessing.

1. `calc_mean_std.py` applies low pass filter, calculates mean and std of the data and saves it in `data/mean_std.pickle` for future use.
2. `modify_data.py` applies preprocessing, selects some files for validation and saves them in `data/preprocessed` (validation files have `.val` appended to their name).
3. `train_lstm.lua`/`train_conv.lua` reads preprocessed data, puts it into Torch tensors and caches them in `data/torch`. Then it trains the corresponding model saving checkpoints in `cv/` folder.
4. `sample.lua` recreates model from specified checkpoint and inputs validation or test data through it. Sampled files are saved in `tmp/sampled_files`.
5. `calc_roc.py` reads files from `tmp/sampled_files`, finds corresponding event files and compares them to generate ROC curves and calculate AUC.

How to use included scripts?
----------------------------

All python and lua scripts can be called with `--help` flag and list supported arguments.
