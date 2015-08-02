import pandas as pd
import argparse
from subprocess import call
from functools import reduce
from random import seed, randint
import shutil

parser = argparse.ArgumentParser(description='Filter out 0 labels from the training set')
parser.add_argument('-n', default=-1, type=int, help='how many files to filter', dest='num_files')
parser.add_argument('-v', default=4, type=int, help='how many files to leave for validation', dest='num_val_files')
parser.add_argument('-c', default=False, action='store_true', help='clear filtered file directory', dest='should_rm')
args = parser.parse_args()

seed(123)

if args.should_rm:
    print('removing old files')
    call(['rm', '-rf', 'data/filtered'])
    call(['mkdir', 'data/filtered'])
    print('done')

data_in_path = "data/train/subj{0}_series{1}_data.csv"
events_in_path = "data/train/subj{0}_series{1}_events.csv"
data_out_path = "data/filtered/subj{0}_series{1}_data.csv"
events_out_path = "data/filtered/subj{0}_series{1}_events.csv"
num_subjects = 12
num_series = 8
offset = 7

val_files = set()
for i in range(0, args.num_val_files):
    indexes = (randint(1, num_subjects), randint(1, num_series))
    while indexes in val_files:
        indexes = (randint(1, num_subjects), randint(1, num_series))
    val_files.add(indexes)
    print('copying subject {} series {} as validation'.format(indexes[0], indexes[1]))
    shutil.copy2(data_in_path.format(indexes[0], indexes[1]),
                 data_out_path.format(indexes[0], indexes[1]) + ".val")
    shutil.copy2(events_in_path.format(indexes[0], indexes[1]),
                 events_out_path.format(indexes[0], indexes[1]) + ".val")

total_samples = 0
total_used_samples = 0
# we want inclusive ranges
for subj in range(1, num_subjects + 1):
    for series in range(1, num_series + 1):
        if (subj, series) in val_files:
            continue

        # load files
        print('reading files for subject {}, series {}'.format(subj, series))
        data_df = pd.read_csv(data_in_path.format(subj, series))
        events_df = pd.read_csv(events_in_path.format(subj, series))

        # tidy up the indexes
        data_df['id'] = data_df['id'].map(lambda x: int(x.split('_')[2]))
        events_df['id'] = events_df['id'].map(lambda x: int(x.split('_')[2]))

        # find event indices
        print('filtering events')
        hs_df = events_df[events_df['HandStart'] != 0].id
        fdt_df = events_df[events_df['FirstDigitTouch'] != 0].id
        bslp_df = events_df[events_df['BothStartLoadPhase'] != 0].id
        lo_df = events_df[events_df['LiftOff'] != 0].id
        r_df = events_df[events_df['Replace'] != 0].id
        br_df = events_df[events_df['BothReleased'] != 0].id
        num_events = int(hs_df.count() / 150)

        # check if it's one of the strange files
        counts = [
            hs_df.count(),
            fdt_df.count(),
            bslp_df.count(),
            lo_df.count(),
            r_df.count(),
            br_df.count()
        ]
        file_is_ok = True
        for i in range(1, len(counts)):
            if counts[0] != counts[i]:
                file_is_ok = False

        if not file_is_ok or counts[0] % 150 != 0:
            print('there is a problem with this file...')
            continue

        # the file has to be ok now
        print('found: ' + str(num_events) + ' events')

        # event_boundaries is a list of pairs (t_start, t_end)
        event_boundaries = []
        for i in range(0, num_events * 150, 150):
            event_boundaries.append((hs_df.iloc[i] - offset, hs_df.iloc[i] + 150 + offset))
            event_boundaries.append((fdt_df.iloc[i] - offset, fdt_df.iloc[i] + 150 + offset))
            event_boundaries.append((bslp_df.iloc[i] - offset, bslp_df.iloc[i] + 150 + offset))
            event_boundaries.append((lo_df.iloc[i] - offset, lo_df.iloc[i] + 150 + offset))
            event_boundaries.append((r_df.iloc[i] - offset, r_df.iloc[i] + 150 + offset))
            event_boundaries.append((br_df.iloc[i] - offset, br_df.iloc[i] + 150 + offset))

        # print some information
        event_lengths = map(lambda ev: ev[1] - ev[0] + 1, event_boundaries)
        used_samples = reduce(lambda x, y: x + y, event_lengths)
        avg_length = used_samples / num_events / 6
        percent_used = float(used_samples) / events_df['id'].count() * 100
        total_samples += events_df['id'].count()
        total_used_samples += used_samples
        print('using {} samples ({:.2f}%)'.format(used_samples, percent_used))
        print('average event length: {}'.format(avg_length))
        sparsity = 1 - (150.) / avg_length
        print('sparsity: {}'.format(sparsity))

        # extract only selected ranges
        data_slices = []
        events_slices = []
        for i in range(0, num_events * 6):
            start, end = event_boundaries[i]
            data_slices.append(data_df.iloc[start:end+1])
            events_slices.append(events_df.iloc[start:end+1])

        # concat and save the slices
        pd.concat(data_slices).to_csv(data_out_path.format(subj, series), index=False)
        pd.concat(events_slices).to_csv(events_out_path.format(subj, series), index=False)

        print('done')

        if args.num_files > -1 and ((subj - 1) * num_series) + series - args.num_val_files >= args.num_files:
            quit()

total_percent_used = float(total_used_samples) / total_samples * 100
print('used {} samples ({:.2f}%)'.format(total_used_samples, total_percent_used))
