import pandas as pd
import argparse
from subprocess import call
from functools import reduce
from random import seed, randint
from scipy.signal import butter, lfilter
import shutil
import pickle

parser = argparse.ArgumentParser(description='Filter out 0 labels from the training set')
parser.add_argument('-n', default=-1, type=int, help='how many files to filter', dest='num_files')
parser.add_argument('-s', default=1, type=int, help='subsampling', dest='subsample')
parser.add_argument('-v', default=4, type=int, help='how many files to leave for validation', dest='num_val_files')
parser.add_argument('-o', default=10, type=int, help='', dest='offset')
parser.add_argument('-subject', default=-1, type=int, help='', dest='subject')
parser.add_argument('-f', default=False, action='store_true', help='clear filtered file directory', dest='filter')
args = parser.parse_args()

data_in_path = "data/train/subj{0}_series{1}_data.csv"
events_in_path = "data/train/subj{0}_series{1}_events.csv"
data_out_path = "data/filtered/subj{0}_series{1}_data.csv"
events_out_path = "data/filtered/subj{0}_series{1}_events.csv"
num_subjects = 12
num_series = 8
total_samples = 0
total_used_samples = 0
event_length = 150 // args.subsample
if args.subject == -1:
  subjects = range(1, num_subjects + 1)
else
  subjects = [args.subject]

seed(123)

if 150 % args.subsample != 0:
  print("Subsample should divide 150!")
  quit()

# remove old files
print('removing old files...')
call(['rm', '-rf', 'data/filtered'])
call(['mkdir', 'data/filtered'])

# save information about the data
with open('data/filtered/info', 'w') as f:
    f.write(str(args.offset))
    f.write('\n')
    f.write(str(args.subsample))
    f.write('\n')


with open('./data/mean_std.pickle', 'rb') as f:
    mean, std = pickle.load(f)

# select some files for validation
val_files = set()
for i in range(0, args.num_val_files):
    indexes = (randint(1, num_subjects), randint(1, num_series))
    while indexes in val_files:
        indexes = (randint(1, num_subjects), randint(1, num_series))
    val_files.add(indexes)


sparsity = 1 - event_length / (event_length + 2 * args.offset // args.subsample)
print('output sparsity: {:.3f}'.format(sparsity))

def filterData(data_df, events_df):
    global args
    global event_length
    # find event indices
    hs_df = events_df[events_df['HandStart'] != 0].id
    fdt_df = events_df[events_df['FirstDigitTouch'] != 0].id
    bslp_df = events_df[events_df['BothStartLoadPhase'] != 0].id
    lo_df = events_df[events_df['LiftOff'] != 0].id
    r_df = events_df[events_df['Replace'] != 0].id
    br_df = events_df[events_df['BothReleased'] != 0].id
    num_events = hs_df.count() // event_length

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

    if not file_is_ok or counts[0] % event_length != 0:
        print('there is a problem with this file...')
        return None, None

    # the file has to be ok now
    print('found:     ' + str(num_events) + ' events')

    # event_boundaries is a list of pairs (t_start, t_end)
    event_boundaries = []
    for i in range(0, num_events * event_length, event_length):
        event_boundaries.append((hs_df.iloc[i] - args.offset, hs_df.iloc[i] + 150 + args.offset))
        event_boundaries.append((fdt_df.iloc[i] - args.offset, fdt_df.iloc[i] + 150 + args.offset))
        event_boundaries.append((bslp_df.iloc[i] - args.offset, bslp_df.iloc[i] + 150 + args.offset))
        event_boundaries.append((lo_df.iloc[i] - args.offset, lo_df.iloc[i] + 150 + args.offset))
        event_boundaries.append((r_df.iloc[i] - args.offset, r_df.iloc[i] + 150 + args.offset))
        event_boundaries.append((br_df.iloc[i] - args.offset, br_df.iloc[i] + 150 + args.offset))

    # extract only selected ranges
    data_slices = []
    events_slices = []
    for i in range(0, num_events * 6):
        start, end = event_boundaries[i]
        data_slices.append(data_df.loc[start:end])
        events_slices.append(events_df.loc[start:end])
    # concat and save the slices
    data_pd = pd.concat(data_slices)
    events_pd = pd.concat(events_slices)

    return data_pd, events_pd




# we want inclusive ranges
for subj in subjects:
    for series in range(1, num_series + 1):

        # load files
        print('reading file for subject {}, series {}'.format(subj, series))
        data_df = pd.read_csv(data_in_path.format(subj, series))
        events_df = pd.read_csv(events_in_path.format(subj, series))
        num_samples = data_df['id'].count()

        # handle subsampling
        if args.subsample > 1:
            data_df = data_df.ix[([i for i in range(0, num_samples) if i % args.subsample == 0])]
            events_df = events_df.ix[([i for i in range(0, num_samples) if i % args.subsample == 0])]

        # tidy up the indexes
        data_id = pd.DataFrame(data_df['id'].map(lambda x: int(x.split('_')[2])).values)
        events_df['id'] = events_df['id'].map(lambda x: int(x.split('_')[2]))

        data_df.drop('id', axis=1, inplace=True)
        b,a = butter(3,2/250.0,btype='lowpass')
        data_df = pd.DataFrame(lfilter(b, a, data_df, axis=0))
        data_df.insert(0, 'id', data_id)

        # handle validation files
        if (subj, series) in val_files:
            print('copying as validation')
            data_df = (data_df - mean) / std
            data_df.to_csv(data_out_path.format(subj, series) + '.val', index=False)
            events_df.to_csv(events_out_path.format(subj, series) + '.val', index=False)
            continue

        # extract events with neighbourhoods
        if args.filter:
            data_df, events_df = filterData(data_df, events_df)
            if data_df is None:
              continue

        # print some information
        used_samples = data_df['id'].count()
        percent_used = used_samples / num_samples * 100
        total_samples += num_samples
        total_used_samples += used_samples
        print('using:     {} samples ({:.2f}%)'.format(used_samples, percent_used))

        data_df = (data_df - mean) / std

        data_df.to_csv(data_out_path.format(subj, series), index=False)
        events_df.to_csv(events_out_path.format(subj, series), index=False)

        # check if we have exceded the file limit
        if args.num_files > -1 and ((subj - 1) * num_series) + series - args.num_val_files >= args.num_files:
            quit()

total_percent_used = total_used_samples / total_samples * 100
print('used {} samples ({:.2f}%)'.format(total_used_samples, total_percent_used))
