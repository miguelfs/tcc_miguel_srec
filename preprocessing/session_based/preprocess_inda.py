import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

#data config (all methods)
DATA_PATH = '../data/raw/'
DATA_PATH_PROCESSED = '../data/prepared/'
#DATA_FILE = 'yoochoose-clicks-10M'
# DATA_FILE = 'yoochoose-clicks-1M'
DATA_FILE = 'inda'

#filtering config (all methods)
MIN_SESSION_LENGTH = 2
MIN_ITEM_SUPPORT = 5

# keys
TYPE_KEY = 'action'
USER_KEY='UserId'
ITEM_KEY='ItemId'
TIME_KEY='Time'
SESSION_KEY='SessionId'

#min date config
MIN_DATE = '2023-01-01'

#slicing default config
NUM_SLICES = 10
DAYS_OFFSET = 0
DAYS_SHIFT = 5
DAYS_TRAIN = 9
#days test default config
DAYS_TEST = 1


#preprocessing from original gru4rec
def preprocess_org( path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH ):

    data = load_data( path+file )
    data = filter_data( data, min_item_support, min_session_length )
    split_data_org( data, path_proc+file )

#preprocessing from original gru4rec but from a certain point in time
def preprocess_org_min_date( path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH, min_date=MIN_DATE ):

    data = load_data( path+file )
    data = filter_data( data, min_item_support, min_session_length )
    data = filter_min_date( data, min_date )
    split_data_org( data, path_proc+file )

#preprocessing adapted from original gru4rec
def preprocess_days_test( path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH, days_test=DAYS_TEST ):

    data = load_data( path+file )
    data = filter_data( data, min_item_support, min_session_length )
    split_data( data, path_proc+file, days_test )

#preprocessing to create data slices with a window
def preprocess_slices( path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH,
                       num_slices = NUM_SLICES, days_offset = DAYS_OFFSET, days_shift = DAYS_SHIFT, days_train = DAYS_TRAIN, days_test=DAYS_TEST ):

    data = load_data( path+file )
    data = filter_data( data, min_item_support, min_session_length )
    slice_data( data, path_proc+file, num_slices, days_offset, days_shift, days_train, days_test )

#just load and show info
def preprocess_info( path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH ):

    data = load_data( path+file )
    data = filter_data( data, min_item_support, min_session_length )

#just load and show info
def preprocess_buys( path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED ):

    data = load_data( path+file )
    data.to_csv( path_proc + file + '.txt', sep='\t', index=False)



def load_data( file ) :

    #load csv
    data_columns = [SESSION_KEY, ITEM_KEY, TIME_KEY]
    data = pd.read_csv(file+'.csv', usecols=data_columns)
    #
    # data = pd.read_csv( file+'.csv', sep=',', header=None, usecols=[0,1,2], dtype={0:np.int32, 1:str, 2:np.int64})
    #specify header names
    data.columns = [SESSION_KEY, ITEM_KEY, TIME_KEY]

    #output
    data_start = datetime.fromtimestamp( data[TIME_KEY].min(), timezone.utc )
    data_end = datetime.fromtimestamp( data[TIME_KEY].max(), timezone.utc )

    print('Loaded data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format( len(data), data[SESSION_KEY].nunique(), data[ITEM_KEY].nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )

    return data;


def filter_data( data, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH ) :

    #y?
    session_lengths = data.groupby(SESSION_KEY).size()
    data = data[np.in1d(data[SESSION_KEY], session_lengths[ session_lengths >= min_session_length ].index)]

    #filter item support
    item_supports = data.groupby(ITEM_KEY).size()
    data = data[np.in1d(data[ITEM_KEY], item_supports[ item_supports>= min_item_support ].index)]

    #filter session length
    session_lengths = data.groupby(SESSION_KEY).size()
    data = data[np.in1d(data[SESSION_KEY], session_lengths[ session_lengths >= min_session_length ].index)]

    #output
    data_start = datetime.fromtimestamp( data[TIME_KEY].min(), timezone.utc )
    data_end = datetime.fromtimestamp( data[TIME_KEY].max(), timezone.utc )

    print('Filtered data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format( len(data), data[SESSION_KEY].nunique(), data[ITEM_KEY].nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )

    return data;

def filter_min_date( data, min_date=MIN_DATE ) :

    min_datetime = datetime.strptime(min_date + ' 00:00:00', '%Y-%m-%d %H:%M:%S')

    #filter
    session_max_times = data.groupby(SESSION_KEY)[TIME_KEY].max()
    session_keep = session_max_times[ session_max_times > min_datetime.timestamp() ].index

    data = data[ np.in1d(data[SESSION_KEY], session_keep) ]

    #output
    data_start = datetime.fromtimestamp( data[TIME_KEY].min(), timezone.utc )
    data_end = datetime.fromtimestamp( data[TIME_KEY].max(), timezone.utc )

    print('Filtered data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format( len(data), data[SESSION_KEY].nunique(), data[ITEM_KEY].nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )

    return data;



def split_data_org( data, output_file ) :

    tmax = data[TIME_KEY].max()
    session_max_times = data.groupby(SESSION_KEY)[TIME_KEY].max()
    session_train = session_max_times[session_max_times < tmax-86400].index
    session_test = session_max_times[session_max_times >= tmax-86400].index
    train = data[np.in1d(data[SESSION_KEY], session_train)]
    test = data[np.in1d(data[SESSION_KEY], session_test)]
    test = test[np.in1d(test[ITEM_KEY], train[ITEM_KEY])]
    tslength = test.groupby(SESSION_KEY).size()
    test = test[np.in1d(test[SESSION_KEY], tslength[tslength>=2].index)]
    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train[SESSION_KEY].nunique(), train[ITEM_KEY].nunique()))
    train.to_csv(output_file + '_train_full.txt', sep='\t', index=False)
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test[SESSION_KEY].nunique(), test[ITEM_KEY].nunique()))
    test.to_csv(output_file + '_test.txt', sep='\t', index=False)

    tmax = train[TIME_KEY].max()
    session_max_times = train.groupby(SESSION_KEY)[TIME_KEY].max()
    session_train = session_max_times[session_max_times < tmax-86400].index
    session_valid = session_max_times[session_max_times >= tmax-86400].index
    train_tr = train[np.in1d(train[SESSION_KEY], session_train)]
    valid = train[np.in1d(train[SESSION_KEY], session_valid)]
    valid = valid[np.in1d(valid[ITEM_KEY], train_tr[ITEM_KEY])]
    tslength = valid.groupby(SESSION_KEY).size()
    valid = valid[np.in1d(valid[SESSION_KEY], tslength[tslength>=2].index)]
    print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_tr), train_tr[SESSION_KEY].nunique(), train_tr[ITEM_KEY].nunique()))
    train_tr.to_csv( output_file + '_train_tr.txt', sep='\t', index=False)
    print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid[SESSION_KEY].nunique(), valid[ITEM_KEY].nunique()))
    valid.to_csv( output_file + '_train_valid.txt', sep='\t', index=False)



def split_data( data, output_file, days_test=DAYS_TEST, last_nth=None ) :

    data_end = datetime.fromtimestamp( data[TIME_KEY].max(), timezone.utc )
    test_from = data_end - timedelta( days_test )

    session_max_times = data.groupby(SESSION_KEY)[TIME_KEY].max()
    session_train = session_max_times[ session_max_times < test_from.timestamp() ].index
    session_test = session_max_times[ session_max_times >= test_from.timestamp() ].index
    train = data[np.in1d(data[SESSION_KEY], session_train)]

    if last_nth is not None:
        train.sort_values([SESSION_KEY, TIME_KEY], inplace=True)

        session_data = list(data[SESSION_KEY].values)
        lenth = int(len(session_data) / last_nth)
        session_data = session_data[-lenth:]
        for i in range(len(session_data)):
            if session_data[i] != session_data[i+1]:
                break

        train = data.reset_index()
        train = train[-lenth + i + 1:]

    test = data[np.in1d(data[SESSION_KEY], session_test)]
    test = test[np.in1d(test[ITEM_KEY], train[ITEM_KEY])]
    tslength = test.groupby(SESSION_KEY).size()
    test = test[np.in1d(test[SESSION_KEY], tslength[tslength>=2].index)]
    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train[SESSION_KEY].nunique(), train[ITEM_KEY].nunique()))
    train.to_csv(output_file + (str(last_nth) if last_nth is not None else '') + '_train_full.txt', sep='\t', index=False)
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test[SESSION_KEY].nunique(), test[ITEM_KEY].nunique()))
    test.to_csv(output_file + (str(last_nth) if last_nth is not None else '') + '_test.txt', sep='\t', index=False)

    data_end = datetime.fromtimestamp( train[TIME_KEY].max(), timezone.utc )
    test_from = data_end - timedelta( days_test )

    session_max_times = train.groupby(SESSION_KEY)[TIME_KEY].max()
    session_train = session_max_times[ session_max_times < test_from.timestamp() ].index
    session_valid = session_max_times[ session_max_times >= test_from.timestamp() ].index
    train_tr = train[np.in1d(train[SESSION_KEY], session_train)]
    valid = train[np.in1d(train[SESSION_KEY], session_valid)]
    valid = valid[np.in1d(valid[ITEM_KEY], train_tr[ITEM_KEY])]
    tslength = valid.groupby(SESSION_KEY).size()
    valid = valid[np.in1d(valid[SESSION_KEY], tslength[tslength>=2].index)]
    print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_tr), train_tr[SESSION_KEY].nunique(), train_tr[ITEM_KEY].nunique()))
    train_tr.to_csv( output_file + (str(last_nth) if last_nth is not None else '') + '_train_tr.txt', sep='\t', index=False)
    print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid[SESSION_KEY].nunique(), valid[ITEM_KEY].nunique()))
    valid.to_csv( output_file + (str(last_nth) if last_nth is not None else '') + '_train_valid.txt', sep='\t', index=False)


def slice_data( data, output_file, num_slices=NUM_SLICES, days_offset=DAYS_OFFSET, days_shift=DAYS_SHIFT, days_train=DAYS_TRAIN, days_test=DAYS_TEST ):

    for slice_id in range( 0, num_slices ) :
        split_data_slice( data, output_file, slice_id, days_offset+(slice_id*days_shift), days_train, days_test )

def split_data_slice( data, output_file, slice_id, days_offset, days_train, days_test ) :

    data_start = datetime.fromtimestamp( data[TIME_KEY].min(), timezone.utc )
    data_end = datetime.fromtimestamp( data[TIME_KEY].max(), timezone.utc )

    print('Full data set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}'.
          format( slice_id, len(data), data[SESSION_KEY].nunique(), data[ITEM_KEY].nunique(), data_start.isoformat(), data_end.isoformat() ) )


    start = datetime.fromtimestamp( data[TIME_KEY].min(), timezone.utc ) + timedelta( days_offset )
    middle =  start + timedelta( days_train )
    end =  middle + timedelta( days_test )

    #prefilter the timespan
    session_max_times = data.groupby(SESSION_KEY)[TIME_KEY].max()
    greater_start = session_max_times[session_max_times >= start.timestamp()].index
    lower_end = session_max_times[session_max_times <= end.timestamp()].index
    data_filtered = data[np.in1d(data[SESSION_KEY], greater_start.intersection( lower_end ))]

    print('Slice data set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {} / {}'.
          format( slice_id, len(data_filtered), data_filtered[SESSION_KEY].nunique(), data_filtered[ITEM_KEY].nunique(), start.date().isoformat(), middle.date().isoformat(), end.date().isoformat() ) )

    #split to train and test
    session_max_times = data_filtered.groupby(SESSION_KEY)[TIME_KEY].max()
    sessions_train = session_max_times[session_max_times < middle.timestamp()].index
    sessions_test = session_max_times[session_max_times >= middle.timestamp()].index

    train = data[np.in1d(data[SESSION_KEY], sessions_train)]

    print('Train set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}'.
          format( slice_id, len(train), train[SESSION_KEY].nunique(), train[ITEM_KEY].nunique(), start.date().isoformat(), middle.date().isoformat() ) )

    train.to_csv(output_file + '_train_full.'+str(slice_id)+'.txt', sep='\t', index=False)

    test = data[np.in1d(data[SESSION_KEY], sessions_test)]
    test = test[np.in1d(test[ITEM_KEY], train[ITEM_KEY])]

    tslength = test.groupby(SESSION_KEY).size()
    test = test[np.in1d(test[SESSION_KEY], tslength[tslength>=2].index)]

    print('Test set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {} \n\n'.
          format( slice_id, len(test), test[SESSION_KEY].nunique(), test[ITEM_KEY].nunique(), middle.date().isoformat(), end.date().isoformat() ) )

    test.to_csv(output_file + '_test.'+str(slice_id)+'.txt', sep='\t', index=False)


# ------------------------------------- 
# MAIN TEST
# --------------------------------------
if __name__ == '__main__':

    preprocess_slices();
