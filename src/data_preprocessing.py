import numpy as np

def segmentation(time_series, segment_length_sec, stride_sec, keep_time_stamps=True, input_cadence_sec=4):
    """
    Create a list of 1D (when time_stamps=False) or 2D (when time_stamps=True) arrays, which are overlappig segments of ts.
    Incomplete fragments are rejected.

    time_series = time series to be segmented
    seg_len = length of a segment, 
    stride_sec = step size; difference in the starting position of the consecutive segments
    """
    segment_length = segment_length_sec//input_cadence_sec
    stride = stride_sec//input_cadence_sec
    
    if (segment_length).is_integer() and (stride).is_integer():
        segment_length = int(segment_length)
        stride = int(stride)
    else:
        raise ValueError("segment_length_sec and stride_sec should be multiples of input_cadence_sec")
    
    segments=[]
    for start in range(0, len(time_series[0])-segment_length, stride):
        end=start+segment_length
        if time_series[0][end]-time_series[0][start] != segment_length*input_cadence_sec: #don't allow segments outside of good time intervals
            continue
        if keep_time_stamps==True:
            segments.append(time_series[:,start:end])
        else:
            segments.append(time_series[1:,start:end])
    return np.array(segments)

def binning(time_series, output_cadence, input_cadence=0.125):
    """
    Bin the input time series. First dimension of the time series must be equal to 3. Time series must contain an array of time stamps, 
    count values and uncertainty on the count.
    Make sure that count rates are transformed to count values (uncertainty should be equal to the square root of the count).
    
    time_series = array of size [3, N], where N is the length of the series
    input_cadence = input cadence in seconds
    output_cadence = desired cadence in seconds
    """
    binned_stamps = int(output_cadence/input_cadence) # how many data points to bin
        
    weights = f[2]**-2
    weighted_counts = f[1]*weights # weigh counts by the inverse of squared error
    binned_counts = np.sum(weighted_counts[:(len(weighted_counts)//binned_stamps)*binned_stamps].reshape(-1, binned_stamps), axis=1) # sum weighted counts within each bin
    binned_weights = np.sum(weights[:(len(weights)//binned_stamps)*binned_stamps].reshape(-1, binned_stamps), axis=1) # sum weights within each bin
    binned_counts/=binned_weights # normalise weighted values using sum of weights
    binned_errors = np.sqrt(1.0/(binned_weights)) # calculate uncertainty of each bin
    binned_time = np.mean(f[0][:(len(f[0])//binned_stamps)*binned_stamps].reshape(-1, binned_stamps), axis=1) # find the mean time of each bin
    
    # if bin crosses between two good time intervals, the difference between its binned time and the binned time of preceding bin will not
    # be equal to the desired cadence. Remove those bins from the light curve
    rm_points = []
    skip=False
    for i in range(len(binned_time)-1):
        if skip==True:
            skip=False
            continue
        delta = binned_time[i+1]-binned_time[i]
        if delta > output_cadence:
            rm_points.append(i+1)
            skip=True
    times=np.delete(binned_time,rm_points)
    counts=np.delete(binned_counts,rm_points)
    errors=np.delete(binned_errors,rm_points)
    
    return np.stack((times,counts, errors))


def std1_to_segments(in_data_dir, cadence, seg_len_s, stride_s, random_seed):
    """
    in_data_dir = directory that will be searched for "*_std1_lc.txt" files containing Standard1 light curve data
    cadence = desired amount of time between data points of the final segments, unit of seconds, should be a multiple of 0.125 (std1 resolution)
    seg_len_s = desired segment length in seconds
    stride_s = time difference between consecutive segments; stride size of the moving window in seconds
    random_seed = set the seed of the numpy random state
    
    returns segments_counts, segments_errors, id_per_seg
    """
    np.random.seed(seed=random_seed)
    
    lcs = []
    ids=[]

    binned_stamps = int(cadence/0.125) # how many time stamps go into one bin

    for root, dirnames, filenames in os.walk(in_data_dir): #Std1_PCU2
        for filename in fnmatch.filter(filenames, "*_std1_lc.txt"):
            lc = os.path.join(root, filename)
            ids.append(filename.split("_")[0])
            f=np.loadtxt(lc)
            f=np.transpose(f)#,axis=1)
            
            binned_lc = binning(f, bin_size=binned_stamps)
            lcs.append(binned_lc)
    
    print("Binned {} light curves.".format(len(lcs)))
    clear_output(wait=True)
            
    segments_counts=[]
    segments_errors=[]
    seg_ids=[]


    seg_len = seg_len_s//cadence # segment length and stride size in data points
    stride = stride_s//cadence



    for lc_index, lc in enumerate(lcs):
        if len(lc[1]) >= seg_len: 
            segments = segmentation(lc, seg_len, stride, keep_time_stamps=False, experimental = False)
        else:
            continue
        if len(segments) > 0:
            segments_counts.append(segments[:,0,:])
            segments_errors.append(segments[:,1,:])
            seg_ids.append(ids[lc_index])
            print("Segmented {}/{} light curves.".format(lc_index+1, len(lcs)))
            clear_output(wait=True)
    
    print("Stacking the segments and creating segment IDs, shuffling.")
    id_per_seg = []  # for each light curve, copy the observation id for every segment of the light curve
    for lc_index, lc in enumerate(segments_counts):
        for i in range(len(lc)):
            id_per_seg.append(seg_ids[lc_index]+"_{}".format(i))

    segments_counts=np.vstack(segments_counts)
    segments_errors=np.vstack(segments_errors)
    segments_counts = np.expand_dims(segments_counts, axis=-1)
    segments_errors = np.expand_dims(segments_errors, axis=-1)
    
    rng_state = np.random.get_state()
    np.random.shuffle(segments_counts)
    np.random.set_state(rng_state)
    np.random.shuffle(segments_errors)
    np.random.set_state(rng_state)
    np.random.shuffle(id_per_seg)

    print("Done")
    
    return segments_counts, segments_errors, id_per_seg