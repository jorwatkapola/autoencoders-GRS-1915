import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored



def simple_segmentation(time_series, segment_length, stride, keep_time_stamps, input_cadence_sec):
    
    segments = []
    for start in range(0, int(len(time_series[0])-segment_length), stride): 
        end=start+segment_length
        # test that the segment has the expected duration in seconds
        # if it doesn't then the segment likely goes outside the GTI
        new_segment_time = time_series[0][start:end+1]
        new_segment_duration_sec = new_segment_time[-1]-new_segment_time[0]
        if new_segment_duration_sec != segment_length*input_cadence_sec: #don't allow segments outside of good time intervals
            print("segment outside of good time intervals")
            continue
        if keep_time_stamps==True:
            segments.append(time_series[:,start:end])
        else:
            segments.append(time_series[1:,start:end])
            
    return np.array(segments)

def segmentation(time_series, segment_length_sec, stride_sec, keep_time_stamps=True, input_cadence_sec=4):
    """
    Create a list of 1D (when time_stamps=False) or 2D (when time_stamps=True) arrays, which are overlappig segments of ts.
    Incomplete fragments are rejected.

    time_series = time series to be segmented
    seg_len = length of a segment, 
    stride_sec = step size; difference in the starting position of the consecutive segments
    """
    segment_length = int(segment_length_sec//input_cadence_sec)
    stride = int(stride_sec//input_cadence_sec)
    
    time_deltas = np.diff(time_series[0])
    gaps = np.where(time_deltas>input_cadence_sec)[0]
    
    segments_all=[]

    if len(gaps) == 0:
        segments_all = simple_segmentation(time_series, segment_length, stride, keep_time_stamps, input_cadence_sec)
                
    else:
        for gap_ind, gap in enumerate(gaps):
            if gap_ind == 0:
                segments_all.append(simple_segmentation(time_series[:,:gap+1], segment_length, stride, keep_time_stamps, input_cadence_sec))
                
            else:
                segments_all.append(simple_segmentation(time_series[:,gaps[gap_ind-1]+1:gap+1], segment_length, stride, keep_time_stamps, input_cadence_sec))
                
            if gap == gaps[-1]:
                segments_all.append(simple_segmentation(time_series[:,gap+1:], segment_length, stride, keep_time_stamps, input_cadence_sec))
                
        
        segments_all = [item for sublist in segments_all for item in sublist] # vstack the list of lists of data segments
    
    return np.array(segments_all)

def verify_not_segmented_light_curves(lcs, ob_state, seg_ids, lc_ids, seg_len_sec=1024):
    """
    Plot and show GTI length of the light curves which were not segmented
    
    Parameters:
    --------------
    lcs: list of ndarrays 
        ndarrays of shape (3,light_curve_length) containing time, count rate and count rate error data
        
    ob_state: dict
        Classifications of light curves in the form {observation_ID: class_label}
        
    seg_ids: list of strings
        identifiers for segment light curves in the form of  "{}_{}".format(observation_ID, integer)
    
    lc_ids: list of strings
        identifiers for light curvesin the for of observation_ID, e.g. "10408-01-17-03"
    
    """
    rejected_lcs_with_label = 0
    for k,v in ob_state.items():
        if k not in [x.split("_")[0] for x in seg_ids]:
            print(k, v)
            try:
                n = np.where(np.array(lc_ids) == k)[0][0]
                plt.plot(lcs[n][0], lcs[n][1])
                plt.show()
                cadence = np.min(np.diff(lcs[n][0]))
                print("sum of GTI durations in seconds: ", (len(lcs[n][0])-1)*cadence)
                time_deltas = np.diff(lcs[n][0])
                gaps = np.where(time_deltas>cadence)[0]
                if len(gaps)>0:
                    print("individual GTI durations in seconds:")
                    for gap_ind, gap in enumerate(gaps):
                        if gap_ind == 0:
                            if gap*cadence >= seg_len_sec:
                                print(gap*cadence, colored("GTI seems long enough for segmentation", "red"))
                            else:
                                print(gap*cadence)
                        else:
                            if (gap-(gaps[gap_ind-1]+1))*cadence >= seg_len_sec:
                                print((gap-(gaps[gap_ind-1]+1))*cadence, colored("GTI seems long enough for segmentation", "red"))
                            else:
                                print((gap-(gaps[gap_ind-1]+1))*cadence)

                        if gap == gaps[-1]:
                            if (len(lcs[n][0])-(gap+1))*cadence >= seg_len_sec:
                                print((len(lcs[n][0])-(gap+1))*cadence, colored("GTI seems long enough for segmentation", "red"))
                            else:
                                print((len(lcs[n][0])-(gap+1))*cadence)
                            
                elif (len(lcs[n][0])-1)*cadence >= seg_len_sec:
                    print(colored("Light curve seems long enough for segmentation", "red"))
                    
            except:
                print("No data")
            rejected_lcs_with_label+=1
            print("\n\n")
    print(rejected_lcs_with_label)


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