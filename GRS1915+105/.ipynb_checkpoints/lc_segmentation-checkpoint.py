import numpy as np
import pickle
from scipy.stats import zscore
import datetime
import pytz



with open('../../../data_GRS1915/1776_light_curves_1s_bin.pkl', 'rb') as f:
    lcs = pickle.load(f)
with open('../../../data_GRS1915/1776_light_curves_1s_bin_ids.pkl', 'rb') as f:
    ids = pickle.load(f)
    
    
clean_belloni = open('../../../data_GRS1915/1915Belloniclass_updated.dat')
lines = clean_belloni.readlines()
states = lines[0].split()
belloni_clean = {}
for h,l in zip(states, lines[1:]):
    belloni_clean[h] = l.split()
    #state: obsID1, obsID2...
ob_state = {}
for state, obs in belloni_clean.items():
    if state == "chi1" or state == "chi2" or state == "chi3" or state == "chi4": state = "chi"
    for ob in obs:
        ob_state[ob] = state

# lcs_4s = []
# lcs_4s_ids = []
# for lc_ind, lc in enumerate(lcs):
#     ###1s average and time check to eliminate points outside of GTIs
#     f4t = np.mean(lc[0][:(len(lc[0])//4)*4].reshape(-1, 4), axis=1)
#     f4c = np.mean(lc[1][:(len(lc[1])//4)*4].reshape(-1, 4), axis=1)
#     f4e = np.mean(lc[2][:(len(lc[1])//4)*4].reshape(-1, 4), axis=1)
#     rm_points = []
#     skip=False
#     for i in range(len(f4t)-1):
#         if skip==True:
#             skip=False
#             continue
#         delta = f4t[i+1]-f4t[i]
#         if delta > 4.0:
#             rm_points.append(i+1)
#             skip=True

#     times=np.delete(f4t,rm_points)
#     counts=np.delete(f4c,rm_points)
#     errors=np.delete(f4e,rm_points)
#     lcs_4s.append(np.stack((times,counts, errors)))
#     lcs_4s_ids.append(ids[lc_ind])
    
    
def segmentation(time_series, seg_len, stride, keep_time_stamps=True, experimental = False):
    """
    Create a list of 1D (when time_stamps=False) or 2D (when time_stamps=True) arrays, which are overlappig segments of ts. Incomplete fragments are rejected.
    
    time_series = time series to be segmented
    seg_len = length of a segment, 
    stride = step size; difference in the starting position of the consecutive segments
    """
    
    
    segments=[]
    for start in range(0, len(time_series[0])-seg_len, stride):
        end=start+seg_len
        ############################################# *4 because of the 4 second cadance 
        if time_series[0][end]-time_series[0][start] != seg_len: #don't allow temporally discontinous segments
            continue
        if keep_time_stamps==True:
            segments.append(time_series[:,start:end])
        else:
            segments.append(time_series[1:,start:end])
    return np.array(segments) # check why time stamps are kept 


segments_counts=[]
segments_errors=[]
seg_ids=[]
for lc_index, lc in enumerate(lcs):
    if len(lc[1]) >= 1024: 
        segments = segmentation(lc, 1024, 64, keep_time_stamps=False, experimental = False)
    else:
        continue
    if len(segments) > 0:
        segments_counts.append(segments[:,0,:])
        segments_errors.append(segments[:,1,:])
        seg_ids.append(ids[lc_index])
        print(lc_index+1, "/{}".format(len(lcs)))
        clear_output(wait=True)
        
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


with open('../../../data_GRS1915/94465_len512_s40_counts.pkl', 'wb') as f:
    pickle.dump(segments_counts, f)
    
with open('../../../data_GRS1915/94465_len512_s40_errors.pkl', 'wb') as f:
    pickle.dump(segments_errors, f)
    
with open('../../../data_GRS1915/94465_len512_s40_ids.pkl', 'wb') as f:
    pickle.dump(id_per_seg, f)