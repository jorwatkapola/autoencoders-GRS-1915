"""
Code used to split data before implementing stratification
"""


# calculate total number of data points
total_data_volume = 0
for lc in lcs:
    total_data_volume += len(lc[0])
print(total_data_volume)


# split into subsets according to observation length

needed_validation_data = total_data_volume*0.1
needed_testing_data = total_data_volume*0.2

shuffle_indices = np.array(range(len(lcs)))
np.random.seed(seed=11)
np.random.shuffle(shuffle_indices)


test_set_obs = []
test_set_size = 0
for ob_index in shuffle_indices:
    test_set_obs.append(lc_ids[ob_index])
    test_set_size += len(lcs[ob_index][0])
    if test_set_size >= needed_testing_data:
        break

valid_set_obs = []
valid_set_size = 0
for ob_index_val in shuffle_indices[len(test_set_obs):]:
    valid_set_obs.append(lc_ids[ob_index_val])
    valid_set_size += len(lcs[ob_index_val][0])
    if valid_set_size >= needed_validation_data:
        break


train_set_obs = np.take(lc_ids, shuffle_indices[len(test_set_obs)+len(valid_set_obs):])

split_indices = [train_set_obs, valid_set_obs, test_set_obs]

print("Test set ", test_set_size/total_data_volume)
print("Validation set percentage", valid_set_size/total_data_volume)
print("Training set percentage", (total_data_volume-valid_set_size-test_set_size)/total_data_volume)


print("Observation ID intersection between: \ntest-valid {} \ntest-train {} \nvalid-train sets {}".format(
      len([ob for ob in test_set_obs if ob in valid_set_obs]),
      len([ob for ob in test_set_obs if ob in train_set_obs]), 
      len([ob for ob in valid_set_obs if ob in train_set_obs])))
print()
print("Sum of train/val/test sizes: {}".format(np.sum([len(subset) for subset in split_indices])))

# with open('{}/lightcurve1776_train70_val10_test20.pkl'.format(data_dir), 'wb') as f:
#     pickle.dump(split_indices, f)