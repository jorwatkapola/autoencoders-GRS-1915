from scipy.spatial import distance
import networkx 
from networkx.algorithms.components.connected import connected_components
from  sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from sklearn.mixture import GaussianMixture
import pandas as pd
from IPython.display import clear_output


def component_mahalanobis_distances(GMmodel):
    """
    input = Gaussian mixture model
    output = matrix of mahalanobis distances between Gaussian components
    """
    no_components = GMmodel.means_.shape[0]
    GM_comp_mahal_distances = np.zeros((no_components,no_components))
    for comp1_ind, comp1 in enumerate(GMmodel.means_):
        for comp2_ind, comp2 in enumerate(GMmodel.means_):
            GM_comp_mahal_distances[comp1_ind, comp2_ind] = distance.mahalanobis(comp1, comp2, np.linalg.inv(GMmodel.covariances_[comp2_ind]))
            if comp2_ind%10 == 0 or comp2_ind+1 == len(GMmodel.means_):
                print(comp1_ind, comp2_ind)
                clear_output(wait=True)
    return GM_comp_mahal_distances



def merge_gaussian_component_labels(distance_matrix, observation_labels, sigma_threshold):
    """
        if mahalanobis distance between Gaussian components is smaller than the sigma_threshold, relabel the data within
    those components as belonging to a new, merged cluster
    
    input:
        distance_matrix - square matrix containing distances between Gaussian component means, output of component_mahalanobis_distances function
        observation_labels - list of data labels corresponding to the Gaussian components
        sigma_threshold - threshold for the merger of Gaussian components. If distances between two component means are less then the threshold,
        (both when calculating from A to B and from B to A), the components will be treated as one cluster.
                            
    output
        new_observation_labels - modified observation_labels, where a new index was created for merged components, and both component indices were
        replaced with that new index
    

    uses the graph solution from https://stackoverflow.com/questions/4842613/merge-lists-that-share-common-elements
    """
    # find pairs of components whose means are separated by mahalanobis distance smaller than the threshold (both ways)
    couples = np.array(np.where(((np.triu(distance_matrix)<sigma_threshold)&(np.triu(distance_matrix)>0))
      &(np.triu(distance_matrix.T)<sigma_threshold)&(np.triu(distance_matrix.T)>0))).T
    
    # build a graph of connections
    l = []
    for couple in couples:
        l.append([str(c) for c in couple])
    def to_graph(l):
        G = networkx.Graph()
        for part in l:
            # each sublist is a bunch of nodes
            G.add_nodes_from(part)
            # it also imlies a number of edges:
            G.add_edges_from(to_edges(part))
        return G
    def to_edges(l):
        """ 
            treat `l` as a Graph and returns it's edges 
            to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
        """
        it = iter(l)
        last = next(it)

        for current in it:
            yield last, current
            last = current    
    G = to_graph(l)

    new_observation_labels = observation_labels
    # merge components
    for n_connection, connection in enumerate(connected_components(G)):
        node_indices=np.array([int(node) for node in connection])
#         new_observation_labels = np.where(np.isin(new_observation_labels, node_indices),
#                                           n_connection+np.unique(observation_labels).shape[0], new_observation_labels)
        new_observation_labels = np.where(np.isin(new_observation_labels, node_indices),
                                          node_indices[0], new_observation_labels)
        
    return new_observation_labels


def grid_search_random_forest(data_labels, data, train_set_ids, val_set_ids, seg_ObIDs, ob_state):
    """
    prepare representations of observations based on their make up in terms of Gaussian mixture component contributions.
    test the representation as the feature set for classification task
    """
        
    # find GMM component labels for data
    data_GMMcomp_labels = data_labels
    
    # make a dict that groups indices of segments of the same observation 
    # i.e. where each observation id can be found in seg_ObIDs
    #i.e. ObID_SegIndices_dict == {'10258-01-01-00': [916, 949, 1046...467528, 467578], ....}
    ObID_SegIndices_dict = {key:[] for key in np.unique(seg_ObIDs)}
    for ID_index, ObID in enumerate(seg_ObIDs):
        ObID_SegIndices_dict.setdefault(ObID, []).append(ID_index)
    
    # make a dictionary of Gaussian component labels instead of segment indices  
    #i.e. ObID_GaussComps_dict_comp == {'10258-01-01-00': [401, 433, 382...101, 152], ....}
    ObID_GaussComps_dict_comp = {}
    for ObID, Indices in ObID_SegIndices_dict.items():
        ObID_GaussComps_dict_comp[ObID] = [data_GMMcomp_labels[ind] for ind in Indices]
        
    # make a data frame containing the counts of light curve segments in each of the Gaussian components, for each observation
    obs_component_counts_df_comp = pd.DataFrame(np.zeros((len(ObID_GaussComps_dict_comp),len(np.unique(data_GMMcomp_labels)))),
                                               index=np.unique(seg_ObIDs), columns=np.unique(data_GMMcomp_labels), dtype=int)
        
    # populate the data frame
    for ObID, GaussComps in ObID_GaussComps_dict_comp.items():
        for comp_id, comp_count in np.array(np.unique(GaussComps, return_counts=True)).T:
            obs_component_counts_df_comp.loc[ObID][comp_id] = comp_count
    
    
    obs_component_counts_df_comp = obs_component_counts_df_comp.iloc[:,:].div(np.sum(obs_component_counts_df_comp.iloc[:,:], axis=1), axis="rows") # normalise rows
    
    # add classification column
    obs_component_counts_df_comp["Class"] = "Unknown" 
    for k,v in ob_state.items():
        if v == "eta": v = "Unknown" ########## remove eta classifications, there are only two in the set of 1738 observations
        if str(k) in obs_component_counts_df_comp.index.values:
            obs_component_counts_df_comp.loc[str(k), "Class"] = v
            
#     obs_component_counts_df_comp.loc["10258-01-10-00", "Class"] = "mu"
    
    
    # create sets of shuffled (in as stratified manner) training and validaiton data 
    class_names, class_counts = np.unique(obs_component_counts_df_comp.loc[val_set_ids].loc[obs_component_counts_df_comp.loc[val_set_ids].iloc[:,-1] != "Unknown"].Class.values, return_counts=True)
    validation_sets = []
    training_sets = []

    for repetition in range(100):
        validation_set = []
        training_set = []
        for class_name, class_count in zip(class_names, class_counts):
            subset_ids = np.hstack((val_set_ids, train_set_ids))
            class_ids = obs_component_counts_df_comp.loc[subset_ids].loc[obs_component_counts_df_comp.loc[subset_ids].iloc[:,-1] == class_name].index.values
            np.random.seed(seed=repetition)
            validation_ids = np.random.choice(class_ids, size=class_count, replace=False)
            training_ids = np.array([x for x in class_ids if x not in validation_ids])

            validation_set.append(validation_ids)
            training_set.append(training_ids)

        validation_sets.append(np.concatenate(validation_set))
        training_sets.append(np.concatenate(training_set))

    #random forest hyperparameters
    max_depth_list = [None, 5, 10, 15, 25] # None 
    criterion_list= ["entropy", 'gini']

    reports = []
    
    for criterion in criterion_list:
        for max_depth in max_depth_list:
            for train_ids, val_ids in zip(training_sets, validation_sets):
                
                # training data
                train_data = obs_component_counts_df_comp.loc[train_ids] 
                # validation data
                val_data = obs_component_counts_df_comp.loc[val_ids]
                                
                RF_clf = RandomForestClassifier(random_state=0,
                                                criterion=criterion,
                                                class_weight="balanced",
                                                n_estimators=100,
                                                max_depth=max_depth, 
                                                min_samples_split= 2,
                                                min_samples_leaf = 1,
                                                n_jobs=35
                                               ).fit(train_data.iloc[:,:-1], train_data.iloc[:,-1])
                preds = RF_clf.predict(val_data.iloc[:,:-1])

                reports.append((f1_score(val_data.iloc[:,-1], preds, average="weighted", zero_division=0),
                                accuracy_score(val_data.iloc[:,-1], preds),
                               (criterion,max_depth)))
    return reports

def test_classification(data_labels, data, train_set_ids, val_set_ids, seg_ObIDs):
    """

    """
        
    # find GMM component labels for data
    data_GMMcomp_labels = data_labels
    
    # make a dict that groups indices of segments of the same observation 
    # i.e. where each observation id can be found in seg_ObIDs
    #i.e. ObID_SegIndices_dict == {'10258-01-01-00': [916, 949, 1046...467528, 467578], ....}
    ObID_SegIndices_dict = {key:[] for key in np.unique(seg_ObIDs)}
    for ID_index, ObID in enumerate(seg_ObIDs):
        ObID_SegIndices_dict.setdefault(ObID, []).append(ID_index)
    
    # make a dictionary of Gaussian component labels instead of segment indices  
    #i.e. ObID_GaussComps_dict_comp == {'10258-01-01-00': [401, 433, 382...101, 152], ....}
    ObID_GaussComps_dict_comp = {}
    for ObID, Indices in ObID_SegIndices_dict.items():
        ObID_GaussComps_dict_comp[ObID] = [data_GMMcomp_labels[ind] for ind in Indices]
        
    # make a data frame containing the counts of light curve segments in each of the Gaussian components, for each observation
    obs_component_counts_df_comp = pd.DataFrame(np.zeros((len(ObID_GaussComps_dict_comp),len(np.unique(data_GMMcomp_labels)))),
                                               index=np.unique(seg_ObIDs), columns=np.unique(data_GMMcomp_labels), dtype=int)
        
    # populate the data frame
    for ObID, GaussComps in ObID_GaussComps_dict_comp.items():
        for comp_id, comp_count in np.array(np.unique(GaussComps, return_counts=True)).T:
            obs_component_counts_df_comp.loc[ObID][comp_id] = comp_count
    
    
    obs_component_counts_df_comp = obs_component_counts_df_comp.iloc[:,:].div(np.sum(obs_component_counts_df_comp.iloc[:,:], axis=1), axis="rows") # normalise rows
    
    # add classification column
    obs_component_counts_df_comp["Class"] = "Unknown" 
    for k,v in ob_state.items():
        if v == "eta": v = "Unknown" ########## remove eta classifications, there are only two in the set of 1738 observations
        if str(k) in obs_component_counts_df_comp.index.values:
            obs_component_counts_df_comp.loc[str(k), "Class"] = v
            
    obs_component_counts_df_comp.loc["10258-01-10-00", "Class"] = "mu"
    
    
    # create sets of shuffled (in as stratified manner) training and validaiton data 
    class_names, class_counts = np.unique(obs_component_counts_df_comp.loc[val_set_ids].loc[obs_component_counts_df_comp.loc[val_set_ids].iloc[:,-1] != "Unknown"].Class.values, return_counts=True)
    validation_sets = []
    training_sets = []

    for repetition in range(100):
        validation_set = []
        training_set = []
        for class_name, class_count in zip(class_names, class_counts):
            subset_ids = np.hstack((val_set_ids, train_set_ids))
            class_ids = obs_component_counts_df_comp.loc[subset_ids].loc[obs_component_counts_df_comp.loc[subset_ids].iloc[:,-1] == class_name].index.values
            np.random.seed(seed=repetition)
            validation_ids = np.random.choice(class_ids, size=class_count, replace=False)
            training_ids = np.array([x for x in class_ids if x not in validation_ids])

            validation_set.append(validation_ids)
            training_set.append(training_ids)

        validation_sets.append(np.concatenate(validation_set))
        training_sets.append(np.concatenate(training_set))

    #random forest hyperparameters


    reports = []
    
    for train_ids, val_ids in zip(training_sets, validation_sets):

        # training data
        train_data = obs_component_counts_df_comp.loc[train_ids] 
        # validation data
        val_data = obs_component_counts_df_comp.loc[val_ids]

        RF_clf = RandomForestClassifier(random_state=0,
                                        criterion='gini',
                                        class_weight="balanced",
                                        n_estimators=1000,
                                        max_depth=5, 
                                        min_samples_split= 2,
                                        min_samples_leaf = 1,
                                        n_jobs=35
                                       ).fit(train_data.iloc[:,:-1], train_data.iloc[:,-1])
        preds = RF_clf.predict(val_data.iloc[:,:-1])

        reports.append((f1_score(val_data.iloc[:,-1], preds, average="weighted"),
                        accuracy_score(val_data.iloc[:,-1], preds)),
        (val_data.iloc[:,-1], preds))
        #(precision_recall_fscore_support(val_data.iloc[:,-1], preds, zero_division=0, average="weighted")[2],accuracy_score(val_data.iloc[:,-1], preds)))
    return reports
