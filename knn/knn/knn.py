def KNN(X_train, X_test, y_train, y_test, k_val):
    predictions = []
    count = 0
    for test_data in X_test.to_numpy():
        count+=1
        distances = []
        for i in range(len(X_train)):
            distances.append(get_manhattan_distance((np.array(X_train.iloc[i])), test_data))
        
        distance_data = pd.DataFrame(data=distances, columns = ['distance'], index = y_train.index)
        k_neighbors_list = distance_data.sort_values(by=['distance'], axis=0)[:k_val]
        labels = y_train.loc[k_neighbors_list.index]
        most_common = Counter(labels).most_common()[0][0]
    
        predictions.append(most_common)
        
    return np.array(predictions) 
    
def KNN_Weighted(X_train, X_test, y_train, y_test, k_val):
    predictions = []
    count = 0
    for test_data in X_test.to_numpy():
        count+=1
        distances = []
        for i in range(len(X_train)):
            distances.append(get_manhattan_distance((np.array(X_train.iloc[i])), test_data))
            
        weights = [1/x for x in distances]
        distance_data = pd.DataFrame(index = y_train.index)
        distance_data['distance'] = distances
        distance_data['weight']= weights
        k_neighbors_list = distance_data.sort_values(by=['distance'], axis=0)[:k_val]
        labels = y_train.loc[k_neighbors_list.index]
        distances = k_neighbors_list.iloc[:,0]
        weights = k_neighbors_list.iloc[:,1]
        
        sumOfWeights={}
        
        for i in range(len(labels)):
            if labels.iloc[i] in sumOfWeights.keys():
                sumOfWeights[labels.iloc[i]] += weights.iloc[i]
            else:
                sumOfWeights[labels.iloc[i]] = weights.iloc[i]
                             
        max_key = max(sumOfWeights, key=sumOfWeights.get)
        predictions.append(max_key)
        
    return predictions
