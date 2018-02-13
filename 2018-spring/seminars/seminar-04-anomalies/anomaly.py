import numpy as np 


def get_anomaly_multivariate(X, y, normal_label, anomalies_share=0.01, copy=True):
    if copy:
        X_data, y_data = X.copy(), y.copy()
    else:
        X_data, y_data = X, y
    
    random_idx = np.random.permutation(np.arange(y_data.shape[-1]))
    X_data, y_data = X_data[random_idx], y_data[random_idx]

    X_normal = X_data[y_data == normal_label]
    y_normal = y_data[y_data == normal_label]
    X_abnormal = X_data[y_data != normal_label]
    y_abnormal = y_data[y_data != normal_label]
    last_abnormal_idx = np.ceil(X_normal.shape[0] * anomalies_share).astype(int)
    
    X_contaminated = np.vstack((X_normal, X_abnormal[:last_abnormal_idx]))
    y_contaminated = np.concatenate((y_normal, y_abnormal[:last_abnormal_idx]))
    
    # shuffle again to output in random order
    random_idx = np.random.permutation(np.arange(y_contaminated.shape[-1]))
    return X_contaminated[random_idx], y_contaminated[random_idx]