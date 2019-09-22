import numpy as np

def top_k_accuracy_score(y_true, y_pred, k=5, normalize=True):
    """
     Top k Accuracy classification score.
    :param y_true: the true labels (np.ndarray)
    :param y_pred: the predicted labels (np.ndarray)
    :param k: calculates top k accuracy (int)
    :param normalize: whether to normalize by the number of observations
    :return: the top k accuracy
    """

    if len(y_true.shape) == 2:
        y_true = y_true[0] # should be one-dimensional

    num_obs, num_labels = y_pred.shape

    idx = num_labels - k - 1
    counter = 0
    argsorted = np.argsort(y_pred, axis=1)

    for i in range(num_obs):
        if y_true[i] in argsorted[i, idx+1:]:
            counter += 1
    if normalize:
        return counter / num_obs
    else:
        return counter