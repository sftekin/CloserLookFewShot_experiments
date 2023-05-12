import numpy as np


def find_majority(row):
    count = np.bincount(row.astype(int))
    threshold = len(row) // 2
    major_satisfied = count.max() > threshold
    if major_satisfied:
        ret_val = float(np.argmax(count))
    else:
        ret_val = np.nan
    return ret_val


def find_plurality(row):
    count = np.bincount(row.astype(int))
    return np.argmax(count)


def voting(pred_arr, method, n_query, n_way):
    num_samples, num_classes = pred_arr.shape
    if method == "majority":
        voting_method = find_majority
    elif method == "plurality":
        voting_method = find_plurality
    else:
        raise KeyError(f"input method: '{method}' is not found")
    ens_pred_flat = np.apply_along_axis(voting_method, axis=1, arr=pred_arr)
    ens_pred = ens_pred_flat.reshape(-1, n_query * n_way)
    return ens_pred


# def majority_plus(pred_arr):
#     def find_majority_plus(preds, other_preds):
#         count = np.bincount(preds.astype(int))
#         threshold = len(preds) // 2
#         major_satisfied = count.max() > threshold
#         if major_satisfied:
#             ret_val = float(np.argmax(count))
#         else:
#             if (len(preds) % 2 == 0) & (len(preds) > 2):
#                 agree_idx = count == count.max()
#                 find_majority_plus()
#             else:
#                 ret_val = np.nan
#         return ret_val


ensemble_methods = {
    "voting": voting
}
