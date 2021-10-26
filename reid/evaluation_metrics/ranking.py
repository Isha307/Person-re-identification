from __future__ import absolute_import
from collections import defaultdict

import numpy as np
from sklearn.metrics._base import _average_binary_score
from sklearn.metrics import precision_recall_curve, auc
from ..utils import to_numpy

def map_cmc(distmat, query_ids=None, gallery_ids=None,
            query_cams=None, gallery_cams=None, topk=100):
    distmat = to_numpy(distmat)
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute mAP and CMC for each query
    ret = np.zeros(topk)
    aps = []
    num_valid_queries = 0
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        if not np.any(matches[i, valid]): continue

        # Compute mAP
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        if not np.any(y_true): continue
        aps.append(average_precision_score(y_true, y_score))

        # Compute CMC
        index = np.nonzero(matches[i, valid])[0]
        for j, k in enumerate(index):
            if k >= topk: break
            ret[k] += 1
            break
        num_valid_queries += 1
    if num_valid_queries == 0:
        raise RuntimeError("No valid query")
    return np.mean(aps), ret.cumsum() / num_valid_queries

