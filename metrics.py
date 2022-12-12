
import numpy as np


class Recall(object):
  """Recall metric."""

  def __init__(self, k=1):
    self.topk = k

  def __call__(self, topk_items, true_items):
    topk_items = topk_items[:self.topk]
    hit_items = set(true_items) & set(topk_items)
    return len(hit_items) / (len(true_items) + 1e-12)


class NormalizedRecall(object):
  """Recall metric normalized to max 1."""

  def __init__(self, k=1):
    self.topk = k

  def __call__(self, topk_items, true_items):
    topk_items = topk_items[:self.topk]
    hit_items = set(true_items) & set(topk_items)
    return len(hit_items) / min(self.topk, len(true_items) + 1e-12)


class Precision(object):
  """Precision metric."""

  def __init__(self, k=1):
    self.topk = k

  def __call__(self, topk_items, true_items):
    topk_items = topk_items[:self.topk]
    hit_items = set(true_items) & set(topk_items)
    return len(hit_items) / (self.topk + 1e-12)


class F1(object):
  def __init__(self, k=1):
    self.precision_k = Precision(k)
    self.recall_k = Recall(k)

  def __call__(self, topk_items, true_items):
    p = self.precision_k(topk_items, true_items)
    r = self.recall_k(topk_items, true_items)
    return 2 * p * r / (p + r + 1e-12)


class DCG(object):
  """ Calculate discounted cumulative gain
  """

  def __init__(self, k=1):
    self.topk = k

  def __call__(self, topk_items, true_items):
    topk_items = topk_items[:self.topk]
    true_items = set(true_items)
    return sum(1 / np.log(2 + i) for i, item in enumerate(topk_items) if item in true_items)


class NDCG(object):
  """Normalized discounted cumulative gain metric."""

  def __init__(self, k=1):
    self.topk = k

  def __call__(self, topk_items, true_items):
    topk_items = topk_items[:self.topk]
    dcg_fn = DCG(k=self.topk)
    idcg = dcg_fn(true_items[:self.topk], true_items)
    dcg = dcg_fn(topk_items, true_items)
    return dcg / (idcg + 1e-12)


class MRR(object):
  """MRR metric"""

  def __init__(self, k=1):
    self.topk = k

  def __call__(self, topk_items, true_items):
    topk_items = topk_items[:self.topk]
    true_items = set(true_items)
    return sum(1 / (i + 1.0) for i, item in enumerate(topk_items) if item in true_items)


class HitRate(object):
  def __init__(self, k=1):
    self.topk = k

  def __call__(self, topk_items, true_items):
    topk_items = topk_items[:self.topk]
    hit_items = set(true_items) & set(topk_items)
    return 1 if len(hit_items) > 0 else 0


class MAP(object):
  """
  Calculate mean average precision.
  """

  def __init__(self, k=1):
    self.topk = k

  def __call__(self, topk_items, true_items):
    topk_items = topk_items[:self.topk]
    true_items = set(true_items)
    pos = 0
    precision = 0
    for i, item in enumerate(topk_items):
      if item in true_items:
        pos += 1
        precision += pos / (i + 1.0)
    return precision / (pos + 1e-12)
