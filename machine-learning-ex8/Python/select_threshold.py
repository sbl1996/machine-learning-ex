import numpy as np

def select_threshold(yval, pval):
  best_epsilon = 0
  best_F1 = 0
  F1 = 0

  for epsilon in np.linspace(pval.min(), pval.max(), 1000):
    pred = pval < epsilon
    tp = ((pred == 1) & (yval == 1)).sum() 
    fp = ((pred == 1) & (yval == 0)).sum() 
    fn = ((pred == 0) & (yval == 1)).sum() 
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    F1 = 2 * prec * rec / (prec + rec)

    if F1 > best_F1:
      best_F1 = F1
      best_epsilon = epsilon

  return best_epsilon, best_F1