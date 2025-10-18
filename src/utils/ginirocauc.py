import numpy as np

class S21RocAuc:
    def __init__(self):
        pass


    def _validate(self, y_true, proba):
        y_true = np.asarray(y_true)
        proba = np.asarray(proba)

        if y_true.shape != proba.shape:
            raise ValueError("y_true and proba must have the same shape")

        uniq = np.unique(y_true)
        if not np.array_equal(np.sort(uniq), [0, 1]):
            raise ValueError("y_true must be binary {0,1}")

        if np.any((proba < 0) | (proba > 1)):
            raise ValueError("proba must be between 0 and 1")

        n_pos = int((y_true == 1).sum())
        n_neg = int((y_true == 0).sum())
        if n_pos == 0 or n_neg == 0:
            raise ValueError("y_true must contain at least one positive and one negative example (else -> division by zero)")

        return y_true, proba, n_pos, n_neg


    def s21_roc_auc_score(self, y_true, proba):
        y, s, n_pos, n_neg = self._validate(y_true, proba)

        uniq, inv = np.unique(s, return_inverse=True) 
        pos_per_score = np.bincount(inv, weights=(y == 1))
        neg_per_score = np.bincount(inv, weights=(y == 0))

        order = np.argsort(-uniq)
        tp = np.cumsum(pos_per_score[order])
        fp = np.cumsum(neg_per_score[order])

        tpr = np.r_[0.0, tp] / n_pos
        fpr = np.r_[0.0, fp] / n_neg

        auc = np.trapz(tpr, fpr)
        
        return float(auc)


    def s21_gini_score(self, y_true, proba):
        auc = self.s21_roc_auc_score(y_true, proba)
        gini = 2.0 * auc - 1.0

        return gini