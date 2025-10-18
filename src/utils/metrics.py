import numpy as np
from sklearn.metrics import (
    average_precision_score as sk_auc_pr,
    f1_score as sk_f1_score,
    precision_score as sk_precision_score,
    recall_score as sk_recall_score,
)


class Calculation:
    def __init__(self, threshold=0.5):
        self.threshold = float(threshold)

    @staticmethod
    def _validate_binary(y):
        arr = np.asarray(y).ravel()
        uniq = np.unique(arr)
        
        if not np.array_equal(np.sort(uniq), [0, 1]):
            raise ValueError("y must be binary and contain both classes {0,1}")

        return arr.astype(int)

    
    @staticmethod
    def _validate_proba(y_true, proba):
        y = Calculation._validate_binary(y_true)
        scores = np.asarray(proba, dtype=float).ravel()
    
        if scores.shape != y.shape:
            raise ValueError("y_true and proba must have the same length")

        if np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError("Predicted probabilities must lie within [0, 1]")

        if y.sum() == 0:
            raise ValueError("AUC PR is undefined when there are no positive samples")

        return y, scores


    def _prepare_labels(self, y_true, y_pred=None, proba=None, threshold=None):
        y = self._validate_binary(y_true)

        if proba is not None:
            scores = np.asarray(proba, dtype=float).ravel()

            if scores.shape != y.shape:
                raise ValueError("y_true and proba must have the same length")

            thr = self.threshold if threshold is None else float(threshold)
            y_hat = (scores >= thr).astype(int)

        else:
            if y_pred is None:
                raise ValueError("Provide either y_pred or proba")

            y_hat = self._validate_binary(y_pred)
            if y_hat.shape != y.shape:
                raise ValueError("y_true and y_pred must have the same length")

        return y, y_hat


    def _s21_recall(self, y_true, y_pred=None, proba=None, threshold=None):
        y, y_hat = self._prepare_labels(y_true, y_pred=y_pred, proba=proba, threshold=threshold)
        
        tp = np.sum((y == 1) & (y_hat == 1))
        fn = np.sum((y == 1) & (y_hat == 0))
        
        if tp + fn == 0:
            raise ValueError("Recall is undefined when there are no positive samples")

        return float(tp / (tp + fn))


    def _s21_precision(self, y_true, y_pred=None, proba=None, threshold=None):
        y, y_hat = self._prepare_labels(y_true, y_pred=y_pred, proba=proba, threshold=threshold)

        tp = np.sum((y == 1) & (y_hat == 1))
        fp = np.sum((y == 0) & (y_hat == 1))
        
        if tp + fp == 0:
            raise ValueError("Precision is undefined when there are no predicted positives")

        return float(tp / (tp + fp))


    def _s21_F1_score(self, y_true, y_pred=None, proba=None, threshold=None):
        prec = self._s21_precision(y_true, y_pred=y_pred, proba=proba, threshold=threshold)
        rec = self._s21_recall(y_true, y_pred=y_pred, proba=proba, threshold=threshold)

        if prec + rec == 0:
            print("Precision or recall = 0")

            return 0.0

        return float(2.0 * prec * rec / (prec + rec))

    def _s21_AUC_PR(self, y_true, proba):
        y, scores = self._validate_proba(y_true, proba)
        order = np.argsort(-scores, kind="mergesort")
        y_sorted = y[order]

        tp = np.cumsum(y_sorted).astype(float)
        fp = np.cumsum(1 - y_sorted).astype(float)

        precision = np.divide(tp, tp + fp, out=np.ones_like(tp), where=(tp + fp) != 0)
        recall = tp / tp[-1]

        precision = np.r_[1.0, precision]
        recall = np.r_[0.0, recall]

        auc_pr = np.sum((recall[1:] - recall[:-1]) * precision[1:])

        return float(auc_pr)

    def compare_to_original(self, y_true, proba, threshold=None):
        thr = self.threshold if threshold is None else float(threshold)

        s21_precision = self._s21_precision(y_true, proba=proba, threshold=thr)
        s21_recall = self._s21_recall(y_true, proba=proba, threshold=thr)
        s21_f1 = self._s21_F1_score(y_true, proba=proba, threshold=thr)
        s21_auc_pr = self._s21_AUC_PR(y_true, proba)

        y_pred = (np.asarray(proba) >= thr).astype(int)

        sk_metrics = {
            "precision": sk_precision_score(y_true, y_pred),
            "recall": sk_recall_score(y_true, y_pred),
            "f1": sk_f1_score(y_true, y_pred),
            "auc_pr": sk_auc_pr(y_true, proba),
        }

        return {
            "s21": {
                "precision": s21_precision,
                "recall": s21_recall,
                "f1": s21_f1,
                "auc_pr": s21_auc_pr,
            },
            "sklearn": sk_metrics,
        }