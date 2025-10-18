from itertools import product
from typing import List, Tuple

import pandas as pd
from category_encoders import CountEncoder
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from utils.metrics import S21RocAuc

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

class PipelineManager:
    def __init__(
        self, clf, 
        model_name,
        Xs: List,
        ys: List
    ):
        self.clf = clf

        self.model_name = model_name

        self.X_train, self.X_val = Xs[0], Xs[1]
        self.y_train, self.y_val = ys[0], ys[1]


    def _preprocess_builder(self):
        X_train = self.X_train

        cat_cols = X_train.select_dtypes(include=["object","category"]).columns
        num_cols = X_train.select_dtypes(include=["number"]).columns

        num_block = Pipeline([
            ("impute", SimpleImputer(strategy="median")), # for NaN
            ("sc",     StandardScaler())
        ])

        cat_block = Pipeline([
            ("cnt", CountEncoder(cols=cat_cols, handle_unknown=0, handle_missing=0, normalize=True))
        ])

        self.preprocess = ColumnTransformer(
            transformers=[
                ("num", num_block, num_cols),
                ("cat", cat_block, cat_cols),
            ]
        )

        return self


    def _pipeline_builder(self):
        pipe = Pipeline([
            ("preprocess", self.preprocess),
            ("clf", self.clf)
        ])

        self.pipe = pipe.fit(self.X_train, self.y_train)

        return self


    def _s21_evaluate_model(self):
        X = self.X_val
        y = self.y_val
        name = self.model_name
        threshold = 0.5

        proba = self.pipe.predict_proba(X)[:, 1]
        pred = (proba >= threshold).astype(int)

        auc  = roc_auc_score(y, proba)
        gini = 2 * auc - 1
        ap   = average_precision_score(y, proba)
        rec  = recall_score(y, pred)
        prec = precision_score(y, pred)
        f1   = f1_score(y, pred)

        print(f"=== {name} ===")
        print(f"ROC AUC : {auc:.4f}   | Gini: {gini:.4f}")
        print(f"PR AUC  : {ap:.4f}")
        print(f"F1      : {f1:.4f}")
        print(f"Precision: {prec:.4f} | Recall: {rec:.4f}")

        self.proba = proba

        return self

    
    def estimate(self):
        self._preprocess_builder()._pipeline_builder()._s21_evaluate_model()

        y_val = self.y_val
        proba = self.proba

        est = S21RocAuc()
        s21_rocauc = est.s21_roc_auc_score(y_val, proba)
        s21_gini = est.s21_gini_score(y_val, proba)

        print("S21 ROC AUC: {:.4f}".format(s21_rocauc))
        print("S21 GINI: {:.4f}".format(s21_gini))


    def l1_search(self):
        self._preprocess_builder()

        pipe = Pipeline([
            ("preprocess", self.preprocess),
            ("clf", LogisticRegression(max_iter=5000, solver='saga', penalty="l1")),
        ])

        pipe.fit(self.X_train, self.y_train)

        n_nonzero = int((pipe.named_steps["clf"].coef_.ravel() != 0).sum())

        self.pipe = pipe

        print(f"nonzero weights = {n_nonzero}")
        print()

        self._s21_evaluate_model()

    
    def param_tune(self, pipe, data: Tuple[List, List]):
        X_train, X_val = data[0]
        y_train, y_val = data[1]

        C_common = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
        max_iter_saga = [3000, 5000]
        max_iter_lbfgs = [1000, 3000, 5000]
        class_weights = [None, 'balanced']
        saga_solver = ['saga']


        param_grid_l1 = {
            'clf__penalty': ['l1'],
            'clf__solver': saga_solver,
            'clf__C': C_common,
            'clf__max_iter': max_iter_saga,
            'clf__class_weight': class_weights
        }

        param_grid_l2 = {
            'clf__penalty': ['l2'],
            'clf__solver': ['lbfgs'],
            'clf__C': C_common,
            'clf__max_iter': max_iter_lbfgs,
            'clf__class_weight': class_weights
        }

        param_grid_enet = {
            'clf__penalty': ['elasticnet'],
            'clf__solver': saga_solver,
            'clf__C': C_common,
            'clf__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            'clf__max_iter': max_iter_saga,
            'clf__class_weight': class_weights
        }

        grids = {
            "l1": param_grid_l1,
            "l2": param_grid_l2,
            "elasticnet": param_grid_enet
        }

        best_gini = -1.0
        best_cfg = None
        best_pipe = None
        search_log = []

        for family, grid in grids.items():
            keys = list(grid.keys())
            combos = list(product(*grid.values())) # сartesian product of params' sets

            for values in tqdm(
                combos,
                desc=f"Tuning {family}",
                unit="cfg",
                leave=False,
            ):
                params = dict(zip(keys, values))
                candidate = clone(pipe)
                candidate.set_params(**params)
                candidate.fit(X_train, y_train)

                proba = candidate.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, proba)
                gini = 2.0 * auc - 1.0

                record = {
                    **params,
                    "gini": gini,
                }
                search_log.append(record)

                if gini > best_gini:
                    best_gini = gini
                    best_cfg = params
                    best_pipe = candidate

        results = pd.DataFrame(search_log).sort_values("gini", ascending=False)
        print(results.head(10))

        print(f"\nGini={best_gini:.4f}")
        print(f"Params: {best_cfg}")

        self.pipe = best_pipe
        self.model_name = 'Logistic Regression (param tuned)'

        self._s21_evaluate_model()

        return {
            "best pipe": best_pipe,
            "params": best_cfg,
            "gini": best_gini,
            "results": results,
        }

    @staticmethod
    def gini_eval(model, X, y):
        metric = S21RocAuc()
        proba = model.predict_proba(X)[:, 1]
        auc = metric.s21_roc_auc_score(y, proba)
        gini = metric.s21_gini_score(y, proba)

        return gini