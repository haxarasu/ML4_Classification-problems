import numpy as np 
import pandas as pd
from typing import List, Tuple, Optional, Dict

class FeatureGenerator:
    def __init__(
        self,
        ratios: Optional[List[Tuple[str, str]]] = None,
        groupbys: Optional[List[Tuple[str, str]]] = None,
        eps: float = 1e-9 # defence from division by zero
    ):
        self.ratios = ratios
        self.groupbys = groupbys
        self.eps = eps

        self._maps: Dict[Tuple[str, str], pd.Series] = {}
        self._globals: Dict[Tuple[str, str], float] = {}
        self._ratio_fill = {} # change nan to median

        self._fitted: bool = False


    def _safe_div(self, num: pd.Series, den: pd.Series) -> pd.Series:
        num = pd.to_numeric(num, errors="coerce")
        den = pd.to_numeric(den, errors="coerce").replace(0, np.nan)

        return num / (den + self.eps)


    def fit(self, X_train: pd.DataFrame, y=None):
        df = X_train.copy()

        for a, b in (self.ratios or []):
            if a in df.columns and b in df.columns:
                col = f"R_{a}_over_{b}"
                df[col] = self._safe_div(df[a], df[b])
                self._ratio_fill[col] = float(df[col].median(skipna=True))

        for cat_col, num_col in (self.groupbys or []):
            cat = df[cat_col].astype("object")
            val = pd.to_numeric(df[num_col], errors="coerce")
            mp = val.groupby(cat).mean()
            self._maps[(cat_col, num_col)] = mp
            self._globals[(cat_col, num_col)] = float(val.mean())

        self._fitted = True

        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() before calling transform()")

        df = X.copy()

        for a, b in (self.ratios or []):
            if a in df.columns and b in df.columns:
                col = f"R_{a}_over_{b}"
                df[col] = self._safe_div(df[a], df[b])
                if col in self._ratio_fill:
                    df[col] = df[col].fillna(self._ratio_fill[col])
            else:
                raise ValueError(f"There is no column {a} or {b} in the dataframe")

        for (cat_col, num_col), mp in self._maps.items():
            if cat_col in df.columns:
                glob = self._globals[(cat_col, num_col)]
                vals = df[cat_col].astype("object").map(mp)
                df[f"GB_{cat_col}_mean_{num_col}"] = vals.fillna(glob)

        return df


    def fit_transform(self, X_train: pd.DataFrame, y=None) -> pd.DataFrame:
        self.fit(X_train, y=y)

        return self.transform(X_train)


    def transform_many(
        self,
        X_train: pd.DataFrame,
        X_val: Optional[pd.DataFrame] = None,
        X_test: Optional[pd.DataFrame] = None,
        y=None
    ):
        Xtr_new = self.fit_transform(X_train, y=y)
        Xva_new = self.transform(X_val) if X_val is not None else None
        Xte_new = self.transform(X_test) if X_test is not None else None
        
        return Xtr_new, Xva_new, Xte_new