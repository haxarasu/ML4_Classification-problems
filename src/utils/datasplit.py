import pandas as pd

class SplitByThirds:
    def __init__(self, df: pd.DataFrame, date_column: str = "PurchDate"):
        self.date_column = date_column
        self.df = self._prepare(df, date_column)


    @staticmethod
    def _prepare(df: pd.DataFrame, column: str) -> pd.DataFrame:
        df = df.copy()
        df[column] = pd.to_datetime(df[column], errors="coerce")
        df = df.sort_values(column)

        return df


    def _positions(self):
        rows = self.df.shape[0]
        rows -= rows % 3  # remainder will be discarded
        third = int(rows / 3)
        
        train_pos = (0, third)
        val_pos = (third, 2 * third)
        test_pos = (2 * third, rows)

        return train_pos, val_pos, test_pos


    def split(self):
        train_pos, val_pos, test_pos = self._positions()

        df_train = self.df.iloc[train_pos[0]:train_pos[1]]
        df_val   = self.df.iloc[val_pos[0]:val_pos[1]]
        df_test  = self.df.iloc[test_pos[0]:test_pos[1]]

        return df_train, df_val, df_test