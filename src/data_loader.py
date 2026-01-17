import numpy as np
import pandas as pd

def load_data(csv_path="data/sales_data.csv"):
    """
    CSVを読み込んでX（入力）とy（出力）を返す
    """
    # CSVを読み込む
    df = pd.read_csv(csv_path)

    # X: 入力（広告費）
    # 実務では1列だけの単回帰でも、numpy計算のために2次元配列に整形
    X = df["ad_budget"].values.reshape(-1, 1)

    # y: 出力（売上）
    y = df["sales"].values.reshape(-1, 1)

    return X, y

if __name__ == "__main__":
    X, y = load_data()
    print("Xの形:", X.shape)
    print("yの形:", y.shape)
    print("最初の5件:\n", np.hstack((X[:5], y[:5])))
    print("データを正常に読み込みました。")
