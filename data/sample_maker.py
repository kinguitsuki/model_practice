import numpy as np
import pandas as pd
import os

# ディレクトリ作成（存在しなければ）
os.makedirs("data", exist_ok=True)

# 乱数の再現性
np.random.seed(42)

# データ件数
N = 100

# 入力：広告費（0〜100万円）
ad_budget = np.random.uniform(0, 100, N)

# ノイズ（売上に自然な揺らぎ）
noise = np.random.normal(0, 10, N)

# 出力：売上 = 5 + 2.5*広告費 + ノイズ
sales = 5 + 2.5 * ad_budget + noise

# データフレーム作成
df = pd.DataFrame({
    "ad_budget": ad_budget,
    "sales": sales
})

# CSV保存
df.to_csv("data/sales_data.csv", index=False)

print("CSVファイルを作成しました: data/sales_data.csv")
