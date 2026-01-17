import matplotlib.pyplot as plt
from data_loader import load_data
from model import LinearRegression

# 1. データ読み込み
X, y = load_data()

# 2. モデル作成・学習
model = LinearRegression()
model.fit(X, y, lr=0.00001, epochs=5000)  # 安定学習用に小さい学習率

# 3. 予測値
y_pred = model.predict(X)

# 4. 可視化
plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', label='実データ')       # 実際の広告費 vs 売上
plt.plot(X, y_pred, color='red', label='予測直線')      # 学習した直線
plt.xlabel("広告費 (万円)")
plt.ylabel("売上 (万円)")
plt.title("広告費と売上の線形回帰フィット")
plt.legend()
plt.grid(True)
plt.show()
