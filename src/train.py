from data_loader import load_data
from model import LinearRegression

# 1. データ読み込み
X, y = load_data()

# 2. モデル作成
model = LinearRegression()

# 3. 学習
model.fit(X, y, lr=0.0001, epochs=5000)

# 4. 学習結果で予測
y_pred = model.predict(X)

# 5. 学習結果確認（最初の5件）
for xi, yi, yhat in zip(X[:5], y[:5], y_pred[:5]):
    print(f"入力: {xi[0]:.2f}, 正解: {yi[0]:.2f}, 予測: {yhat[0]:.2f}")
