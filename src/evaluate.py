from data_loader import load_data
from model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. データ読み込み
X, y = load_data()

# 2. モデル作成・学習
model = LinearRegression()
model.fit(X, y, lr=0.00001, epochs=5000)

# 3. 予測値
y_pred = model.predict(X)

# 4. 評価指標計算
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

# 5. 結果表示
print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")

# 6. 判定（簡易ルール）
if r2 > 0.9:
    print("判定: モデルの予測精度は非常に良いです。")
elif r2 > 0.7:
    print("判定: モデルはまずまずの精度です。改善余地あり。")
else:
    print("判定: モデルの精度は低いです。特徴量やモデルを改善しましょう。")
