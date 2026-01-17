import numpy as np

class LinearRegression:
    def __init__(self):
        # 学習するパラメータ w(傾き)、b(切片) を初期化
        self.w = None
        self.b = None

    def predict(self, X):
        """
        入力 X に対して予測値を計算
        y_hat = w * X + b
        """
        return X * self.w + self.b

    def compute_loss(self, X, y):
        """
        平均二乗誤差(MSE)を計算
        """
        N = len(y)  # データ件数
        y_pred = self.predict(X)  # 現在のパラメータで予測
        loss = np.sum((y_pred - y)**2) / N  # MSE
        return loss

    def fit(self, X, y, lr=0.0005, epochs=1000):
        """
        0から勾配降下法で学習
        lr: 学習率
        epochs: 繰り返し回数
        """
        N = len(y)
        self.w = 0.0  # 傾き初期値
        self.b = 0.0  # 切片初期値

        for i in range(epochs):
            y_pred = self.predict(X)  # 予測
            # 損失の勾配を計算
            dw = (2/N) * np.sum((y_pred - y) * X)  # wの偏微分
            db = (2/N) * np.sum(y_pred - y)        # bの偏微分

            # パラメータ更新
            self.w -= lr * dw
            self.b -= lr * db

            # 100エポックごとに損失を表示して学習を確認
            if i % 100 == 0:
                print(f"Epoch {i}, Loss: {self.compute_loss(X,y):.4f}")

        # 学習完了後の最終パラメータ
        print(f"最終 w: {self.w:.4f}, b: {self.b:.4f}")
