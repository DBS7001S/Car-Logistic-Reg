import numpy as np
import pandas as pd                      # đọc CSV, mã hóa dữ liệu
import matplotlib.pyplot as plt          # vẽ đồ thị

from sklearn.model_selection import train_test_split
# chia data set thành train/test set
from sklearn.linear_model import LogisticRegression
# dùng để đối chiếu với bản tự code
from sklearn.metrics import (accuracy_score, roc_auc_score, log_loss,
                             confusion_matrix, precision_score,
                             recall_score, f1_score)
# các hàm đánh giá

path = 'car_combined.csv'
np.random.seed(42)                       # để kết quả tái lập được

# ---------- đọc và mã hóa dữ liệu từ CSV ----------
#   class = unacc            -> 0 (xe không chấp nhận được)
#   class = acc/good/vgood   -> 1 (xe chấp nhận được)
# Mọi thuộc tính đều là biến phân loại -> mã hóa one-hot.
# Bỏ 1 mức làm mức tham chiếu (cột bias đã đóng vai trò đó) để tránh đa cộng tuyến.
CATEGORIES = {
    'buying':   ['low', 'med', 'high', 'vhigh'],     # mức tham chiếu: low
    'maint':    ['low', 'med', 'high', 'vhigh'],     # mức tham chiếu: low
    'doors':    ['2', '3', '4', '5more'],            # mức tham chiếu: 2
    'persons':  ['2', '4', 'more'],                  # mức tham chiếu: 2
    'lug_boot': ['small', 'med', 'big'],             # mức tham chiếu: small
    'safety':   ['low', 'med', 'high'],              # mức tham chiếu: low
}


def load_data(path):
    data = pd.read_csv(path)
    for c in CATEGORIES:
        data[c] = data[c].astype(str).str.strip()

    # mã hóa label: unacc -> 0, còn lại -> 1
    data['Acceptable'] = (data['class'] != 'unacc').astype(int)

    # mã hóa one-hot cho từng thuộc tính (bỏ mức đầu tiên)
    feature_names = []
    for col, levels in CATEGORIES.items():
        for level in levels[1:]:
            name = col + '_' + level
            data[name] = (data[col] == level).astype(int)
            feature_names.append(name)

    x = data[feature_names].values
    y = data['Acceptable'].values
    return x, y, feature_names


x, y, feature_names = load_data(path)
print('Số mẫu:', x.shape[0], '| số đặc trưng:', x.shape[1],
      '| tỷ lệ lớp 1 (acceptable): %.3f' % y.mean())

# ---------- chia data thành train/test set ----------
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y)
x_train = x_train.astype(float)
x_test = x_test.astype(float)

# ---------- chuẩn hóa dữ liệu ----------
# Không cần: toàn bộ đặc trưng là biến nhị phân 0/1 (one-hot) nên đã cùng thang đo.


# ---------- Bias trick ----------
def bias_trick(x):
    n = x.shape[0]
    column1 = np.ones((n, 1))            # cột toàn số 1
    xNew = np.hstack((column1, x))       # ghép theo chiều ngang
    return xNew.T                        # ma trận đặc trưng (d+1, N)


X_train = bias_trick(x_train)
X_test = bias_trick(x_test)


def calS(w, X):
    return np.dot(w.T, X)                # s = W^T X, dạng (1, N)


def sigmoid(s):
    # dạng ổn định số học, tránh tràn exp khi s rất lớn/nhỏ
    s = np.clip(s, -500, 500)
    return 1 / (1 + np.exp(-s))


# ---------- Loss function (cross-entropy) ----------
def calLoss(w, X, y):
    z = sigmoid(calS(w, X))              # xác suất của mỗi xe, dạng (1, N)
    N = X.shape[1]
    eps = 1e-12                          # tránh log(0)

    tong = 0
    for i in range(N):
        zi = min(max(z[0, i], eps), 1 - eps)
        yi = y[i]
        tong = tong + yi * np.log(zi) + (1 - yi) * np.log(1 - zi)

    return -tong


w_zero = np.zeros((X_train.shape[0], 1))
print('Loss khi w = 0 (kỳ vọng N*ln2 = %.2f): %.2f'
      % (X_train.shape[1] * np.log(2), calLoss(w_zero, X_train, y_train)))


# ---------- Logistic Regression bằng SGD ----------
def logistic_sigmoid_regression(X, y, w_init, eta, tol=1e-4, max_count=10000):
    w = [w_init]
    N = X.shape[1]
    d = X.shape[0]
    count = 0
    check_w_after = 20
    while count < max_count:
        # xáo trộn thứ tự các mẫu
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = y[i]
            zi = sigmoid(np.dot(w[-1].T, xi))
            w_new = w[-1] + eta * (yi - zi) * xi
            count += 1
            # điều kiện dừng
            if count % check_w_after == 0:
                if np.linalg.norm(w_new - w[-check_w_after]) < tol:
                    w.append(w_new)
                    return w
            w.append(w_new)
            if count >= max_count:
                break
    return w


eta = 0.01
d = X_train.shape[0]
w_init = np.random.randn(d, 1)

w = logistic_sigmoid_regression(X_train, y_train, w_init, eta, max_count=300000)
w_final = w[-1]

N_train = X_train.shape[1]
print('\nSố lần cập nhật w:', len(w) - 1)
print('Loss trung bình mỗi xe (train):',
      calLoss(w_final, X_train, y_train) / N_train)

# ---------- 1. Đánh giá mô hình trên tập Test ----------
s_test = calS(w_final, X_test)           # s = W^T X cho tập test
z_test = sigmoid(s_test)                 # xác suất dự đoán (0 đến 1)

# Ngưỡng cứng 0.5: >= 0.5 là chấp nhận (1), < 0.5 là không chấp nhận (0)
y_pred = (z_test >= 0.5).astype(int).flatten()

print('\n--- Kết quả bản tự code (SGD) trên tập test ---')
print('Accuracy :', round(accuracy_score(y_test, y_pred), 4))
print('Precision:', round(precision_score(y_test, y_pred), 4))
print('Recall   :', round(recall_score(y_test, y_pred), 4))
print('F1-score :', round(f1_score(y_test, y_pred), 4))
print('ROC-AUC  :', round(roc_auc_score(y_test, z_test.flatten()), 4))
print('Ma trận nhầm lẫn [[TN, FP], [FN, TP]]:\n', confusion_matrix(y_test, y_pred))

# ---------- Đối chứng với sklearn ----------
clf = LogisticRegression(C=1e4, max_iter=5000)   # C lớn ~ không regularization
clf.fit(x_train, y_train)
p_sk = clf.predict_proba(x_test)[:, 1]
print('\n--- Đối chứng sklearn LogisticRegression ---')
print('Accuracy :', round(accuracy_score(y_test, clf.predict(x_test)), 4))
print('ROC-AUC  :', round(roc_auc_score(y_test, p_sk), 4))
print('Log-loss (sklearn):', round(log_loss(y_test, p_sk), 4),
      '| Log-loss (tự code):', round(log_loss(y_test, z_test.flatten()), 4))

# ---------- Trọng số học được ----------
print('\nTrọng số w (tự code) - hệ số dương: tăng khả năng được chấp nhận:')
names = ['bias'] + feature_names
for n, v in sorted(zip(names, w_final.flatten()), key=lambda t: -abs(t[1])):
    print('  %-16s %+.3f' % (n, v))

# ---------- 2. Vẽ đồ thị Sigmoid ----------
s_test_1d = s_test.flatten()
s_range = np.linspace(np.min(s_test_1d) - 1, np.max(s_test_1d) + 1, 500)
sigmoid_range = sigmoid(s_range)

plt.figure(figsize=(10, 6))

# đường cong sigmoid
plt.plot(s_range, sigmoid_range, color='red', linewidth=2.5,
         label=r'Đường cong Sigmoid: $\sigma(s)$')

# các điểm dữ liệu thực tế của tập test
plt.scatter(s_test_1d[y_test == 1], y_test[y_test == 1],
            color='blue', alpha=0.3, label='Thực tế: Chấp nhận (1)', marker='o')
plt.scatter(s_test_1d[y_test == 0], y_test[y_test == 0],
            color='orange', alpha=0.3, label='Thực tế: Không chấp nhận (0)', marker='x')

# vạch ngưỡng 0.5 và s = 0
plt.axhline(0.5, color='black', linestyle='--', alpha=0.6, label='Ngưỡng phân lớp (0.5)')
plt.axvline(0, color='black', linestyle='--', alpha=0.6)

plt.xlabel(r'Tổ hợp tuyến tính $s = W^T X$')
plt.ylabel('Xác suất dự đoán xe được chấp nhận')
plt.title('Đồ thị Logistic Regression (Sigmoid) trên tập dữ liệu Car Evaluation')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()