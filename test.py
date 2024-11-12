import numpy as np
import matplotlib.pyplot as plt

# Hàm f(x, y) - Bài toán bậc 2
def f(xy):
    A = np.array([[10, 5], [2, 10]])  # Ma trận A
    b = np.array([4, 5])  # Vector b
    xy = xy.reshape(-1, 1)  # Đảm bảo xy là cột vector

    term1 = 2 * np.dot(xy.T, np.dot(A, xy))  # 2 * [x, y]^T * A * [x, y]
    term2 = -4 * np.dot(b.T, xy)  # -4 * b^T * [x, y]

    return term1 - term2

# Gradient của f(x, y)
def grad(xy):
    A = np.array([[10, 5], [2, 10]])  # Ma trận A
    b = np.array([4, 5])  # Vector b
    return 4 * np.dot(A, xy) - 4 * b.reshape(-1, 1)  # Gradient của f(x, y)

# Hàm cost: f(x, y)
def cost(xy):
    return f(xy)

# Kiểm tra gradient bằng phương pháp số
def numerical_grad(xy, cost):
    eps = 1e-4
    g = np.zeros_like(xy)
    for i in range(len(xy)):
        xy_p = xy.copy()
        xy_n = xy.copy()
        xy_p[i] += eps
        xy_n[i] -= eps
        g[i] = (cost(xy_p) - cost(xy_n)) / (2 * eps)
    return g

# Kiểm tra độ chính xác của gradient
def check_grad(xy, cost, grad):
    grad1 = grad(xy)
    grad2 = numerical_grad(xy, cost)
    return np.linalg.norm(grad1 - grad2) < 1e-6

# Gradient Descent
def myGD(w_init, grad, eta, max_iter=100, tol=1e-3):
    w = [w_init]
    for it in range(max_iter):
        w_new = w[-1] - eta * grad(w[-1])  # Cập nhật w theo gradient
        if np.linalg.norm(grad(w_new)) / len(w_new) < tol:  # Dừng nếu gradient nhỏ
            break
        w.append(w_new)
    return w, it

# Giá trị khởi tạo cho [x, y]
w_init = np.array([[2], [1]])

# Kiểm tra gradient
print("Checking gradient:", check_grad(np.random.rand(2, 1), cost, grad))

# Chạy Gradient Descent
(w1, it1) = myGD(w_init, grad, eta=0.1)
print('Solution found by GD: w = ', w1[-1].T, ', after %d iterations.' % (it1 + 1))

# Vẽ đồ thị f(x, y)
x_vals = np.linspace(-2, 2, 100)
y_vals = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x_vals, y_vals)

Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = f(np.array([X[i, j], Y[i, j]]))

# Vẽ contour của f(x, y)
plt.contour(X, Y, Z, 20)
plt.plot(w1[-1][0], w1[-1][1], 'ro', label="Solution")
plt.title("Gradient Descent for Quadratic Function")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
