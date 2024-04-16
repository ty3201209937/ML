import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ---------------1.准备数据--------------------------
data = np.array([[32, 31], [53, 68], [61, 62], [47, 71], [59, 87], [55, 78], [52, 79], [39, 59],
                 [48, 75], [52, 71], [45, 55], [54, 82], [44, 62], [58, 75], [56, 81], [48, 60],
                 [44, 82], [60, 97], [45, 48], [38, 56], [66, 83], [65, 118], [47, 57], [41, 51],
                 [51, 75], [59, 74], [57, 95], [63, 95], [46, 79], [50, 83]])

# 提取data中的两列数据，分别作为x，y
x = data[:, 0]
y = data[:, 1]


# ---------2. 定义损失函数------------------------
def compute_cost(w, b, data):
    total_cost = 0
    M = len(data)
    # 逐点计算平方损失误差，然后求平均数
    for i in range(M):
        x = data[i, 0]
        y = data[i, 1]
        total_cost += (y - w * x - b) ** 2
    return total_cost / M


# -----------3. 定义模型的超参数------------
alpha = 0.0001  # 学习率
initial_w = 0   # 初始权重
initial_b = 0   # 初始偏差
num_iter = 1000 # 迭代次数



# ----------4. 定义核心梯度下降算法函数------------------------
def grad_desc(data, initial_w, initial_b, alpha, num_iter):
    w = initial_w
    b = initial_b
    # 定义一个list保存所有的损失函数值，用来显示下降的过程
    cost_list = []
    for i in range(num_iter):
        cost_list.append(compute_cost(w, b, data))
        w, b = step_grad_desc(w, b, alpha, data)
    return [w, b, cost_list]


def step_grad_desc(current_w, current_b, alpha, data):
    sum_grad_w = 0
    sum_grad_b = 0
    M = len(data)
    # 对每个点，代入公式求和
    for i in range(M):
        x = data[i, 0]
        y = data[i, 1]
        sum_grad_w += (current_w * x + current_b - y) * x
        sum_grad_b += current_w * x + current_b - y
    # 用公式求当前梯度
    grad_w = 2 / M * sum_grad_w
    grad_b = 2 / M * sum_grad_b
    # 梯度下降，更新当前的w和b
    updated_w = current_w - alpha * grad_w
    updated_b = current_b - alpha * grad_b
    return updated_w, updated_b


#--------- 5. 测试：运行梯度下降算法计算最优的w和b-----------
w, b, cost_list = grad_desc(data, initial_w, initial_b, alpha, num_iter)
print("w is:", w)
print("b is:", b)
cost = compute_cost(w, b, data)
print("cost is:", cost)
plt.plot(cost_list)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost vs Iterations')
plt.show()

# ----------------6. 画出拟合曲线--------------------
plt.scatter(x, y)
#针对每一个X，计算出预测的y值
pred_y=w*x+ b
plt.plot(x,pred_y,c='r')
plt.show()