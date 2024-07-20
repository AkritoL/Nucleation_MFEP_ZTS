import matplotlib.pyplot as plt
from contextlib import redirect_stdout
import numpy as np

# 数据
x_axis = np.log(1 - np.array([0.92, 0.925, 0.93, 0.935, 0.94, 0.945]))
eb_polar = np.log(np.array([4.42, 4.70, 5.15, 5.59, 6.06, 6.68])*1e-4)
eb_cartesian = np.log(np.array([4.49, 4.87, 5.29, 5.80, 6.43, 7.16])*1e-4)
#eb_cnt = np.log(np.array([4.94, 5.23, 5.56, 5.94, 6.38, 6.91])*1e-4)
eb_cnt = np.log(np.array([0.00048474,0.00051371,0.00054689,0.00058525,0.0006301,0.00068322]))

# 创建图形和轴
fig, ax = plt.subplots(figsize=(8, 6))

# 为每个数据集绘制散点图
ax.scatter(x_axis, eb_polar, color='black', label='EB Polar', marker='x', s=50)
ax.scatter(x_axis, eb_cartesian, color='blue', label='EB Cartesian', marker='s', s=50)
ax.scatter(x_axis, eb_cnt, color='green', label='EB CNT', marker='^', s=50)

# 分别对每个数据集进行线性拟合，并绘制拟合线
for y_data, color, label in zip([eb_polar, eb_cartesian, eb_cnt], ['black', 'blue', 'green'], ['EB Polar', 'EB Cartesian', 'EB CNT']):
    coef = np.polyfit(x_axis, y_data, 1)
    poly1d_fn = np.poly1d(coef)
    ax.plot(x_axis, poly1d_fn(x_axis), color=color, linestyle='-', label=f'{label} Fit')
    with open(f"/home/wll/akrito/string-method-nucleation/binary-mixtures/two-dimensional-case/other-result/data_eb.log", "a") as f1:
        with redirect_stdout(f1):
            print(f'EB {label}, Slope: {coef[0]}, Intercept: {coef[1]}')


ax.legend(fontsize=14)
plt.xlabel('log(1-c)',fontsize=14)
plt.ylabel("log($E_{br}$)",fontsize=14)
plt.title('Energy Barrier',fontsize=16)
plt.grid(True)

plt.savefig("/home/wll/akrito/string-method-nucleation/binary-mixtures/two-dimensional-case/other-result/Energy Barrier Comparison.pdf")
