import matplotlib.pyplot as plt
import numpy as np
from contextlib import redirect_stdout

'''
plt.rcParams['font.size'] = 9  # 设置全局字体大小
plt.rcParams['font.family'] = 'sans-serif'  # 设置全局字体族
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']  # 指定具体的字体作为首选项
'''

# 数据
x_axis = np.log(1 - np.array([0.92, 0.925, 0.93, 0.935, 0.94, 0.945]))
cr_polar = np.log(np.array([2.58, 2.48, 2.67, 2.93, 3.09, 3.49])*1e-2)
cr_cartesian = np.log(np.array([2.63, 2.68, 2.97, 3.12, 3.53, 4.03])*1e-2)
#cr_cnt = np.log(np.array([2.36, 2.50, 2.65, 2.83, 3.05, 3.30])*1e-2)
cr_cnt = np.log([0.02318293,0.02457383,0.02616761,0.02801185,0.03017032,0.03273064])
# 创建图形和轴
fig, ax = plt.subplots(figsize=(8, 6))

# 为每个数据集绘制散点图
ax.scatter(x_axis, cr_polar, color='black', label='CR Polar', marker='x', s=50)
ax.scatter(x_axis, cr_cartesian, color='blue', label='CR Cartesian', marker='s', s=50)
ax.scatter(x_axis, cr_cnt, color='green', label='CR CNT', marker='^', s=50)

# 分别对每个数据集进行线性拟合，并绘制拟合线
for y_data, color, label in zip([cr_polar, cr_cartesian, cr_cnt], ['black', 'blue', 'green'], ['CR Polar', 'CR Cartesian', 'CR CNT']):
    coef = np.polyfit(x_axis, y_data, 1)
    poly1d_fn = np.poly1d(coef)
    ax.plot(x_axis, poly1d_fn(x_axis), color=color, linestyle='-', label=f'{label} Fit')
    with open(f"/home/wll/akrito/string-method-nucleation/binary-mixtures/two-dimensional-case/other-result/data_cr.log", "a") as f1:
        with redirect_stdout(f1):
            print(f'CR {label}, Slope: {coef[0]}, Intercept: {coef[1]}')


ax.legend(fontsize=14)
plt.xlabel('log(1-c)',fontsize=14)
plt.ylabel('log($R_{cri}$)',fontsize=14)
plt.title('Critical Radius',fontsize=16)
plt.grid(True)

plt.savefig("/home/wll/akrito/string-method-nucleation/binary-mixtures/two-dimensional-case/other-result/Critical Radius Comparison.pdf")
