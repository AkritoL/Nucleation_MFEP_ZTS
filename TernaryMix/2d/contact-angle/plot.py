import numpy as np
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
import matplotlib.patches as mpatches
import os
import matplotlib.pylab as pylab
import numpy as np
from PIL import Image
myparams = {

   'axes.labelsize': '12',

   'xtick.labelsize': '12',

   'ytick.labelsize': '12',

   'lines.linewidth': 1,

   'legend.fontsize': '10',

   'font.family': 'Times New Roman',

   'figure.figsize': '6, 4'  #图片尺寸

}
plt.rcParams['image.interpolation'] = 'nearest' # 设置 interpolation style
plt.rcParams['image.cmap'] = 'gray' # 设置 颜色 style
pylab.rcParams.update(myparams)  #更新自己的设置
# 数据
gamma_bc = [2.00e-3, 4.00e-3, 6.00e-3, 8.00e-3, 1.00e-2, 1.20e-2, 1.40e-2, 1.60e-2, 1.80e-2]
theoretical_CA = [84.26082952273322, 78.46, 72.54, 66.42, 60.00, 53.13, 45.57, 36.87, 25.84]
#experimental_CA = [78.69, 71.57, 68.20, 61.43, 53.13, 45.00, 36.87, 26.57]
#conf_interval_lower = [63.11, 65.23, 61.03, 57.87, 46.71, 39.99, 32.18, 23.91]
#conf_interval_upper = [96.12, 78.40, 76.16, 69.59, 60.84, 51.06, 42.87, 29.83]

experimental_CA = [84.49237413608161, 78.74690828592469, 73.78702627473383, 68.12833704056262, 62.92372624903132, 56.57366571400147, 50.74405165067153, 45.0, 32.086356479510396]
conf_interval_lower = [72.86344955080757, 65.62307500059119, 57.919784945777245, 56.415087271093725, 53.46704305014896, 46.653344597225455, 44.2549110914756, 40.43311772424793, 27.296832542296006]
conf_interval_upper = [98.59626040084883, 89.60950032403187, 91.22596262605069, 82.98071034470873, 81.08792232771467, 71.79214367922836, 62.99468557358884, 51.70022250081691, 36.11316587004259]

# 将置信区间转换为yerr需要的上下界差值格式
conf_interval_yerr = [(experimental_CA[i] - conf_interval_lower[i], conf_interval_upper[i] - experimental_CA[i]) for i in range(len(experimental_CA))]
conf_interval_yerr = list(zip(*conf_interval_yerr))  # 解包为两个列表

# 更新图例标签并重新配色
plt.figure(dpi=500)

# 绘制理论和实验值，使用更符合科学论文规范的配色
plt.plot(gamma_bc, theoretical_CA, color='darkorange', label='Theoretical contact angle', marker='o', markersize=3, linestyle='--')
plt.plot(gamma_bc, experimental_CA, color='royalblue', label='Experimental contact angle', marker='s', markersize=5)

# 绘制置信区间，使用更淡的色彩以保持图表整洁
plt.fill_between(gamma_bc, conf_interval_lower, conf_interval_upper, color='lightgrey', alpha=0.5)

# 更新标注
plt.xlabel('Interface energy $\gamma_{bc}$',fontsize=12)
plt.ylabel('Contact Angles (in degrees)',fontsize=12)
plt.title('Angles at Triple Junctions',fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5)

# 显示更新后的图表
plt.savefig("/home/ms/akrito/string-method-nucleation/ternary-mixtures/two-dimensional-case/contact-angle/comparison.pdf", dpi=500, bbox_inches='tight')
