import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 创建一个示例的排序后DataFrame
data = {'Value': np.arange(1, 101)}
sorted_df = pd.DataFrame(data)
print(data)

# 计算DataFrame的行数
num_rows = len(sorted_df)

# 设定对数方式的采样点
log_sample_points = np.logspace(0, np.log2(num_rows), num=10, endpoint=False, base=2.0, dtype=int)

print(np.log2(num_rows))

# 从DataFrame中获取对应索引的数据
log_sampled_data = sorted_df.iloc[log_sample_points]

# 打印结果
print(log_sample_points)


x1 = log_sample_points
y = np.zeros(len(log_sample_points))
plt.plot(x1, y, 'o', alpha = 1)
plt.ylim([-0.5, 1])
plt.show()