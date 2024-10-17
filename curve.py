import matplotlib.pyplot as plt
import numpy as np

# 模拟一些数据
steps = np.arange(0, 100)
# UCF101
accuracies = np.array([64.70,66.66,66.66,68.62,72.54,72.54,72.54,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,74.50,76.47,76.47,76.47,76.47,76.47,76.47,76.47,76.47,76.47,76.47,76.47,76.47,76.47,76.47,76.47,76.47,76.47,76.47,76.47,76.47,76.47,76.47,76.47,76.47,76.47,76.47,76.47,76.47,76.47,76.47,76.47,76.47,76.47,76.47,76.47,76.47])
smoothed_accuracy = np.convolve(accuracies, np.ones(5)/5, mode='valid')  # 使用简单的滑动平均进行平滑处理

# 创建图形
plt.figure(figsize=(8, 5))
plt.plot(steps, accuracies, label='UCF101', color='blue', alpha=0.3)  # 原始数据用淡蓝色表示
plt.plot(steps[2:-2], smoothed_accuracy, label='UCF101 Smoothed', color='blue')  # 平滑后的数据用蓝色表示
plt.xlabel('# steps')
plt.ylabel('Training accuracy')
plt.title('Training Accuracy over Steps on 1 shot')
plt.legend()
plt.grid(True)
plt.show()
