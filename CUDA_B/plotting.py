import matplotlib.pyplot as plt

x_axis = [1, 5, 10, 50, 100]

host_time = [0.000527, 0.003429, 0.008623, 0.049367, 0.099582]

q2_1_1 = [0.123697, 0.601515, 1.203512, 6.018537, 12.036772]
q2_1_256 = [0.004165, 0.020743, 0.041488, 0.090408, 0.169470]
q2_v_256 = [0.000115, 0.000520, 0.001026, 0.002557, 0.002215]

q3_1_1 = [0.123727, 0.601493, 1.203517, 6.018562, 12.036735]
q3_1_256 = [0.004695, 0.020723, 0.041441, 0.082914, 0.164990]
q3_v_256 = [0.000124, 0.000531, 0.001035, 0.001128, 0.002238]

plt.xlabel('K')
plt.ylabel('Time (s)')
plt.title('Q2: Without Unified Memory')
plt.yscale('log')

plt.plot(x_axis, host_time, label='host')
plt.plot(x_axis, q2_1_1, label='1 block, 1 thread')
plt.plot(x_axis, q2_1_256, label='1 block, 256 threads')
plt.plot(x_axis, q2_v_256, label='any blocks, 256 threads')
plt.legend()

plt.show()

plt.xlabel('K')
plt.ylabel('Time (s)')
plt.title('Q3: With Unified Memory')
plt.yscale('log')

plt.plot(x_axis, host_time, label='host')
plt.plot(x_axis, q3_1_1, label='1 block, 1 thread')
plt.plot(x_axis, q3_1_256, label='1 block, 256 threads')
plt.plot(x_axis, q3_v_256, label='any blocks, 256 threads')
plt.legend()

plt.show()