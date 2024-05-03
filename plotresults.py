import matplotlib.pyplot as plt

x = [10, 15, 20, 30]

fci_gpu  = [0.377, 0.471, 0.59017, 0.61]

fci_cpu= [0.0253 ,0.565, 2.829, 12.25 ]

plt.plot(x, fci_gpu,label='FCI GPU', marker = 'o')
plt.plot(x, fci_cpu, label='FCI CPU', marker = 'o')
plt.xlabel('Number of Nodes (variables)')
plt.ylabel('Time (seconds)')
plt.title('FCI CPU vs GPU \n CPU: Intel i5-13420H, GPU: Nvidia RTX 4050')
# makr data points with a dot

plt.legend()
plt.show()