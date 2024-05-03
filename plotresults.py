import matplotlib.pyplot as plt

# x = [10, 15, 20, 30, 80]

# fci_gpu  = [0.377, 0.471, 0.59017, 0.61, 1.89]

# fci_cpu= [0.0253 ,0.565, 2.829, 12.25 ]
x = [15,20,30,40,50,60,70,80]
fci_gpu = [0.03203, 0.039, 0.059, 0.106295, 0.136, 0.185966, 0.2389, 0.3237]

plt.plot(x, fci_gpu,label='FCI GPU', marker = 'o')
# plt.plot(x, fci_cpu, label='FCI CPU', marker = 'o')
plt.xlabel('Number of Nodes (variables)')
plt.ylabel('Time (seconds)')
plt.title('FCI CPU vs GPU \n CPU: Intel i5-13420H, GPU: Nvidia RTX 4050')
# makr data points with a dot

plt.legend()
plt.show()

nn = [2.398936, 4.446936, 6.494936, 8.542936, 10.590936]
xx = [15,20,30,40,50,60,70,80]
fci_gpu = [0.03203, 0.039, 0.59, 0.106295, 0.136, 0.185966, 0.2389, 0.3237]