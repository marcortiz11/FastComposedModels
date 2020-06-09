import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Create some mock data
t = [5342/s for s in np.arange(1, 3.25, 0.25)]
s = [1, 1.46, 1.94, 2.34, 2.62, 2.81, 2.91, 2.97]
nodes = [pow(2, i) for i in range(8)]


fig, ax1 = plt.subplots()
plt.title("G.A. Roulette")
plt.grid(linestyle='-', linewidth=0.5)
plt.xlim((1, 128))
plt.xticks(list(range(16, 129, 16))+[1], list(range(16, 129, 16))+[1])

# Figure 1
color = 'tab:red'
ax1.set_xlabel('# Nodes')
ax1.set_ylabel('Speedup')
ax1.plot(nodes, s, color=color)

# Figure 2
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Time (s)')
plt.yticks(np.arange(1, 3.25, 0.25), ["%d"%t_ for t_ in t])
ax2.plot(nodes, s, color=color)

fig.tight_layout()
plt.show()


"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Create some mock data
t = [1838, 962, 496, 256, 132, 68, 36, 18]
s = [1, 1.91, 3.70, 7.17, 13.92, 27.02, 51.05, 102.11]
nodes = [pow(2, i) for i in range(8)]

fig, ax1 = plt.subplots()
plt.grid(linestyle='-', linewidth=0.5)

ax1.set_title("G.A. most fit")
plt.xscale('log', basex=2)
plt.yscale('log', basey=2)
plt.xticks(nodes, nodes)
color = 'tab:red'
ax1.set_xlabel('# Nodes')
ax1.set_ylabel('Speedup')
ax1.plot(nodes, s, color=color)


ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
plt.xscale('log', basex=2)
plt.yscale('log', basey=2)
plt.xticks(nodes, nodes)
plt.yticks(nodes, [1838/n for n in nodes])

ax2.set_ylabel('Time (s)')  # we already handled the x-label with ax1
ax2.plot(nodes, s, color=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
"""