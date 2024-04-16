import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
data = {'cat':4,'horse':13,'dog':6,'rabbit':15,'cow':21}
names = list(data.keys())
values = list(data.values())
fig,axs = plt.subplots(1,3,figsize=(9,3),sharey=True)

axs[0].bar(names,values)
axs[1].scatter(names,values)
axs[2].plot(names,values)
fig.suptitle('Categorical Plotting')
plt.show()
