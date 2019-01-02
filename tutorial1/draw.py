from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

# %config InlineBackend.figure_format="svg" #使用svg格式显示在网页上
import numpy as np
import matplotlib.pyplot as plt

I = np.linspace(100,10000)

print(I)


t1=(0.14*0.025*1)/((I/67.9)**0.02-1)#LV side of 750kVA
t2=(0.14*0.025*5)/((I/180)**0.02-1)#HV side of 750kVA
t3=(0.14*0.025*14)/((I/480)**0.02-1)#Relay C
t4=(0.14*0.025*8)/((I/2250)**0.02-1)#Relay B
t5=(0.14*0.025*15)/((I/2250)**0.02-1)#Relay A
t6=I#Relay D
print(t1)
plt.subplot(321)
plt.plot(I, t1, '-')
plt.title('LV side of 750kVA')
plt.xlabel('Current (A)')
plt.ylabel('Time (s)')


plt.subplot(322)
plt.plot(I, t2, '-')
plt.title('HV side of 750kVA')
plt.xlabel('Current (A)')
plt.ylabel('Time (s)')

plt.subplot(323)
plt.plot(I, t3, '-')
plt.title('Relay C')
plt.xlabel('Current (A)')
plt.ylabel('Time (s)')

plt.subplot(324)
plt.plot(I, t4, '-')
plt.title('Relay B')
plt.xlabel('Current (A)')
plt.ylabel('Time (s)')

plt.subplot(325)
plt.plot(I, t5, '-')
plt.title('Relay A')
plt.xlabel('Current (A)')
plt.ylabel('Time (s)')

plt.subplot(326)
plt.plot(I, t6, '-')
plt.title('Differential Relay')
plt.xlabel('Current LV(A)')
plt.ylabel('Current HV(A)')

plt.show()
