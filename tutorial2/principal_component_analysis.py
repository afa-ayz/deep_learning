import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;
sns.set()

n_samples = 200

rng = np.random.RandomState(7) #设置伪随机为了更好的复现
data_raw = np.dot(rng.rand(2,2),rng.randn(2,n_samples))

plt.scatter(data_raw[0,:],data_raw[1,:],marker='x',color='b')
plt.title('Two-dimensional Raw Data')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis('equal')
plt.show()

data_norm = data_raw - data_raw.mean(axis=1).reshape([2,1])

# Compute the Scatter Matrix
Sigma = data_norm.dot(data_norm.T)/n_samples
print(Sigma)

lamda, U = np.linalg.eig(Sigma) # eigen values, eigen vectors
print('lamda=\n',lamda)
print('U=\n',U)

# Make a list of (eigen value, eigen vector) tuples
eig_pairs = [(np.abs(lamda[i]), U[:,i]) for i in range(len(lamda))]

# Sort the (eigen value, eigen vector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# create new eigen vector
lamda_sort_abs = np.zeros(len(lamda))
U_sort = np.zeros([len(lamda),len(lamda)])
for i in range(len(lamda)):
    lamda_sort_abs[i] = eig_pairs[i][0]
    U_sort[:,i] = eig_pairs[i][1]
print(U_sort)

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# plot data
plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.scatter(data_norm[0,:], data_norm[1,:], marker='x', color='b', alpha=0.2)
origin = np.zeros([len(lamda)])
for i in range(len(lamda)):
    direction = U[:,i] * np.sqrt(np.abs(lamda[i])) * 2
    draw_vector(origin, origin + direction)
plt.title('Two-dimensional Normalized Data')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis('equal')

plt.subplot(122)
# Compute transformed data
# To do here (1)
data_trans = \
plt.scatter(data_trans[0,:], data_trans[1,:], marker='x', color='b', alpha=0.2)
plt.title('Two-dimensional Transformed Data')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis('equal')