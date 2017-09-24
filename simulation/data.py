import numpy as np
from scipy.misc  import imsave as ims
import matplotlib
matplotlib.use('Agg')
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
import pdb
import pandas as pd
import seaborn as sns
np.set_printoptions(threshold='nan')
sampleNo = 10000
mu1 = np.array([[5, 5]])
mu2 = np.array([[-5, -5]])
Sigma = np.array([[1, 0], [0, 1]])

ss1 = np.random.multivariate_normal(mu1[0],Sigma,sampleNo)
ss2 = np.random.multivariate_normal(mu2[0],Sigma,sampleNo)
ssidx = np.random.random(sampleNo)


ss=np.zeros([sampleNo,2])
pct=0.5
ss[ssidx<pct,:]=ss1[ssidx<pct,:]
ss[ssidx>=pct,:]=ss2[ssidx>=pct,:]



theta=np.array([[3, -1], [1, -3]])
ss=theta.dot(ss.transpose()).transpose()
Sigma = np.array([[1, 0], [0, 1]])
ss1=np.zeros([sampleNo,2])
# pdb.set_trace()
# for i in xrange(ss.shape[0]):
    # ss1[i,:] = np.random.multivariate_normal(ss[i,:],Sigma*0.01)
# ssframe=pd.DataFrame(ss1,columns=['X','Y'])
ssframe=pd.DataFrame(ss,columns=['X','Y'])
ssframe.to_csv('newdata.csv')

# theta = np.array([[1, 0], [0, 1]])
# ss=ss.dot(theta)


sns.jointplot('X','Y', ssframe, kind = 'kde')
# pdb.set_trace()
# plt.scatter(ss[:,0],ss[:,1])
plt.savefig('out.jpg')