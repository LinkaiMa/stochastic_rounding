from SR_tools import *
from tqdm import tqdm
from sklearn.datasets import make_sparse_spd_matrix
import os
n = 100; steps = 2000; digits = 5; avg = 10; rate = 0.99; path = 'SR_no_avg'  # SR_no_avg
if not os.path.exists(path):
    os.makedirs(path)

# generate a random SPD matrix with specified convergence rate for Richardson iteration
A = make_sparse_spd_matrix(n)
A = A/np.linalg.norm(A,ord=2)*rate
A += np.eye(n)
A0 = A.copy()
x_sol = np.ones(shape=n)
b = np.sum(A,axis=1)
b0 = b.copy()

rn_error = np.zeros(shape=steps)
rn_error_avg = np.zeros(shape=steps)
sr_error = np.zeros(shape=steps)
sr_error_avg = np.zeros(shape=steps)
x_rn = np.zeros(shape=n); x_sr = np.zeros(shape=n)
A_rn = RN(A,digits); b_rn = RN(b,digits)
x_vals_rn = np.zeros(shape=(n,steps))
x_vals_sr = np.zeros(shape=(n,steps))

for i in tqdm(range(steps)):
    rn_error[i] = np.linalg.norm(x_rn-x_sol) 
    sr_error[i] = np.linalg.norm(x_sr-x_sol)
    x_rn = RN(x_rn+RN(b_rn-A_rn@x_rn,digits),digits)
    x_vals_rn[:,i] = x_rn
    x_rn_avg = np.average(x_vals_rn[:,i-avg+1:i+1],axis=1)
    rn_error_avg[i] = np.linalg.norm(x_rn_avg-x_sol)
    A_sr = SR(A0,digits); b_sr = SR(b0,digits)
    x_sr = SR(x_sr+SR(b_sr-A_sr@x_sr,digits),digits)
    x_vals_sr[:,i] = x_sr
    x_sr_avg = np.average(x_vals_sr[:,i-avg+1:i+1],axis=1)
    sr_error_avg[i] = np.linalg.norm(x_sr_avg-x_sol)
plt.semilogy(rn_error,label='deterministic rounding')
plt.semilogy(sr_error,label='stochastic rounding')
plt.semilogy(rn_error_avg,label='deterministic rounding (average)') 
plt.semilogy(sr_error_avg,label='stochastic rounding (average)')
plt.semilogy([1/10**digits]*steps,label=str(1/10**digits))
plt.xlabel('# iteration')
plt.ylabel('error')
plt.title('Richardson convergence rate '+str(rate))
plt.legend()
plt.savefig(path+'/Richardson'+path+'.png')
plt.close()


with open(path+'/parameters.txt', 'w') as file:
    file.write("n = "+str(n)+"\n")
    file.write("steps = "+str(steps)+"\n")
    file.write("digits = "+str(digits)+"\n")
    file.write("avg = "+str(avg)+"\n")
    file.write("rate = "+str(rate)+"\n")

np.save(path+'/A.npy',A)
np.save(path+'/x_vals_rn.npy',x_vals_rn)
np.save(path+'/x_vals_sr.npy',x_vals_sr)