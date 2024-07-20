'''The string method is not used, but a single image is used to iterate to see if it can converge to nucleation'''

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
import os
from line_profiler import LineProfiler
from contextlib import redirect_stdout
import matplotlib.patches as mpatches
jax.config.update("jax_enable_x64", True)

### THE ITERATION PARAMETERs
#physical parameters
delta = 0.015 #interface width
Lambda = jnp.sqrt(2)*delta/4

gamma_ab = 1e-2 #interface energy between components A,B
gamma_bc = 1e-2
gamma_ca = 7e-3
gamma = jnp.array([gamma_ab, gamma_bc, gamma_ca])

if gamma_ca>1e-2:
    Ac = (9*gamma_ca + 12*(gamma_bc - gamma_ab))/(4*jnp.sqrt(2)*Lambda)
    Aa = (9*gamma_ca - 12*(gamma_bc - gamma_ab))/(4*jnp.sqrt(2)*Lambda)
    Ab = 3*jnp.sqrt(2)*gamma_bc/Lambda - 4*Ac/3 - Aa/3 
else:
    Ab = jnp.sqrt(2)*(9*gamma_bc + 12*(gamma_ab - gamma_ca))/(8*Lambda)
    Ac = jnp.sqrt(2)*(9*gamma_bc - 12*(gamma_ab - gamma_ca))/(8*Lambda)
    Aa = 3*jnp.sqrt(2)*gamma_ab/Lambda - 4*Ab/3 - Ac/3

A = jnp.array([Aa, Ab, Ac])

C = jnp.max(A + jnp.roll(A, 1))/6

rc = 0.18 # the final radius of conponent C
mean_uc = 4*jnp.pi*jnp.power(rc,3)/3 #The mean concentration of component c

#numerical parameters
stepmax = 800000 # max number of iterations
N = 256 #num of nodes
eps = 1e-14 #用于处理在计算化学势时P约等于0所引起的numerical error

#meshigird
l = 1
dr = l/N
r = jnp.linspace(0, l, N+1)
r_mid = (r[1:] + r[:-1]) / 2
rx, ry, rz = jnp.meshgrid(r_mid, r_mid, r_mid)

#频率空间的网格
k = 2*jnp.pi*jnp.fft.fftfreq(N, d=dr)
kx, ky, kz = jnp.meshgrid(k, k, k)
k_square = kx**2 + ky**2 + kz**2
print(mean_uc)

save_dir = f"result/gamma_ac-{gamma_ca}/heterogeneous"
os.makedirs(save_dir, exist_ok=True)

def initialize(f):
    index = jnp.argmax(f)

    R = rc/3
    xc = (4*jnp.pi*jnp.power(rc,3)/3 - 4*jnp.pi*jnp.power(R,3)/3) / (jnp.power(l,3) - 4*jnp.pi*jnp.power(R,3)/3)

    uc = jnp.full((N, N, N), xc)
    ua = jnp.full((N, N, N), 1 - xc)
    ub = jnp.full((N, N, N), 1 - xc)

    d = (rx - l/2)**2 + (ry - l/2)**2 + (rz - 0.5)**2
    sphere_check = d<=R**2
    semicubic_check = rz>0.5

    uc = jnp.where(sphere_check, 1, uc)
    ua = jnp.where(semicubic_check, 0, ua)
    ua = jnp.where(sphere_check, 0, ua)
    ub = jnp.where(semicubic_check, ub, 0)
    ub = jnp.where(sphere_check, 0, ub)

    ua = ua*0.5*(1 - mean_uc)/jnp.mean(ua)
    ub = ub*0.5*(1 - mean_uc)/jnp.mean(ub)
    uc = uc*mean_uc/jnp.mean(uc)
    print(jnp.mean(ua + ub + uc))
    return ua, ub, uc

@jax.jit
def cal_ChemicalPotential(i, u, S, P):#u为某一组分i的浓度profile
    mu = 2*A[i]*u*(1 - u)*(1 - 2*u) + 2*C*(u*(S - u**2) - (1-u)*(P + eps)/(jnp.power(1-u, 2) + eps))
    return mu

@jax.jit
def pde_solver_FFT(h, ua, ub, uc):
    ua_hat = jnp.fft.fftn(ua, axes=(0,1,2))
    ub_hat = jnp.fft.fftn(ub, axes=(0,1,2))
    uc_hat = jnp.fft.fftn(uc, axes=(0,1,2))

    S = ua**2 + ub**2 +uc**2
    P = jnp.power(1-ua, 2)*jnp.power(1-ub, 2)*jnp.power(1-uc, 2)

    mu_a_hat = jnp.fft.fftn(cal_ChemicalPotential(0, ua, S, P), axes=(0,1,2))
    mu_b_hat = jnp.fft.fftn(cal_ChemicalPotential(1, ub, S, P), axes=(0,1,2))
    mu_c_hat = jnp.fft.fftn(cal_ChemicalPotential(2, uc, S, P), axes=(0,1,2))

    ua_hat_new = (ua_hat - h*k_square*mu_a_hat) / (1 + h*jnp.power(Lambda, 2)*(A[0] + C)*jnp.power(k_square, 2))
    ub_hat_new = (ub_hat - h*k_square*mu_b_hat) / (1 + h*jnp.power(Lambda, 2)*(A[1] + C)*jnp.power(k_square, 2))
    uc_hat_new = (uc_hat - h*k_square*mu_c_hat) / (1 + h*jnp.power(Lambda, 2)*(A[2] + C)*jnp.power(k_square, 2))

    ua_new = jnp.fft.ifftn(ua_hat_new, axes=(0,1,2)).real
    ub_new = jnp.fft.ifftn(ub_hat_new, axes=(0,1,2)).real
    uc_new = jnp.fft.ifftn(uc_hat_new, axes=(0,1,2)).real

    return ua_new, ub_new, uc_new


def main(ua, ub, uc, load, usedata):
    h =  5e-6# time-step (limited by the ODE step but independent of n1)
    dh = (1e-5 - 5e-6)/(stepmax/1000)
    print(h, dh)
    if load==True : 
        u = np.load(f"result/gamma_ac-{gamma_ca}/heterogeneous/concentration.npy")
        ua = jnp.array(u[0])
        ub = jnp.array(u[1])
        uc = jnp.array(u[2])
    
        if usedata == True:
            for step in range(stepmax):
                ua0 = ua
                ub0 = ub
                uc0 = uc
                ua, ub, uc = pde_solver_FFT(h, ua, ub, uc)
                ua = ua*0.5*(1 - mean_uc)/jnp.mean(ua)
                ub = ub*0.5*(1 - mean_uc)/jnp.mean(ub)
                uc = uc*mean_uc/jnp.mean(uc)
                if step%1000==0: 
                    #h += dh
                    err = (jnp.linalg.norm(ua - ua0) + jnp.linalg.norm(ub - ub0) + jnp.linalg.norm(uc - uc0)) / 3
                    print(f"iteration {step} has done, err {err}, h {h}, dh {dh}")    
                if step==stepmax-1: 
                    err = (jnp.linalg.norm(ua - ua0) + jnp.linalg.norm(ub - ub0) + jnp.linalg.norm(uc - uc0)) / 3
                    print(f"iteration {step} has done, err {err}")
            return ua, ub, uc
        
        return ua, ub, uc

    else:
        for step in range(1, stepmax):
            ua0 = ua
            ub0 = ub
            uc0 = uc
            ua, ub, uc = pde_solver_FFT(h, ua, ub, uc)
            ua = ua*0.5*(1 - mean_uc)/jnp.mean(ua)
            ub = ub*0.5*(1 - mean_uc)/jnp.mean(ub)
            uc = uc*mean_uc/jnp.mean(uc)
            if step%1000==0: 
                h += dh
                err = (jnp.linalg.norm(ua - ua0) + jnp.linalg.norm(ub - ub0) + jnp.linalg.norm(uc - uc0)) / 3
                print(f"iteration {step} has done, err {err}, h {h}, dh {dh}")    
            if step==stepmax-1: 
                err = (jnp.linalg.norm(ua - ua0) + jnp.linalg.norm(ub - ub0) + jnp.linalg.norm(uc - uc0)) / 3
                print(f"iteration {step} has done, err {err}")
    return ua, ub, uc

def cal_energy(ua, ub, uc):
    bulk_term = Aa*jnp.power(ua, 2)*jnp.power(ua-1, 2) + Ab*jnp.power(ub, 2)*jnp.power(ub-1, 2) + Ac*jnp.power(uc, 2)*jnp.power(uc-1, 2) + C*(jnp.power(ua,2)*jnp.power(ub, 2) + jnp.power(ua,2)*jnp.power(uc,2) + jnp.power(uc,2)*jnp.power(ub,2) + jnp.power(1-ua,2)*jnp.power(1-ub,2)*jnp.power(1-uc,2))
    
    Wa = jnp.power(Lambda, 2)*(Aa + C)
    Wb = jnp.power(Lambda, 2)*(Ab + C)
    Wc = jnp.power(Lambda, 2)*(Ac + C)
    
    @jax.jit
    def cal_gradsquare(x):
        xh = jnp.fft.fftn(x, axes=(0,1,2))
        px = jnp.fft.ifftn(kx*1j*xh, axes=(0,1,2)).real
        py = jnp.fft.ifftn(ky*1j*xh, axes=(0,1,2)).real
        pz = jnp.fft.ifftn(kz*1j*xh, axes=(0,1,2)).real
        return px**2 + py**2 + pz**2
    interface_term = Wa*cal_gradsquare(ua)/2 + Wb*cal_gradsquare(ub)/2 + Wc*cal_gradsquare(uc)/2

    G = bulk_term + interface_term
    return jnp.mean(G)

def plot_RGB(ua, ub, uc, name, savefig):
    u = jnp.stack((ua, ub, uc), axis=2)
    plt.figure()
    plt.gca().set_aspect('equal')
    plt.imshow(u, interpolation='nearest', origin='lower')
    plt.title(name)
    plt.xticks([])
    plt.yticks([])

    patch1 = mpatches.Patch(color='red', label='Phase A')
    patch2 = mpatches.Patch(color='lightgreen', label='Phase B')
    patch3 = mpatches.Patch(color='blue', label='Phase C')
    plt.legend(handles=[patch1, patch2, patch3], loc='upper right', prop={'size': 7})
    
    if savefig == True: plt.savefig(f"result/gamma_ac-{gamma_ca}/heterogeneous/{name}.pdf")

def Energy_along_string(F, Fc, savefig):
    plt.figure()
    plt.plot(range(len(F)), F)
    plt.plot(range(len(F)), np.full(len(F), Fc), linestyle='--')
    plt.title('Energy along the string')
    plt.tight_layout()
    if savefig == True: plt.savefig(f"result/gamma_ac-{gamma_ca}/heterogeneous/Energy_along_string.pdf") 

if __name__=="__main__":
    f = np.load(f"/home/ms/akrito/string-method-nucleation/ternary-mixtures/three-dimensional-case/different-gamma-ac/result/gamma_ac-{gamma_ca}/heterogeneous/free-energy.npy")
    uai, ubi, uci = initialize(f=f)
    uaf, ubf, ucf = main(ua=uai, ub=ubi, uc=uci, load=False, usedata=True)
    print(f"sum: {jnp.mean(uaf + ubf + ucf)}")
    #F = cal_energy(ua=uaf, ub=ubf, uc=ucf)
    #Energy_along_string(F=f, Fc=F, savefig=True)
    np.save(f'result/gamma_ac-{gamma_ca}/heterogeneous/concentration.npy', np.array(jnp.stack([uaf, ubf, ucf])))

    m = int(N/2)
    mid_ua = (uaf[m, :, :] + uaf[:, m, :] + uaf[m-1, :, :] + uaf[:, m-1, :])/4
    mid_ub = (ubf[m, :, :] + ubf[:, m, :] + ubf[m-1, :, :] + ubf[:, m-1, :])/4
    mid_uc = (ucf[m, :, :] + ucf[:, m, :] + ucf[m-1, :, :] + ucf[:, m-1, :])/4

    plot_RGB(mid_ua, mid_ub, mid_uc, name="concentration profile", savefig=True)
    with open(f"result/gamma_ac-{gamma_ca}/heterogeneous/data.log", "w") as f1:
        with redirect_stdout(f1):
            print(f"delta : {delta}")
            print(f"gamma[AB, BC, CA] : {gamma}")
            print(f"A[a, b ,c] : {A}")
            print(f"experimental A: [{jnp.sqrt(2)*Lambda*(Aa + Ab + 2*C)/6}, {jnp.sqrt(2)*Lambda*(Ac + Ab + 2*C)/6}, {jnp.sqrt(2)*Lambda*(Aa + Ac + 2*C)/6}]")
            print(f"epsilon : {eps}")
            print(f"mean_uc : {mean_uc}")
            print(f"N : {N}\n")

            print("------------------------------------------------------\n")
            #print(f"experomental energy: {F}")
            print("------------------------------------------------------\n")



    