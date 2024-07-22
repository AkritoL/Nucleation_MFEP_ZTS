import os
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from line_profiler import LineProfiler
from tqdm import *
from contextlib import redirect_stdout
jax.config.update("jax_enable_x64", True)


# Parameters Transmission
parser = argparse.ArgumentParser(description="Process some parameters.")
parser.add_argument("--NUM_GPU", type=int, help="An integer for the parameter NUM_GPU")
parser.add_argument("--NUCLEATION_MODE", type=str, help="A str for the parameter NUCLEATION_MODE") 
parser.add_argument("--CASE", type=str, help="A str for the parameter CASE") 
parser.add_argument("--INTERFACE_WIDTH", type=float, help="A float for the parameter INTERFACE_WIDTH")
parser.add_argument("--GAMMA_AB", type=float, help="A float for the parameter GAMMA_AB")
parser.add_argument("--GAMMA_BC", type=float, help="A float for the parameter GAMMA_BC")
parser.add_argument("--GAMMA_CA", type=float, help="A float for the parameter GAMMA_CA")
parser.add_argument("--FINAL_RADIUS", type=float, help="A float for the parameter FINAL_RADIUS")
parser.add_argument("--STEPMAX", type=int, help="An integer for the parameter STEPMAX")
parser.add_argument("--TIME_STEP", type=float, help="A float for the parameter TIME_STEP")
parser.add_argument("--TIME_STEP_INCREMENT", type=float, help="A float for the parameter TIME_STEP_INCREMENT")
parser.add_argument("--GRID_NUMBER", type=int, help="An integer for the parameter GRID_NUMBER")
parser.add_argument("--NUM_STRING_IMAGES", type=int, help="An integer for the parameter NUM_STRING_IMAGES")
parser.add_argument("--OUTPUT_DIR", type=str, help="A str for the parameter OUTPUT_DIR")  
parser.add_argument("--PLOT_ENERGY", type=lambda x: (str(x).lower() == 'true'), help="A bool for the parameter PLOT_ENERGY")  
parser.add_argument("--PLOT_CONCENTRATION_RGB", type=lambda x: (str(x).lower() == 'true'), help="A bool for the parameter PLOT_CONCENTRATION_RGB")  
parser.add_argument("--SAVE_CONCENTRATION_DATA", type=lambda x: (str(x).lower() == 'true'), help="A bool for the parameter SAVE_CONCENTRATION_DATA")  
parser.add_argument("--SAVE_ENERGY_DATA", type=lambda x: (str(x).lower() == 'true'), help="A bool for the parameter SAVE_ENERGY_DATA")  
parser.add_argument("--PRE_DATA_LOAD", type=lambda x: (str(x).lower() == 'true'), help="A bool for the parameter PRE_DATA_LOAD")   

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.NUM_GPU)
os.makedirs(args.OUTPUT_DIR, exist_ok=True)


### THE PARAMETERS
#physical parameters
delta = args.INTERFACE_WIDTH #interface width
Lambda = jnp.sqrt(2)*delta/4 #a related para

gamma_ab = args.GAMMA_AB #gamma refers to surface tension
gamma_bc = args.GAMMA_BC
gamma_ca = args.GAMMA_CA

if args.CASE == "SymmetricCase":
    if gamma_ab > 1e-2:
        Aa = (9*gamma_ab + 12*(gamma_ca - gamma_bc))/(4*jnp.sqrt(2)*Lambda)
        Ab = (9*gamma_ab - 12*(gamma_ca - gamma_bc))/(4*jnp.sqrt(2)*Lambda)
        Ac = 3*jnp.sqrt(2)*gamma_ca/Lambda - 4*Aa/3 - Ab/3 
    else:
        Ab = jnp.sqrt(2)*(9*gamma_bc + 12*(gamma_ab - gamma_ca))/(8*Lambda)
        Ac = jnp.sqrt(2)*(9*gamma_bc - 12*(gamma_ab - gamma_ca))/(8*Lambda)
        Aa = 3*jnp.sqrt(2)*gamma_ab/Lambda - 4*Ab/3 - Ac/3
elif args.CASE == "AsymmetricCase":
    if gamma_ca>1e-2:
        Ac = (9*gamma_ca + 12*(gamma_bc - gamma_ab))/(4*jnp.sqrt(2)*Lambda)
        Aa = (9*gamma_ca - 12*(gamma_bc - gamma_ab))/(4*jnp.sqrt(2)*Lambda)
        Ab = 3*jnp.sqrt(2)*gamma_bc/Lambda - 4*Ac/3 - Aa/3 
    else:
        Ab = jnp.sqrt(2)*(9*gamma_bc + 12*(gamma_ab - gamma_ca))/(8*Lambda)
        Ac = jnp.sqrt(2)*(9*gamma_bc - 12*(gamma_ab - gamma_ca))/(8*Lambda)
        Aa = 3*jnp.sqrt(2)*gamma_ab/Lambda - 4*Ab/3 - Ac/3
else: 
    raise NameError("CASE name incorrectly used, we only have SymmetricCase or AsymmetricCase.")

A = jnp.array([Aa, Ab, Ac])
C = jnp.max(A + jnp.roll(A, 1))/6

rf = args.FINAL_RADIUS # final radius of conponent C, which is going to nucleate
mean_uc = jnp.pi*rf**2 #The mean concentration of component c

#numerical parameters
stepmax = args.STEPMAX # max number of iterations
h = args.TIME_STEP # time-step (limited by the ODE step but independent of N)
dh = args.TIME_STEP_INCREMENT 
N = args.GRID_NUMBER #num of nodes
image_num = args.NUM_STRING_IMAGES # num of string images
file_path = args.OUTPUT_DIR

#meshigird
l = 1 #mesh_len
dr = l/N
r = jnp.linspace(0, l, N+1)
r_mid = (r[1:] + r[:-1]) / 2
rx, ry = jnp.meshgrid(r_mid, r_mid)

#frequency space
k = 2*jnp.pi*jnp.fft.fftfreq(N, d=dr)
kx, ky = jnp.meshgrid(k, k)
k_square = kx**2 + ky**2


@jax.jit
def generator(R, ratioAB):  #phase a is above the interface and R is the intermediate radus of the droplets
    '''
    To generate each initial image on the string, in heterogeneous case, there are two caps,
    while in homogeneous case, there are a circle in phase A
    '''
    xc = (jnp.pi*rf**2 - jnp.pi*R**2) / (l**2 - jnp.pi*R**2)

    if args.NUCLEATION_MODE == "homogeneous":
        if args.CASE == "AsymmetricCase" and gamma_ca > 1e-2:
            uc = jnp.full((N, N), xc)
            ub = jnp.full((N, N), 1 - xc)
            ua = jnp.zeros((N, N))

            d = (rx - 0.5)**2 + (ry - ratioAB/2)**2
            sphere_check = d<R**2
            semisquare_check = ry>ratioAB

            uc = jnp.where(sphere_check, 1.0, uc)
            ub = jnp.where(semisquare_check, mean_uc - xc, ub)
            ub = jnp.where(sphere_check, 0., ub)
            ua = jnp.where(semisquare_check, 1 - mean_uc, ua)
            ua = jnp.where(sphere_check, 0., ua)
        else:
            uc = jnp.full((N, N), xc)
            ua = jnp.full((N, N), 1 - xc)
            ub = jnp.zeros((N, N))

            d = (rx - 0.5)**2 + (ry - 3*ratioAB/2)**2
            sphere_check = d<R**2
            semisquare_check = ry<ratioAB

            uc = jnp.where(sphere_check, 1.0, uc)
            ua = jnp.where(semisquare_check, mean_uc - xc, ua)
            ua = jnp.where(sphere_check, 0., ua)
            ub = jnp.where(semisquare_check, 1 - mean_uc, ub)
            ub = jnp.where(sphere_check, 0., ub)
    elif args.NUCLEATION_MODE == "heterogeneous":
        uc = jnp.full((N, N), xc)
        ua = jnp.full((N, N), 1 - xc)
        ub = jnp.zeros((N, N))

        d = (rx - 0.5)**2 + (ry - ratioAB)**2
        sphere_check = d<R**2
        semisquare_check = ry<ratioAB

        uc = jnp.where(sphere_check, 1, uc)
        ua = jnp.where(semisquare_check, 0, ua)
        ua = jnp.where(sphere_check, 0, ua)
        ub = jnp.where(semisquare_check, 1 - xc, ub)
        ub = jnp.where(sphere_check, 0, ub)
    else:
        raise NameError("NUCLEATION_MODE name incorrectly used, we only have homogeneous or heterogeneous.")

    '''
    key = jax.random.PRNGKey(seed=0)
    noise = jax.random.normal(key, shape=(N, N), dtype=jnp.float64) * 1e-8
    ua = ua + noise
    ub = ub + noise
    uc = uc + noise
    '''

    ua = ua*ratioAB*(1 - mean_uc)/jnp.mean(ua)
    ub = ub*ratioAB*(1 - mean_uc)/jnp.mean(ub)
    uc = uc*mean_uc/jnp.mean(uc)
    return ua, ub, uc


# Initialize the string
def initialize(ratioAB):
    A = []
    B = []
    C = []
    for i in range(image_num):
        R = rf*i/(image_num-1)
        ua, ub, uc = generator(R, ratioAB)
        A.append(ua)
        B.append(ub)
        C.append(uc)
    return jnp.array(A), jnp.array(B), jnp.array(C)


#compute the interface chemical potential
@jax.jit
def cal_ChemicalPotential(i, u, S, P):#u为某一组分i的浓度profile
    eps=1e-13
    mu = 2*A[i]*u*(1 - u)*(1 - 2*u) + 2*C*(u*(S - u**2) - (1-u)*(P + eps)/(jnp.power(1-u, 2) + eps))
    return mu


#FFT to Fourier space to solve the ODE
@jax.jit
def pde_solver_FFT(h, ua, ub, uc):
    ua_hat = jnp.fft.fft2(ua)
    ub_hat = jnp.fft.fft2(ub)
    uc_hat = jnp.fft.fft2(uc)

    S = ua**2 + ub**2 +uc**2
    P = jnp.power(1-ua, 2)*jnp.power(1-ub, 2)*jnp.power(1-uc, 2)

    mu_a_hat = jnp.fft.fft2(cal_ChemicalPotential(0, ua, S, P))
    mu_b_hat = jnp.fft.fft2(cal_ChemicalPotential(1, ub, S, P))
    mu_c_hat = jnp.fft.fft2(cal_ChemicalPotential(2, uc, S, P))

    ua_hat_new = (ua_hat - h*k_square*mu_a_hat) / (1 + h*jnp.power(Lambda, 2)*(A[0] + C)*jnp.power(k_square, 2))
    ub_hat_new = (ub_hat - h*k_square*mu_b_hat) / (1 + h*jnp.power(Lambda, 2)*(A[1] + C)*jnp.power(k_square, 2))
    uc_hat_new = (uc_hat - h*k_square*mu_c_hat) / (1 + h*jnp.power(Lambda, 2)*(A[2] + C)*jnp.power(k_square, 2))

    ua_new = jnp.fft.ifft2(ua_hat_new).real
    ub_new = jnp.fft.ifft2(ub_hat_new).real
    uc_new = jnp.fft.ifft2(uc_hat_new).real

    return ua_new, ub_new, uc_new


# reparametrize the string using arc-length reparametrization
@jax.jit
def reparameterize(images):
    diff = images[1:, :, :] - images[:-1, :, :]
    dist = jnp.sum(diff**2, axis=(1, 2))
    l = jnp.sqrt(jnp.concatenate((jnp.array([0.]), dist)))
    cumsum = jnp.cumsum(l) 
    cumsum = cumsum / cumsum[-1]

    @jax.jit
    def interplate(x, i):
        return jax.jit(jnp.interp)(jnp.linspace(0, 1, image_num), cumsum, x[:, i])
    
    return jax.vmap(interplate, in_axes=(None, 0), out_axes=1)(images.reshape((image_num, N**2)), jnp.arange(N**2)).reshape(images.shape)


# To cal the Error, L2 norm
@jax.jit
def mynorm(err):
    return jnp.sum(jax.vmap(jax.jit(jnp.linalg.norm), in_axes=0)(err)) / image_num


#main loop of this program
def main_loop(ua, ub, uc, h, ratioAB):
    if args.PRE_DATA_LOAD:
        u = jnp.load(os.path.join(file_path, "concentration.npy"))
        ua = u[0]
        ub = u[1]
        uc = u[2]

    print("\n<<<Simulation Start>>>")
    print("__CASE INFORMATION__:")
    print(f"2d, {args.CASE}, nucleation_mode = {args.NUCLEATION_MODE}")
    print(f"Gamma_ab={jnp.sqrt(2)*Lambda*(A[0] + A[1] + 2*C)/6: .3f}, Gamma_bc={jnp.sqrt(2)*Lambda*(A[2] + A[1] + 2*C)/6: .3f}, Gamma_ca={jnp.sqrt(2)*Lambda*(A[0] + A[2] + 2*C)/6:.3f}")
    print(f"time_step={args.TIME_STEP}, time_step_increment={args.TIME_STEP_INCREMENT}\n")
    for step in tqdm(range(stepmax)):
        ua0 = ua
        ub0 = ub
        uc0 = uc
        ua, ub, uc = pde_solver_FFT(h, ua, ub, uc)
        ua = reparameterize(ua)
        ub = reparameterize(ub)
        uc = reparameterize(uc)

        ua = ua*ratioAB*(1 - mean_uc)/jnp.mean(ua)
        ub = ub*ratioAB*(1 - mean_uc)/jnp.mean(ub)
        uc = uc*mean_uc/jnp.mean(uc)
        if step%10000==0: 
            h += dh
    err = (mynorm(ua - ua0) + mynorm(ub - ub0) + mynorm(uc - uc0)) / 3
    print(f"\n{step + 1} iterations have done, error={err}")
    print("<<<Simulation End>>>\n")
    return ua, ub, uc, err


#compute the free energy (bulk + interface)
def cal_energy(ua, ub, uc):
    bulk_term = Aa*jnp.power(ua, 2)*jnp.power(ua-1, 2) + Ab*jnp.power(ub, 2)*jnp.power(ub-1, 2) + Ac*jnp.power(uc, 2)*jnp.power(uc-1, 2) + C*(jnp.power(ua,2)*jnp.power(ub, 2) + jnp.power(ua,2)*jnp.power(uc,2) + jnp.power(uc,2)*jnp.power(ub,2) + jnp.power(1-ua,2)*jnp.power(1-ub,2)*jnp.power(1-uc,2))
    
    Wa = jnp.power(Lambda, 2)*(Aa + C)
    Wb = jnp.power(Lambda, 2)*(Ab + C)
    Wc = jnp.power(Lambda, 2)*(Ac + C)
    
    @jax.jit
    def cal_gradsquare(x):
        xh = jnp.fft.fft2(x)
        p = jnp.fft.ifft2(k_square*xh).real
        return x*p
    interface_term = Wa*cal_gradsquare(ua)/2 + Wb*cal_gradsquare(ub)/2 + Wc*cal_gradsquare(uc)/2

    G = bulk_term + interface_term
    return jnp.mean(G, axis=(1,2))


#draw the concentration profile
def plot_RGB(ua, ub, uc, name, file_path, savefig, show):
    u = jnp.stack((ua, ub, uc), axis=2)
    u = jnp.clip(u, 0, 1)
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
    
    if savefig == True:
        save_path = os.path.join(file_path, f"{name}.pdf")
        plt.savefig(save_path)
    if show == True: plt.show()


# Draw the Energy Pathway on the string
def plot_EnergyPath(F, image_num, file_path, savefig, show):
    plt.figure()
    plt.plot(jnp.arange(image_num)+1, F)
    plt.title('Energy along the string')
    plt.tight_layout()
    if savefig == True:
        save_path = os.path.join(file_path, f"Energy_along_string.pdf")
        plt.savefig(save_path)
    if show == True: plt.show()


### MAIN
ua, ub, uc = initialize(ratioAB=0.5)
#sum = ua + ub + uc
#print(jnp.mean(sum - 1, axis=(1,2)))
#print(jnp.all(ua + ub + uc == 1.0)) 
lp = LineProfiler()
func = lp(main_loop)
ua, ub, uc, err = func(ua, ub, uc, h, ratioAB=0.5)

F = cal_energy(ua, ub, uc)
critical_index = jnp.argmax(F)
barrier = F[critical_index] - F[0]

### SAVE DATA & PLOT
if args.SAVE_CONCENTRATION_DATA: np.save(os.path.join(file_path, "concentration.npy"), np.array(jnp.stack([ua, ub, uc]))) #save: ua， ub， uc，shape=[3, 50, 256, 256]
if args.SAVE_ENERGY_DATA: np.save(os.path.join(file_path, "free-energy.npy"), np.array(F))
if args.PLOT_CONCENTRATION_RGB: 
    plot_RGB(ua[0, : ,:], ub[0, : ,:], uc[0, : ,:], name="Initial Concentration", file_path=file_path, savefig=True, show=False)
    plot_RGB(ua[critical_index, : ,:], ub[critical_index, : ,:], uc[critical_index, : ,:], name="Critical Concentration", file_path=file_path, savefig=True, show=False)
    plot_RGB(ua[-1, :, :], ub[-1, :, :], uc[-1, :, :], name="Final Concentration", file_path=file_path, savefig=True, show=False)
if args.PLOT_ENERGY: plot_EnergyPath(F, image_num, file_path, savefig=True, show=False)

### PRINT LOG TO data.log
with open(os.path.join(file_path, "data.log"), "w") as f1:
    with redirect_stdout(f1):
        print(f"delta : {delta}")
        print(f"gamma[AB, BC, CA] : {(gamma_ab, gamma_bc, gamma_ca)}")
        print(f"A[a, b ,c] : {A}")
        print(f"experimental GAMMA: [{jnp.sqrt(2)*Lambda*(Aa + Ab + 2*C)/6}, {jnp.sqrt(2)*Lambda*(Ac + Ab + 2*C)/6}, {jnp.sqrt(2)*Lambda*(Aa + Ac + 2*C)/6}]")
        print(f"h : {h}")
        print(f"mean_uc : {mean_uc}")
        print(f"N : {N}\n")

        print("------------------------------------------------------\n")
        print(f"err={err}")
        print(f"critical image: {critical_index+1}")
        print(f"Energy barrier: {barrier}")
        #print(f"critical radius: {critical_radius(uc[critical_index, :, :])}")
        print("------------------------------------------------------\n")

        lp.print_stats()
