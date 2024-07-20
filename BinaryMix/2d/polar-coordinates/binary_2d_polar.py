import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
from jax.lax import while_loop
from line_profiler import LineProfiler
from contextlib import redirect_stdout
jax.config.update("jax_enable_x64", True)


### THE ITERATION PARAMETERs
#physical parameters
delta = 0.02 #interface width
gamma  = delta/3 #interf ace energy
A = 3*gamma/delta
Lambda = jnp.sqrt(2)*delta/4
W = Lambda*jnp.sqrt(A)
#W = jnp.power(Lambda, 2)*A
c_uniform = 0.93 #the mean concentration

#numerical parameters
stepmax = 420000 # max number of iterations
tol = 1e-8 # parameter used as stopping criterion
image_num = 100 # number of images along the string
h = 1e-5 # time-step (limited by the ODE step but independent of n1)
N = 5000 #num of nodes
c1, c2 = -1.0, 1.0 #metastable compositions

#meshigird
l = 1.0
dr = l/N
r = jnp.linspace(0, l, N+1)
r_mid = (r[1:] + r[:-1]) / 2

def gradient_NeumannBoundary(N):
    diag = jnp.zeros(N)
    diag1 = jnp.ones(N-1)
    M = np.diag(diag)
    np.fill_diagonal(M[:-1, 1:], diag1)  
    np.fill_diagonal(M[1:, :-1], -diag1) 
    M[0,0] = -1
    M[-1, -1] = 1
    D = jnp.array(M)*(N/2)
    return D.T

def laplace_NeumannBoundary(N):
    diag1 = (2*np.arange(N)) / (2*np.arange(N)+1)
    diag2 = (2*np.arange(N)+2) / (2*np.arange(N)+1)
    M = -2*np.eye(N)
    np.fill_diagonal(M[:-1, 1:], diag2[:-1])  
    np.fill_diagonal(M[1:, :-1], diag1[1:])
    M[-1, -1] = (2-2*N) / (2*N-1)
    L = jnp.array(M) * (N**2)
    return L.T

L = laplace_NeumannBoundary(N)
D = gradient_NeumannBoundary(N)

#the initial string
def endpoints():
    ci = jnp.full(N, c_uniform)
    cf = jnp.full(N, c2)
    def interface(cf):
        r_f = jnp.sqrt((c_uniform - c2) / (c1 - c2)) #because it's two dimensional case 
        print(f"initial interface: {r_f}")
        judge = r_mid <= r_f
        cf = jnp.where(judge, c1, cf)
        return cf
    cf = interface(cf)
    return ci, cf

def initialize_linear(ci, cf):
    coef = jnp.linspace(0, 1, image_num)[:, jnp.newaxis]
    i = jnp.full((image_num, ), 1)[:, jnp.newaxis]
    return coef * (cf - ci) + i * ci

@jax.jit
def generator(c_out):
    c = jnp.full(N, c_out)
    r_f = jnp.sqrt((c_uniform - c_out) / (c1 - c_out)) #because it's two dimensional case
    judge = r_mid<=r_f
    c = jnp.where(judge, c1, c)
    return c

def initialize_growth():
    dc = (c2 - c_uniform) / (image_num - 1)
    c = []
    ci = jnp.full(N, c_uniform)
    c.append(ci)
    for i in range(image_num-1):
        c.append(generator(c_uniform + (i+1)*dc))
    return jnp.array(c)


def initialize_lineargrowth():
    rf = jnp.sqrt((c_uniform - c2) / (c1 - c2))
    c = []
    ci = jnp.full(N, c_uniform)
    c.append(ci)

    @jax.jit
    def generator(R):
        x = (c_uniform - jnp.power(R,2)*c1)/(1 - jnp.power(R,2))
        c = jnp.full(N, x)
        judge = r_mid<R
        c = jnp.where(judge, c1, c)
        return c
    
    for i in range(1, image_num):
        R = rf*i/(image_num-1) 
        c.append(generator(R))
    return jnp.array(c)



#EM & IMEX method
# the energy of each image
def free_energy(c):
    F = A*((c**2 - 1)**2)/4 + (jnp.power(W,2)/2)*(jnp.dot(c, D)**2)
    #+ (jnp.power(W,2)/2)*(0.5*jnp.dot(c**2, L) - c*jnp.dot(c, L))
    F = F *(2*r_mid*jnp.pi*dr) 

    return jnp.sum(F, axis=1)

#bulk_free_energy's gradient
@jax.jit
def bulk_free_energy_term(c):
    return A*(c**3 - c)

matrix1 = jnp.linalg.inv(jnp.eye(N) + h*(W**2)*jnp.dot(L, L))
@jax.jit
def IMEX(c):
    matrix2 = c + h * jnp.dot(bulk_free_energy_term(c), L)
    return jnp.dot(matrix2, matrix1)

M1 = jnp.linalg.inv(jnp.eye(N) + 0.5*h*jnp.power(W,2)*jnp.dot(L,L))
M2 = jnp.eye(N) - 0.5*h*jnp.power(W,2)*jnp.dot(L,L)
@jax.jit
def IMEX_2(c):
    return jnp.dot(jnp.dot(c, M2)+jnp.dot(h*bulk_free_energy_term(c),L), M1)




# string method and main_loop
@jax.jit
def reparameterize(images):
    diff = images[1:, :] - images[:-1, :]
    dist = jnp.sum(diff**2, axis=1)
    l = jnp.sqrt(jnp.concatenate((jnp.array([0.]), dist)))
    cumsum = jnp.cumsum(l) 
    cumsum = cumsum / cumsum[-1]

    @jax.jit
    def interplate(x, i):
        return jnp.interp(jnp.linspace(0, 1, image_num), cumsum, x[:, i])

    return jax.vmap(interplate, in_axes=(None, 0))(images, jnp.arange(N)).T

@jax.jit
def norm(images, images_old):
    err = images - images_old
    #norm_func = lambda x: jnp.linalg.norm(x)
    return jnp.sum(jax.vmap(jnp.linalg.norm, in_axes=0)(err)) / image_num

def main_loop_imex(images):
    for step in range(stepmax):
        images_old = images
        images = IMEX(images)
        images = reparameterize(images)
        #err = norm(images, images_old)
        #if err < tol: 
            #print(f"number of iterations: {step}\n")
            #break
        if step%10000 == 0: 
            err = norm(images, images_old)
            print(f"{step} iteration, done, err = {err}")

    return images

def main_loop_imex2(images):
    images_old = images
    for step in range(stepmax):
        images = IMEX(images)        
        if step%10==0:
            images = reparameterize(images)
            #err = norm(images, images_old)
            if step%10000 == 0: 
                err = norm(images, images_old)
                print(f"{step} iteration, done, err = {err/10}")
            #if err/10 < tol: 
                #print(f"number of iterations: {step}\n")
                #break
            images_old = images
    return images

def critical_radius(critical_c):
    abs_c = jnp.abs(critical_c)
    ind = jnp.argmin(abs_c)
    return r_mid[ind]

def cal_mean_concentration(c):
    ds = 2*r_mid*jnp.pi*dr
    mean_c = ds*c
    S = jnp.pi*jnp.power(l, 2)
    return jnp.sum(mean_c)/S

if __name__=="__main__":
    jax.config.update("jax_enable_x64", True)
    method = "IMEX"

    #jax.profiler.start_trace(f"result-{method}/result-c_uniform-{c_uniform}/data-{c_uniform}")
    #ci, cf = endpoints()
    #c = initialize_growth()
    #c = initialize_linear(ci, cf)
    c = initialize_lineargrowth()
    print(jnp.array([float(cal_mean_concentration(i)) for i in c]))
    
    lp = LineProfiler()
    func = lp(main_loop_imex2)
    c = func(c)
    np.save(f'result-{method}/result-c_uniform-{c_uniform}/concentration.npy', np.array(c))
    print(jnp.array([float(cal_mean_concentration(i)) for i in c]))
 

    F = free_energy(c)
    critical_index = jnp.argmax(F)
    barrier = F[critical_index] - F[0]


    #iteration with line_profiler
    with open(f"result-{method}/result-c_uniform-{c_uniform}/data-{c_uniform}", "w") as f1:
        with redirect_stdout(f1):
            print(f"delta : {delta}")
            print(f"gamma : {gamma}")
            print(f"tol : {tol}")
            print(f"h : {h}")
            print(f"nodes num: {N}\n")

            print("------------------------------------------------------\n")
            print(f"critical image: {critical_index+1}")
            print(f"Energy barrier: {barrier}")
            print(f"critical radius: {critical_radius(c[critical_index])}")

            Gibbs_diff = A*(c_uniform - c_uniform**3)
            theo_radius = gamma/(2*Gibbs_diff)
            theo_E_barrier = jnp.pi*jnp.power(gamma,2)/(2*Gibbs_diff)
            print(f"theoretical critical radius: {theo_radius}")
            print(f"theoretical energy barrier: {theo_E_barrier}\n")
            print("------------------------------------------------------\n")

            lp.print_stats()
        #draw the picture

    def concentration_profile(show_evolution, issavefig, show):
        plt.figure(1)
        plt.plot(r_mid, c[0, :], 'r', label="initial state")
        plt.plot(r_mid, c[-1, :], 'b', label="final state")
        if show_evolution == True:
            for i in range(1, N-1):
                plt.plot(r_mid, c[i, :], 'k', linewidth='0.3')
        plt.plot(r_mid, c[critical_index, :], 'g', label="critical state")
        plt.legend(fontsize=12)
        plt.title("Concentration profiles")
        plt.xlim(0,0.5)
        plt.ylim(-1.2, 1.2)
        if issavefig == True:
            plt.savefig(f"result-{method}/result-c_uniform-{c_uniform}/concentration_profile_2D_{method}_{c_uniform}.png")
        if show == True: plt.show()

    def Energy_along_string(issavefig, show):
        plt.figure(2)
        plt.plot(jnp.arange(image_num)+1, F)
        plt.title('Energy along the string')
        plt.tight_layout()
        if issavefig == True:
            plt.savefig(f"result-{method}/result-c_uniform-{c_uniform}/Energy_along_string_2D_{method}_{c_uniform}.png")
        if show == True: plt.show()

    concentration_profile(show_evolution=True, issavefig=True, show=False)
    Energy_along_string(issavefig=True, show=False)

    #jax.profiler.stop_trace()