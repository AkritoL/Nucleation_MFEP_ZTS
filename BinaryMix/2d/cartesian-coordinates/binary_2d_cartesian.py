import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import jax
from jax import jit, vmap
from line_profiler import LineProfiler
from contextlib import redirect_stdout
jax.config.update("jax_enable_x64", True)


### THE ITERATION PARAMETERs
#physical parameters
delta = 0.02 #interface width
gamma = delta/3 #interface energy
A = 3*gamma/delta
Lambda = jnp.sqrt(2)*delta/4
W = Lambda*jnp.sqrt(A)
c_uniform = 0.945 #the mean concentration
#numerical parameters
stepmax = 500000 # max number of iterations
tol = 1e-8 # parameter used as stopping criterion
image_num = 50 # number of images along the string
h = 1e-6 # time-step (limited by the ODE step but independent of n1)
N = 512 #num of nodes
c1, c2 = -1.0, 1.0 #metastable compositions
#meshigird
L = 1
dr = L/N
r = jnp.linspace(-L/2, L/2, N+1) 
r_mid = (r[1:] + r[:-1]) / 2
#频率空间的网格
k = 2*jnp.pi*jnp.fft.fftfreq(N, d=dr)
kx, ky = jnp.meshgrid(k, k)
k_square = kx**2 + ky**2


#the initial string
def endpoints():
    ci = jnp.full((N, N), c_uniform)
    cf = jnp.full((N, N), c2)
    def interface(cf):
        interface = jnp.sqrt((jnp.power(L,2)*(c2 - c_uniform))/(jnp.pi*(c2 - c1)))
        rx, ry = jnp.meshgrid(r_mid, r_mid)
        r_mid_squared = rx**2 + ry**2
        sphere_check = r_mid_squared <= interface**2
        cf = jnp.where(sphere_check, c1, cf)
        return cf
    cf = interface(cf)
    return ci, cf

def initialize_linear(ci, cf):
    assert ci.shape == cf.shape, "dimension does not match!"
    coef = jnp.linspace(0, 1, image_num)[:, jnp.newaxis, jnp.newaxis]
    i = jnp.full((image_num, ), 1)[:, jnp.newaxis, jnp.newaxis]
    return coef * (cf - ci) + i * ci

def generator(c_out):
    c = jnp.full((N, N), c_out)
    r_f = jnp.sqrt((jnp.power(L,2)*(c_out - c_uniform))/(jnp.pi*(c_out - c1))) #because it's two dimensional case
    rx, ry = jnp.meshgrid(r_mid, r_mid)
    r_mid_squared = rx**2 + ry**2
    sphere_check = r_mid_squared <= r_f**2
    c = jnp.where(sphere_check, c1, c)
    return c

def initialize_growth():
    dc = (c2 - c_uniform) / (image_num - 1)
    c = []
    ci = jnp.full((N, N), c_uniform)
    c.append(ci)
    for i in range(image_num-1):
        c.append(generator(c_uniform + (i+1)*dc))
    return jnp.array(c)



# the energy of each image
@jit
def free_energy(c):
    '''
    def laplace_c(c):
        chat = jnp.fft.fft2(c)
        return jnp.fft.ifft2(-k_square*chat).real
    '''
    ch = jnp.fft.fft2(c)
    px = jnp.fft.ifft2(-kx*1j*ch).real
    py = jnp.fft.ifft2(-ky*1j*ch).real
    F = A*((c**2 - 1)**2)/4 + (jnp.power(W,2)/2)*(px**2 + py**2)
    return jnp.sum(F, axis=(1, 2))*(dr**2)

#bulk_free_energy's gradient,which is chemical potential
@jax.jit
def bulk_free_energy_term(c):
    return A*(c**3 - c)


# string method and main_loop
@jit
def pde_solver_FFT(images):
    imageshat = jit(jnp.fft.fft2)(images)
    inhomohat = jit(jnp.fft.fft2)(bulk_free_energy_term(images))
    imageshat_new = (imageshat - h*k_square*inhomohat) / (W*W*h*(k_square**2) + 1)
    #imageshat_new = (imageshat - h*inhomohat) / (W*W*h*k_square + 1)
    return jit(jnp.fft.ifft2)(imageshat_new).real

@jit
def reparameterize(images):
    diff = images[1:, :, :] - images[:-1, :, :]
    dist = jnp.sum(diff**2, axis=(1, 2))
    l = jnp.sqrt(jnp.concatenate((jnp.array([0.]), dist)))
    cumsum = jnp.cumsum(l) 
    cumsum = cumsum / cumsum[-1]


    @jit
    def interplate(x, i):
        return jit(jnp.interp)(jnp.linspace(0, 1, image_num), cumsum, x[:, i])
    
    return vmap(interplate, in_axes=(None, 0), out_axes=1)(images.reshape((image_num, N**2)), jnp.arange(N**2)).reshape(images.shape)

@jit
def reparameterize_Eweighted(c):
    diff = c[1:, :, :] - c[:-1, :, :]
    dist = jnp.sqrt(jnp.sum(diff**2, axis=(1,2)))
    V = free_energy(c)
    V_mid = (V[1:] + V[:-1])/2

    @jit
    def W(x):
        return x

    dl = jnp.concatenate((jnp.array([0.]), W(V_mid)*dist))
    l = jnp.cumsum(dl)
    l = l/l[-1]

    @jit
    def interplate(x, i):
        return jit(jnp.interp)(jnp.linspace(0, 1, image_num), l, x[:, i])
    
    return vmap(interplate, in_axes=(None, 0), out_axes=1)(c.reshape((image_num, N**2)), jnp.arange(N**2)).reshape(c.shape)

@jit
def norm_func(x):
    return jnp.linalg.norm(x)

@jit
def norm(images, images_old):
    #assert images.shape == images_old.shape, "dimension does not match!"
    err = images - images_old
    return jnp.sum(vmap(norm_func, in_axes=0)(err)) / image_num

dh = (1e-5 - 1e-6)/(stepmax/10000)

def main_loop(images):
    global h
    images_old = images
    for step in range(stepmax):
        #images_old = images
        images = pde_solver_FFT(images)
        if step%10 == 0:
            images = reparameterize(images)
            if step%10000 == 0:
                h += dh 
                err = norm(images, images_old)
                print(f"{step} iteration, done, err = {err}, h {h}")    
            images_old = images
        #if err < tol: break
    return images

def main_loop2(images):
    for step in range(stepmax):
        images_old = images
        images = pde_solver_FFT(images)
        
        images = reparameterize(images)
        if step%10000 == 0:
            err = norm(images, images_old)
            print(f"{step} iteration, done, err = {err}")           #if err < tol: break
    return images

def critical_radius(critical_c):
    '''
    abs_c = jnp.abs(critical_c)
    ind = jnp.argmin(abs_c)
    return r_mid[ind]
    '''
    abs_c = jnp.abs(critical_c)
    ind = jnp.argmin(abs_c)
    index_2d = jnp.unravel_index(ind, critical_c.shape)    
    return jnp.sqrt(r_mid[int(index_2d[0])]**2 + r_mid[int(index_2d[1])]**2)
    

if __name__=="__main__":
    method = "fft"

    #jax.profiler.start_trace(f"result-{method}/result-c_uniform-{c_uniform}/data-{c_uniform}")
    c = initialize_growth()
    #ci, cf = endpoints()
    #c = initialize_linear(ci, cf)

    #c = jnp.array(np.load(f'result-{method}/result-c_uniform-{c_uniform}/concentration.npy'))
    print(jnp.sum(c, axis=(1,2))/(N**2))

    lp = LineProfiler()
    func = lp(main_loop)
    c = func(c)
    print(jnp.sum(c, axis=(1,2))/(N**2))
    np.save(f'result-{method}/result-c_uniform-{c_uniform}/concentration.npy', np.array(c))
    

    F = free_energy(c)
    critical_index = jnp.argmax(F)
    barrier = F[critical_index] - F[0]

    m = int(N/2)
    profile = (c[:, m-1, :] + c[:, m, :] + c[:, :, m-1] + c[:, :, m]) / 4
    #iteration with line_profiler
    with open(f"result-{method}/result-c_uniform-{c_uniform}/data-{c_uniform}", "w") as f1:
        with redirect_stdout(f1):
            print(f"delta : {delta}")
            print(f"gamma : {gamma}")
            print(f"tol : {tol}")
            print(f"h : {h}\n")

            print("------------------------------------------------------\n")
            print(f"critical image: {critical_index+1}")
            print(f"Energy barrier: {barrier}")
            print(f"critical radius: {critical_radius(c[critical_index, :, :])}")

            Gibbs_diff = A*(c_uniform - c_uniform**3)
            theo_radius = gamma/(2*Gibbs_diff)
            theo_E_barrier = jnp.pi*jnp.power(gamma,2)/(2*Gibbs_diff)
            print(f"theoretical critical radius: {theo_radius}")
            print(f"theoretical energy barrier: {theo_E_barrier}\n")
            print("------------------------------------------------------\n")

            lp.print_stats()

    #draw the picture
    def concentration_profile(c, show_evolution, issavefig, show):
        plt.figure(1)
        plt.plot(r_mid, c[0, :], 'r', label="initial state")
        plt.plot(r_mid, c[-1, :], 'b', label="final state")
        if show_evolution == True:
            for i in range(1, N-1):
                plt.plot(r_mid, c[i, :], 'k', linewidth='0.3')
        plt.plot(r_mid, c[critical_index, :], 'g', label="critical state")
        plt.legend(fontsize=6)
        plt.title("Concentration profiles")
        plt.xlim(-0.3,0.3)
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


    concentration_profile(profile, show_evolution=True, issavefig=True, show=False)
    Energy_along_string(issavefig=True, show=False)

    def paint(c, name, issavefig, show):
        plt.figure()
        plt.contourf(r_mid, r_mid, c, cmap='RdBu', levels = np.linspace(-1.2,1.2,200))
        plt.colorbar()
        plt.title(name)
        plt.xticks(jnp.linspace(-0.5, 0.5, 6))
        plt.yticks(jnp.linspace(-0.5, 0.5, 6))
        if issavefig == True:
            plt.savefig(f"result-{method}/result-c_uniform-{c_uniform}/{name}.png")
        if show == True: plt.show()

    paint(c[0, :, :], "Initial state", issavefig=True, show=False)
    paint(c[critical_index, :, :], "Critical state", issavefig=True, show=False)
    paint(c[-1, :, :], "Final state", issavefig=True, show=False)
    #jax.profiler.stop_trace()