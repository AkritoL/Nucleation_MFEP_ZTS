from netCDF4 import Dataset
import numpy as np
import sys

#dirRoot = sys.argv[1]; steps = int(sys.argv[2]);

#write to file
dirRoot = "./"
rootgrp = Dataset(dirRoot+'concentration.nc', 'w', format="NETCDF4")

#NOut = 1000 # output frequency, has to be multiples of 100
#fileName = dirRoot + 'c_0.npy'
#c = np.load(fileName)
c = np.load("./concentration.npy")
NCom = c.shape[0]
steps = c.shape[1]
N =  c.shape[2]

#3 spatial dims, 4th is time
x = rootgrp.createDimension("x", N)
y = rootgrp.createDimension("y", N)
z = rootgrp.createDimension("z", N)
time = rootgrp.createDimension("time", steps)

#dimensions --> vars
xs = rootgrp.createVariable("x", np.float32, ("x",))
ys = rootgrp.createVariable("y", np.float32, ("y",))
zs = rootgrp.createVariable("z", np.float32, ("z",))
times = rootgrp.createVariable("time", np.int32, ("time",))
times.units = "seconds"

#up to 5 components
c1 = rootgrp.createVariable('c1', np.float32,('time','x','y','z'))
c2 = rootgrp.createVariable('c2', np.float32,('time','x','y','z'))
if NCom > 2:
	c3 = rootgrp.createVariable('c3', np.float32,('time','x','y','z'))
if NCom > 3:
	c4 = rootgrp.createVariable('c4', np.float32,('time','x','y','z'))
if NCom > 4:
	c5 = rootgrp.createVariable('c5', np.float32,('time','x','y','z'))

xs[:] = np.linspace(0, 1, N)
ys[:] = np.linspace(0, 1, N)
zs[:] = np.linspace(0, 1, N)
times[:] = np.arange(0, steps, 1)

#for i in range(int(steps/NOut)+1):
for i in range(steps):
	#fileName = dirRoot + 'c_' + str(int(i*NOut)) + '.npy'
	#c = np.load(fileName)

	c1[i, :, :, :] = c[0, i, :, :, :]
	c2[i, :, :, :] = c[1, i, :, :, :]
	if NCom > 2:
		c3[i, :, :, :] = c[2, i, :, :, :]
	if NCom > 3:
		c4[i, :, :, :] = c[3, i, :, :, :]
	if NCom > 4:
		c5[i, :, :, :] = c[4, i, :, :, :]

	print ("data conversion for step %d is finished" %(i))

rootgrp.close()
