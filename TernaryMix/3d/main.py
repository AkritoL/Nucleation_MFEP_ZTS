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


'''contact the author to get get the full code'''
