"""
    Copyright (c) 2020 Ansys Inc. """

######## IMPORTS ########
# General purpose imports
import os,sys
import numpy as np
import scipy as sp
import json
from lumjson import LumEncoder, LumDecoder
sys.path.append("C:\\Program Files\\Lumerical\\v202\\api\\python\\")

# Optimization specific imports
from lumopt.utilities.load_lumerical_scripts import load_from_lsf
from lumopt.utilities.wavelengths import Wavelengths
from lumopt.geometries.parameterized_geometry import ParameterizedGeometry
from lumopt.geometries.polygon import FunctionDefinedPolygon
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.optimizers.generic_optimizers import ScipyOptimizers
from lumopt.optimization import Optimization
from lumopt.utilities.materials import Material

import lumapi

cur_path = os.path.dirname(os.path.realpath(__file__))


# Optimization global parameters
lambda_c = 1.55e-6 
bandwidth_in_nm = 0     #< Only optimize for center frequency of 1550nm
F0 = 0.95
height = 220e-9
etch_depth = 80e-9
y0 = 0
x_begin = -5.1e-6
x_end = 22e-6
n_grates = 25

indexSi = 3.47668
indexSiO2 = 1.44401

params_file = "pid_grating_coupler_initial_params.json"
params_final = "pid_apod_final.json"
base_sim_2d = "pid_grating_coupler_2D_TE_base.fsp"
base_script_2d = 'pid_grating_coupler_2D_TE_base.lsf'
sim_2d_final = "pid_grating_coupler_2D_apodized_final.fsp"

def apodized_grating(params, fdtd, update_only = False):
    verts = grating_params_pos(params)
    
    if not update_only:
        fdtd.addpoly()
        fdtd.set("name", "grating_poly")
    fdtd.setnamed("grating_poly", "vertices", verts)
    fdtd.setnamed("grating_poly", "x", 0)
    fdtd.setnamed("grating_poly", "y", y0)
    fdtd.setnamed("grating_poly", "index", indexSi)

def grating_params_pos(params):
    y3 = y0+height
    y1 = y3-etch_depth

    x_start = params[0]*1e-6  #< First parameter is the starting position
    R  = params[1]*1e6        #< second parameter (unit is 1/um)
    a  = params[2]            #< Third parameter (dim-less)
    b  = params[3]            #< Fourth parameter (dim-less)

    x0 = x_start
  
    verts = np.array( [[x_begin,y0],[x_begin,y3],[x0,y3],[x0,y1]] )       

    ## Iterate over all but the last tooth
    for i in range(n_grates-1):
        F = F0-R*(x0-x_start)
        Lambda = lambda_c / (a+F*b)
        x1 = x0 + (1-F)*Lambda    #< Width of the etched region
        x2 = x0 + Lambda          #< Rest of cell
        verts = np.concatenate((verts,np.array([[x1,y1],[x1,y3],[x2,y3],[x2,y1]])),axis=0)
        x0 = x2

    ## Last tooth is special
    F = F0-R*(x0-x_start)
    Lambda = lambda_c / (a+F*b)
    x1 = x0 + (1-F)*Lambda        #< Width of the etched region
    verts = np.concatenate((verts,np.array([[x1,y1],[x1,y3],[x_end,y3],[x_end,y0]])),axis=0) 

    return verts


def runGratingOptimization(bandwidth_in_nm, etch_depth, n_grates, params, working_dir):

    bounds = [(-4,3),      #< Starting position (in um)
              (0,0.05),    #< Scaling parameter R
              (1.5,3),     #< Parameter a
              (0,2)]       #< Parameter b


    # geometry = ParameterizedGeometry(func = apodized_grating, 
    #                                   initial_params = params, 
    #                                   bounds = bounds, 
    #                                   dx = 1e-5)
    
    geometry = FunctionDefinedPolygon(func = grating_params_pos, 
                                      initial_params = params, 
                                      bounds = bounds, 
                                      z = 0.0, 
                                      depth = 110e-9, 
                                      eps_out = indexSiO2 ** 2, 
                                      eps_in = indexSi ** 2, 
                                      edge_precision = 5, 
                                      dx = 1e-5)

    ######## DEFINE FIGURE OF MERIT ########
    fom = ModeMatch(monitor_name = 'fom', 
                    mode_number = 1, 
                    direction = 'Backward', 
                    multi_freq_src = False,
                    target_T_fwd = lambda wl: np.ones(wl.size), 
                    norm_p = 1,
                    target_fom = 0.0)

    ######## DEFINE OPTIMIZATION ALGORITHM ########
    optimizer = ScipyOptimizers(max_iter = 50, 
                                method = 'L-BFGS-B', 
                                scaling_factor = 1, 
                                pgtol = 1e-6, 
                                ftol = 1e-7,
                                scale_initial_gradient_to = 0.0,
                                penalty_fun = None,
                                penalty_jac = None)

    ######## DEFINE BASE SIMULATION ########
    base_script = load_from_lsf(os.path.join(cur_path, base_script_2d))

    ######## PUT EVERYTHING TOGETHER ########
    lambda_start = lambda_c*1e9 - bandwidth_in_nm/2
    lambda_end   = lambda_c*1e9 + bandwidth_in_nm/2
    lambda_pts   = int(bandwidth_in_nm/10)+1
    wavelengths = Wavelengths(start = lambda_start*1e-9, stop = lambda_end*1e-9, points = lambda_pts)
    opt = Optimization(base_script = base_script, 
                       wavelengths = wavelengths, 
                       fom = fom, 
                       geometry = geometry, 
                       optimizer = optimizer,
                       use_var_fdtd = False,
                       hide_fdtd_cad = False, 
                       use_deps = True, 
                       plot_history = True,
                       store_all_simulations = True,
                       save_global_index = False,
                       label = None)

    ######## RUN THE OPTIMIZER ########
    result = opt.run(working_dir)
    
    return result


if __name__ == "__main__":   
    with open(os.path.join(cur_path, params_file)) as fh:
        initial_params = json.load(fh, cls=LumDecoder)["initial_params"][0]
        
    # Alternate starting point
    # initial_params = [-2.5, 0.03, 2.4, 0.5369]
 
    working_dir = os.path.join(cur_path,'ApodizedGrating')

    result_2D_apodized = runGratingOptimization( bandwidth_in_nm=bandwidth_in_nm,
                            etch_depth=etch_depth,
                            n_grates = n_grates,
                            params=initial_params,
                            working_dir=working_dir)
    
    opt_params_apod_2D = result_2D_apodized[1]
    
    with open(os.path.join(cur_path, params_final), "w") as fh:
        json.dump({ "initial_params": opt_params_apod_2D }, fh, cls=LumEncoder, indent = 4)
        
    ######## 2-D SIMULATION WITH OPTIMIZED STRUCTURE ########
    with lumapi.FDTD(filename = os.path.join(cur_path, base_sim_2d), hide = True) as fdtd:
        vtx = grating_params_pos(opt_params_apod_2D)
        fdtd.addpoly()
        fdtd.set("vertices", vtx)
        fdtd.set("x", 0)
        fdtd.set("y", 0)
        fdtd.set("index", indexSi)
        fdtd.save(os.path.join(cur_path, sim_2d_final))
