""" 
 * File:   main_k_eps_parametrized.py
 * Author: Albert Canovas & Yannick Reischl
 *
 * Created on 10.05.2024 
 
 ### File description

 Implements the training model for the standard k-epsilon turbulence model in 2D. 
 It includes the parameterization implementation for the 3 inlets (u_inlet1, u_inlet2, u_inlet3)
 
 """

##### IMPORT EXTERNAL MODULES #####
import torch
import numpy as np
from sympy import Symbol, Eq, sin, cos, Min, Max, Abs, log, exp, tanh

import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_2d import Rectangle, Line, Channel2D,  Polygon
from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
    PointwiseConstraint
)
from modulus.sym.domain.monitor import PointwiseMonitor
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.utils.io.plotter import ValidatorPlotter
from modulus.sym.key import Key
from modulus.sym.node import Node
from modulus.sym.utils.io.vtk import var_to_polyvtk
from modulus.sym.geometry.parameterization import Parameterization, Parameter

##### IMPORT CUSTOM MODULES #####
from custom_k_ep_ls import kEpsilonInit, kEpsilon, kEpsilonLSWF
from ownAuxiliaryModulus import LineGeneric, NormalDotVecMinusInlet
from auxiliaryFunctions import LoadCsvData, AddDataConstraints, NoSlipWallLSWF

@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    """ 
        #### Runs the training model

        Consists of 6 sections:

        ###### Parameter definition

        - The fluid properties and the parameterization is defined.

        ###### Geometry definition

        - The geometry and boundaries of the domain are defined.

        ###### Neural network architecture

        - The structure of the neral network and the PDE to be included are defined.

        ###### Boundary conditions definition

        - The boundary conditions are defined (noslip, inlets, outlet).

        ###### Domain losses

        - The losses are added to the domain, both the data constraints and the PDE losses.

        ###### Validators and inferencers

        - Postprocesing is done with validators and inferencers.

    """
    ########################################################
    ####              PARAMETER DEFINITION              ####
    ########################################################
    # simulation parameters
    rho = 1     # Density [kg/m3]
    nu = 1e-6   # Kinematic viscosity [m2/s]
    resolved_y_start = 0.0018   #Resolved wall distance covered by the wall function (yplus 30) [m]
    u_inlet1,  u_inlet2,  u_inlet3 = Symbol("u_inlet1"),  Symbol("u_inlet2"), Symbol("u_inlet3")    #Inlet velocities


    # Defining the range of inlet velocities [m/s]
    u_inlet1_vec = [1.0,2.0]
    u_inlet2_vec = [1.0,2.0]
    u_inlet3_vec = [1.0,2.0]

    # Define a dictionary containing the parameterization of the model
    u_inlet_ranges = {
        u_inlet1: (u_inlet1_vec[0],u_inlet1_vec[1]),
        u_inlet2: (u_inlet2_vec[0],u_inlet2_vec[1]),
        u_inlet3: (u_inlet3_vec[0],u_inlet3_vec[1]),
        "normal_distance": resolved_y_start

    }
    pr = Parameterization(u_inlet_ranges)

    ########################################################
    ####              GEOMETRY DEFINITION               ####
    ########################################################

    # Define the coordinates of the channel [m]
    channel_width = (0, 15.15)
    channel_length = (0, 60.78)
    point_a = (0, 0)
    point_b = (0, 4.63)
    point_c = (3.83, 5)
    point_d = (3.83, 5.381367)
    point_e = (0, 5.02)
    point_f = (0, 10.16)
    point_g = (4.78, 10.16)
    point_h = (4.78, 10.95)
    point_i = (0, 10.95)
    point_j = (0, 15.15)
    point_k = (6.887475, 15.15)
    point_l = (9.78, 13.48)
    point_m = (60.78, 13.48)
    point_n = (60.78, 2.41)
    point_o = (12.78, 2.41)
    point_p = (12.57, 2.84)
    point_q = (8.363993, 0.789157)

    # Define the inlet and outline geometry
    inlet1 = Line(
        point_i,
        point_j,
        normal=1,
    )
    inlet2 = Line(
        point_e,
        point_f,
        normal=1,
    )
    inlet3 = Line(
        point_a,
        point_b,
        normal=1,
    )
    outlet = Line(
        point_n,
        point_m,
        normal=1,
    )
    # Define the boundary
    line = [point_a,point_b,point_c,point_d,point_e,point_f,point_g,point_h,point_i,point_j,point_k,point_l,point_m,point_n,point_o,point_p,point_q]
    geo = Polygon(line)
    wallbc = LineGeneric(point_b,point_c,normal=1)
    wallde = LineGeneric(point_d,point_e,normal=1)
    wallfg = LineGeneric(point_f,point_g,normal=1)
    wallhi = LineGeneric(point_h,point_i,normal=1)
    walljk = LineGeneric(point_j,point_k,normal=1)
    wallkl = LineGeneric(point_k,point_l,normal=1)
    walllm = LineGeneric(point_l,point_m,normal=1)
    wallno = LineGeneric(point_n,point_o,normal=1)
    wallop = LineGeneric(point_o,point_p,normal=1)
    wallpq = LineGeneric(point_p,point_q,normal=1)
    wallqa = LineGeneric(point_q,point_a,normal=1)

    ########################################################
    ####              NN ARCHITECTURE                   ####
    ########################################################

    # List of nodes (equations) to be on the NN
    eq = kEpsilon(nu=nu, rho=rho) # k-epsilon PDEs
    wf = kEpsilonLSWF(nu=nu, rho=rho)# Wall function

    # Define the NN architecture
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("u_inlet1"), Key("u_inlet2"), Key("u_inlet3")],
        output_keys=[Key("u"), Key("v"),Key("p"),Key("k_star"),Key("ep_star")],
        frequencies=("axis", [i / 2 for i in range(8)]),
        frequencies_params=("axis", [i / 2 for i in range(8)]),
        cfg=cfg.arch.fourier,
    )
    nodes = (
        eq.make_nodes()
        + wf.make_nodes()
        + [Node.from_sympy(Min(log(1 + exp(Symbol("k_star"))) + 1e-4, 20), "k")]
        + [Node.from_sympy(Min(log(1 + exp(Symbol("ep_star"))) + 1e-4, 180), "ep")]
        + [flow_net.make_node(name="flow_network")]
    )

    x, y = Symbol("x"), Symbol("y")

    # make domain
    domain = Domain()
    ########################################################
    ####              BOUNDARY CONDITIONS               ####
    ########################################################
    # Define boundary conditions line by line
    lambda_wall = 10
    wf_pt_bc = NoSlipWallLSWF(nodes,wallbc,lambda_wall,cfg.batch_size.short,pr)
    domain.add_constraint(wf_pt_bc, "WFbc")

    wf_pt_de = NoSlipWallLSWF(nodes,wallde,lambda_wall,cfg.batch_size.short,pr)
    domain.add_constraint(wf_pt_de, "WFde")

    wf_pt_fg = NoSlipWallLSWF(nodes,wallfg,lambda_wall,cfg.batch_size.short,pr)
    domain.add_constraint(wf_pt_fg, "WFfg")

    wf_pt_hi = NoSlipWallLSWF(nodes,wallhi,lambda_wall,cfg.batch_size.short,pr)
    domain.add_constraint(wf_pt_hi, "WFhi")

    wf_pt_jk = NoSlipWallLSWF(nodes,walljk,lambda_wall,cfg.batch_size.short,pr)
    domain.add_constraint(wf_pt_jk, "WFjk")

    wf_pt_kl = NoSlipWallLSWF(nodes,wallkl,lambda_wall,cfg.batch_size.short,pr)
    domain.add_constraint(wf_pt_kl, "WFkl")

    wf_pt_lm = NoSlipWallLSWF(nodes,walllm,lambda_wall,cfg.batch_size.wf_pt,pr)
    domain.add_constraint(wf_pt_lm, "WFlm")

    wf_pt_no = NoSlipWallLSWF(nodes,wallno,lambda_wall,cfg.batch_size.wf_pt,pr)
    domain.add_constraint(wf_pt_no, "WFno")

    wf_pt_op = NoSlipWallLSWF(nodes,wallop,lambda_wall,cfg.batch_size.tiny,pr)
    domain.add_constraint(wf_pt_op, "WFop")

    wf_pt_pq = NoSlipWallLSWF(nodes,wallpq,lambda_wall,cfg.batch_size.short,pr)
    domain.add_constraint(wf_pt_pq, "WFpq")

    wf_pt_qa = NoSlipWallLSWF(nodes,wallqa,lambda_wall,cfg.batch_size.short,pr)
    domain.add_constraint(wf_pt_qa, "WFqa")
 
    #---- Inlet 1 BC
    inlet1 = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet1,
        outvar={"u": u_inlet1},
        lambda_weighting={"u": 1},
        batch_size=cfg.batch_size.inlet,
        parameterization=pr,
    )
    domain.add_constraint(inlet1, "Inlet")

    #---- Inlet 2 BC
    inlet2 = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet2,
        outvar={"u": u_inlet2},
        lambda_weighting={"u": 1},
        batch_size=cfg.batch_size.inlet,
        parameterization=pr,
    )
    domain.add_constraint(inlet2, "Inlet")

    #---- Inlet 3 BC
    inlet3 = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet3,
        outvar={"u": u_inlet3},
        lambda_weighting={"u": 1},
        batch_size=cfg.batch_size.inlet,
        parameterization=pr,
    )
    domain.add_constraint(inlet3, "Inlet")


    #---- Outlet BC
    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet,
        outvar={"p": 0},
        lambda_weighting={"p": 1},
        batch_size=cfg.batch_size.outlet,
        parameterization=pr,
    )
    domain.add_constraint(outlet, "Outlet")

    ########################################################
    ####                  DOMAIN LOSSES                 ####
    ########################################################

    #---- Interior PDE loss
    lambda_function = 1*tanh(20*Symbol("sdf"))# Non-linear function to weight the points further from the wall (sdf is distance from the wall)
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={
            "continuity": 0,
            "momentum_x": 0,
            "momentum_y": 0,
            "k_equation": 0,
            "ep_equation": 0,
        },
        lambda_weighting={
            "continuity": lambda_function ,
            "momentum_x": lambda_function ,
            "momentum_y": lambda_function ,
            "k_equation": lambda_function ,
            "ep_equation": lambda_function ,
        },
        batch_size=cfg.batch_size.interior,
        bounds={x: channel_length, y: channel_width},
        parameterization=pr,
    )
    domain.add_constraint(interior, "Interior")


    #---- Training Data Constraint
    invar = {"x-coordinate": "x", "y-coordinate": "y"} # Input variables to the model
    outvar = {"x-velocity": "u", "y-velocity": "v", "pressure": "p"}# output variables to be trained on
    folder = "/workingDirectory/thesis/pinn-channel-flow/channel2D/Data_keps/Data50"
    domain = AddDataConstraints(folder,domain,nodes,invar,outvar,u_inlet1_vec,u_inlet2_vec,u_inlet3_vec)
    
    ########################################################
    ####            VALIDATORS AND INFERENCERS          ####
    ########################################################

    #---- Add validator 1-OUTSIDE THE RANGE
    data_path = "ValData/Valdata2D_keps_real_dens1-2,00_0,155_1,44.csv"
    outvar = {"x-velocity": "u", "y-velocity": "v", "pressure": "p","turb-diss-rate": "ep", "turb-kinetic-energy":"k"}# output variables to be trained on
    invar_test = LoadCsvData(data_path,invar)
    outvar_test = LoadCsvData(data_path,outvar)

    # Add the inlet parameters to the training range
    invar_test.update({"u_inlet1": np.full_like(invar_test["x"], 2)})
    invar_test.update({"u_inlet2": np.full_like(invar_test["x"], 0.155)})
    invar_test.update({"u_inlet3": np.full_like(invar_test["x"], 1.44)})
    validator = PointwiseValidator(
        nodes=nodes,
        invar=invar_test,
        true_outvar=outvar_test,
        batch_size=1024,
        plotter=ValidatorPlotter(),
    )
    domain.add_validator(validator,"validator_outside")

    #---- Add validator 2-INSIDE THE RANGE on data
    data_path = "ValData/Valdata50_2D_keps_real_dens1-1,25_1,97_1,57.csv"

    invar_test = LoadCsvData(data_path,invar)
    outvar_test = LoadCsvData(data_path,outvar)

    # Add the inlet parameters to the training range
    invar_test.update({"u_inlet1": np.full_like(invar_test["x"], 1.25)})
    invar_test.update({"u_inlet2": np.full_like(invar_test["x"], 1.97)})
    invar_test.update({"u_inlet3": np.full_like(invar_test["x"], 1.57)})
    validator = PointwiseValidator(
        nodes=nodes,
        invar=invar_test,
        true_outvar=outvar_test,
        batch_size=1024,
        plotter=ValidatorPlotter(),
    )
    domain.add_validator(validator,"validator_inside")


    #---- Add validator 3-INSIDE THE RANGE without data
    data_path = "ValData/Valdata2D_keps_real_dens1-1,50_1,50_1,50.csv"

    invar_test = LoadCsvData(data_path,invar)
    outvar_test = LoadCsvData(data_path,outvar)

    invar_test.update({"u_inlet1": np.full_like(invar_test["x"], 1.5)})
    invar_test.update({"u_inlet2": np.full_like(invar_test["x"], 1.5)})
    invar_test.update({"u_inlet3": np.full_like(invar_test["x"], 1.5)})
    validator = PointwiseValidator(
        nodes=nodes,
        invar=invar_test,
        true_outvar=outvar_test,
        batch_size=1024,
        plotter=ValidatorPlotter(),
    )
    domain.add_validator(validator,"validator_inside_nodata")

####################################################################################################
####################################################################################################

   #---- Add validator rake
    data_path = "ValData/ValdataRake_2D_keps_real_dens1-1,5_1,5_0,5.csv"
    invar_test = LoadCsvData(data_path,invar)
    outvar_test = LoadCsvData(data_path,outvar)
    invar_test.update({"u_inlet1": np.full_like(invar_test["x"], 1.5)})
    invar_test.update({"u_inlet2": np.full_like(invar_test["x"], 1.5)})
    invar_test.update({"u_inlet3": np.full_like(invar_test["x"], 0.5)})
    validator = PointwiseValidator(
        nodes=nodes,
        invar=invar_test,
        true_outvar=outvar_test,
        batch_size=1024,
        plotter=ValidatorPlotter(),
    )
    domain.add_validator(validator,"validator_inside_rake0,5")

    data_path = "ValData/ValdataRake_2D_keps_real_dens1-1,5_1,5_0,9.csv"
    invar_test = LoadCsvData(data_path,invar)
    outvar_test = LoadCsvData(data_path,outvar)
    invar_test.update({"u_inlet1": np.full_like(invar_test["x"], 1.5)})
    invar_test.update({"u_inlet2": np.full_like(invar_test["x"], 1.5)})
    invar_test.update({"u_inlet3": np.full_like(invar_test["x"], 0.9)})
    validator = PointwiseValidator(
        nodes=nodes,
        invar=invar_test,
        true_outvar=outvar_test,
        batch_size=1024,
        plotter=ValidatorPlotter(),
    )
    domain.add_validator(validator,"validator_inside_rake0,9")

    data_path = "ValData/ValdataRake_2D_keps_real_dens1-1,5_1,5_1,0.csv"
    invar_test = LoadCsvData(data_path,invar)
    outvar_test = LoadCsvData(data_path,outvar)
    invar_test.update({"u_inlet1": np.full_like(invar_test["x"], 1.5)})
    invar_test.update({"u_inlet2": np.full_like(invar_test["x"], 1.5)})
    invar_test.update({"u_inlet3": np.full_like(invar_test["x"], 1.0)})
    validator = PointwiseValidator(
        nodes=nodes,
        invar=invar_test,
        true_outvar=outvar_test,
        batch_size=1024,
        plotter=ValidatorPlotter(),
    )
    domain.add_validator(validator,"validator_inside_rake1,0")

    data_path = "ValData/ValdataRake_2D_keps_real_dens1-1,5_1,5_1,1.csv"
    invar_test = LoadCsvData(data_path,invar)
    outvar_test = LoadCsvData(data_path,outvar)
    invar_test.update({"u_inlet1": np.full_like(invar_test["x"], 1.5)})
    invar_test.update({"u_inlet2": np.full_like(invar_test["x"], 1.5)})
    invar_test.update({"u_inlet3": np.full_like(invar_test["x"], 1.1)})
    validator = PointwiseValidator(
        nodes=nodes,
        invar=invar_test,
        true_outvar=outvar_test,
        batch_size=1024,
        plotter=ValidatorPlotter(),
    )
    domain.add_validator(validator,"validator_inside_rake1,1")

    data_path = "ValData/ValdataRake_2D_keps_real_dens1-1,5_1,5_1,9.csv"
    invar_test = LoadCsvData(data_path,invar)
    outvar_test = LoadCsvData(data_path,outvar)
    invar_test.update({"u_inlet1": np.full_like(invar_test["x"], 1.5)})
    invar_test.update({"u_inlet2": np.full_like(invar_test["x"], 1.5)})
    invar_test.update({"u_inlet3": np.full_like(invar_test["x"], 1.9)})
    validator = PointwiseValidator(
        nodes=nodes,
        invar=invar_test,
        true_outvar=outvar_test,
        batch_size=1024,
        plotter=ValidatorPlotter(),
    )
    domain.add_validator(validator,"validator_inside_rake1,9")

    data_path = "ValData/ValdataRake_2D_keps_real_dens1-1,5_1,5_2,0.csv"
    invar_test = LoadCsvData(data_path,invar)
    outvar_test = LoadCsvData(data_path,outvar)
    invar_test.update({"u_inlet1": np.full_like(invar_test["x"], 1.5)})
    invar_test.update({"u_inlet2": np.full_like(invar_test["x"], 1.5)})
    invar_test.update({"u_inlet3": np.full_like(invar_test["x"], 2.0)})
    validator = PointwiseValidator(
        nodes=nodes,
        invar=invar_test,
        true_outvar=outvar_test,
        batch_size=1024,
        plotter=ValidatorPlotter(),
    )
    domain.add_validator(validator,"validator_inside_rake2,0")

    data_path = "ValData/ValdataRake_2D_keps_real_dens1-1,5_1,5_2,1.csv"
    invar_test = LoadCsvData(data_path,invar)
    outvar_test = LoadCsvData(data_path,outvar)
    invar_test.update({"u_inlet1": np.full_like(invar_test["x"], 1.5)})
    invar_test.update({"u_inlet2": np.full_like(invar_test["x"], 1.5)})
    invar_test.update({"u_inlet3": np.full_like(invar_test["x"], 2.1)})
    validator = PointwiseValidator(
        nodes=nodes,
        invar=invar_test,
        true_outvar=outvar_test,
        batch_size=1024,
        plotter=ValidatorPlotter(),
    )
    domain.add_validator(validator,"validator_inside_rake2,1")

    data_path = "ValData/ValdataRake_2D_keps_real_dens1-1,5_1,5_2,5.csv"
    invar_test = LoadCsvData(data_path,invar)
    outvar_test = LoadCsvData(data_path,outvar)
    invar_test.update({"u_inlet1": np.full_like(invar_test["x"], 1.5)})
    invar_test.update({"u_inlet2": np.full_like(invar_test["x"], 1.5)})
    invar_test.update({"u_inlet3": np.full_like(invar_test["x"], 2.5)})
    validator = PointwiseValidator(
        nodes=nodes,
        invar=invar_test,
        true_outvar=outvar_test,
        batch_size=1024,
        plotter=ValidatorPlotter(),
    )
    domain.add_validator(validator,"validator_inside_rake2,5")

    data_path = "ValData/ValdataRand_2D_keps_real_dens1-0,9_0,9_2,1.csv"
    invar_test = LoadCsvData(data_path,invar)
    outvar_test = LoadCsvData(data_path,outvar)
    invar_test.update({"u_inlet1": np.full_like(invar_test["x"], 0.9)})
    invar_test.update({"u_inlet2": np.full_like(invar_test["x"], 0.9)})
    invar_test.update({"u_inlet3": np.full_like(invar_test["x"], 2.1)})
    validator = PointwiseValidator(
        nodes=nodes,
        invar=invar_test,
        true_outvar=outvar_test,
        batch_size=1024,
        plotter=ValidatorPlotter(),
    )
    domain.add_validator(validator,"validator_inside_randout")

    data_path = "ValData/ValdataRand_2D_keps_real_dens1-1,1_1,1_1,9.csv"
    invar_test = LoadCsvData(data_path,invar)
    outvar_test = LoadCsvData(data_path,outvar)
    invar_test.update({"u_inlet1": np.full_like(invar_test["x"], 1.1)})
    invar_test.update({"u_inlet2": np.full_like(invar_test["x"], 1.1)})
    invar_test.update({"u_inlet3": np.full_like(invar_test["x"], 1.9)})
    validator = PointwiseValidator(
        nodes=nodes,
        invar=invar_test,
        true_outvar=outvar_test,
        batch_size=1024,
        plotter=ValidatorPlotter(),
    )
    domain.add_validator(validator,"validator_inside_randin")

    data_path = "ValData/ValdataRand_2D_keps_real_dens1-1,0_1,0_2,0.csv"
    invar_test = LoadCsvData(data_path,invar)
    outvar_test = LoadCsvData(data_path,outvar)
    invar_test.update({"u_inlet1": np.full_like(invar_test["x"], 1.0)})
    invar_test.update({"u_inlet2": np.full_like(invar_test["x"], 1.0)})
    invar_test.update({"u_inlet3": np.full_like(invar_test["x"], 2.0)})
    validator = PointwiseValidator(
        nodes=nodes,
        invar=invar_test,
        true_outvar=outvar_test,
        batch_size=1024,
        plotter=ValidatorPlotter(),
    )
    domain.add_validator(validator,"validator_inside_randbord")



####################################################################################################
####################################################################################################

    #---- Inference more variables of interest
    inference = PointwiseInferencer(
        nodes=nodes,
        invar=invar_test,
        output_names=["u", "v", "p","k","ep","continuity","momentum_x","momentum_y"],
        requires_grad=True
    )
    domain.add_inferencer(inference, "inf_interior")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
