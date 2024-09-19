# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" 
 * File:   main_NavierStokes_simp.py
 * Author: Albert Canovas & Yannick Reischl
 * Created on 24.04.2024 
 
 ### File description

 Implements the training model for NavierStokes in 2D without parameterization.
 Only observed to work in laminar with Re<150
 
 """
##### IMPORT EXTERNAL MODULES #####
import torch
import numpy as np
from sympy import Symbol, Eq, sin, cos, Min, Max, Abs, log, exp

import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry import Parameterization
from modulus.sym.geometry.primitives_2d import Rectangle, Line, Channel2D,  Polygon
from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.eq.pdes.basic import NormalDotVec
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
    PointwiseConstraint
)
from modulus.sym.domain.monitor import PointwiseMonitor
from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.utils.io.plotter import ValidatorPlotter, InferencerPlotter
from modulus.sym.key import Key
from modulus.sym.node import Node

from modulus.sym.utils.io.vtk import var_to_polyvtk

##### IMPORT CUSTOM MODULES #####
from auxiliaryFunctions import LoadCsvData, NoSlipWall2D
from ownAuxiliaryModulus import LineGeneric, NormalDotVecMinusInlet


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    """ 
        #### Runs the training model

        Consists of 6 sections:

        ###### Parameter definition

        - The fluid properties are defined.

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
    # simulation parameters
    rho = 1 # Density [kg/m3]
    nu = 0.1 # Density [kg/m3]
    x, y = Symbol("x"), Symbol("y")
    u_inlet1 = 1.5    #Inlet velocity of inlet 1 [m/s]
    u_inlet2 = 1.5 #Inlet velocity of inlet 2 [m/s]
    u_inlet3 = 1.5 #Inlet velocity of inlet 3 [m/s]

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
    #Integral lines where the mass flow is checked
    integral_line1 = Line((13,2.41),(13,13.48),normal=1)
    integral_line2 = Line((26,2.41),(26,13.48),normal=1)
    integral_line3 = Line((44,2.41),(44,13.48),normal=1)

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
    eq = NavierStokes(nu=nu, rho=rho,dim=2,time=False)
    normal_dot_vel = NormalDotVecMinusInlet(vec=["u","v"],u_inlet1=u_inlet1,u_inlet2=u_inlet2,u_inlet3=u_inlet3)

    # Define the NN architecture
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v"),Key("p")],
        cfg=cfg.arch.fourier,
    )
    nodes = (
    eq.make_nodes()
    + normal_dot_vel.make_nodes()
    + [flow_net.make_node(name="flow_network")]

    )

    # make domain
    domain = Domain()

    ########################################################
    ####              BOUNDARY CONDITIONS               ####
    ########################################################
    # Point where wall funciton is applied
    lambda_wall = 1
    wf_pt_bc = NoSlipWall2D(nodes,wallbc,lambda_wall,cfg.batch_size.short)
    domain.add_constraint(wf_pt_bc, "WFbc")

    wf_pt_de = NoSlipWall2D(nodes,wallde,lambda_wall,cfg.batch_size.short)
    domain.add_constraint(wf_pt_de, "WFde")

    wf_pt_fg = NoSlipWall2D(nodes,wallfg,lambda_wall,cfg.batch_size.short)
    domain.add_constraint(wf_pt_fg, "WFfg")

    wf_pt_hi = NoSlipWall2D(nodes,wallhi,lambda_wall,cfg.batch_size.short)
    domain.add_constraint(wf_pt_hi, "WFhi")

    wf_pt_jk = NoSlipWall2D(nodes,walljk,lambda_wall,cfg.batch_size.short)
    domain.add_constraint(wf_pt_jk, "WFjk")

    wf_pt_kl = NoSlipWall2D(nodes,wallkl,lambda_wall,cfg.batch_size.short)
    domain.add_constraint(wf_pt_kl, "WFkl")

    wf_pt_lm = NoSlipWall2D(nodes,walllm,lambda_wall,cfg.batch_size.wf_pt)
    domain.add_constraint(wf_pt_lm, "WFlm")

    wf_pt_no = NoSlipWall2D(nodes,wallno,lambda_wall,cfg.batch_size.wf_pt)
    domain.add_constraint(wf_pt_no, "WFno")

    wf_pt_op = NoSlipWall2D(nodes,wallop,lambda_wall,cfg.batch_size.tiny)
    domain.add_constraint(wf_pt_op, "WFop")

    wf_pt_pq = NoSlipWall2D(nodes,wallpq,lambda_wall,cfg.batch_size.short)
    domain.add_constraint(wf_pt_pq, "WFpq")

    wf_pt_qa = NoSlipWall2D(nodes,wallqa,lambda_wall,cfg.batch_size.short)
    domain.add_constraint(wf_pt_qa, "WFqa")


    #---- Inlet 1 BC
    inlet1 = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet1,
        outvar={"u": u_inlet1},
        lambda_weighting={"u": 1},
        batch_size=cfg.batch_size.inlet
    )
    domain.add_constraint(inlet1, "Inlet")

    #---- Inlet 2 BC
    inlet2 = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet2,
        outvar={"u": u_inlet2},
        lambda_weighting={"u": 1},
        batch_size=cfg.batch_size.inlet
    )
    domain.add_constraint(inlet2, "Inlet")

    #---- Inlet 3 BC
    inlet3 = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet3,
        outvar={"u": u_inlet3},
        lambda_weighting={"u": 1},
        batch_size=cfg.batch_size.inlet
    )
    domain.add_constraint(inlet3, "Inlet")


    #---- Outlet BC
    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet,
        outvar={"p": 0},
        lambda_weighting={"p": 1},
        batch_size=cfg.batch_size.outlet
    )
    domain.add_constraint(outlet, "Outlet")


    #---- Integral continuity constraint

    integral_continuity1 = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry = integral_line1,
        outvar ={"normal_dot_vel_minus_inlet":0},
        lambda_weighting = {"normal_dot_vel_minus_inlet":1},
        batch_size=1,
        integral_batch_size = 512,
    )
    domain.add_constraint(integral_continuity1,"integral_continuity1")
    
    integral_continuity2 = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry = integral_line2,
        outvar ={"normal_dot_vel_minus_inlet":0},
        lambda_weighting = {"normal_dot_vel_minus_inlet":1},
        batch_size=1,
        integral_batch_size = 512,
    )
    domain.add_constraint(integral_continuity2,"integral_continuity2")
    
    integral_continuity3 = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry = integral_line3,
        outvar ={"normal_dot_vel_minus_inlet":0},
        lambda_weighting = {"normal_dot_vel_minus_inlet":1},
        batch_size=1,
        integral_batch_size = 512,
    )
    domain.add_constraint(integral_continuity3,"integral_continuity3")

    ########################################################
    ####                  DOMAIN LOSSES                 ####
    ########################################################
    #---- Interior PDE loss
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={
            "continuity": 0,
            "momentum_x": 0,
            "momentum_y": 0,
        },
        lambda_weighting={
            "continuity": 1,
            "momentum_x": 1,
            "momentum_y": 1,
        },
        batch_size=cfg.batch_size.interior,
        bounds={x: channel_length, y: channel_width}
    )
    domain.add_constraint(interior, "Interior")

    #---- Training Data Constraint ----

    data_path = "Data/data2D_re150-1,5_1,5_1,5.csv"
    invar = {"x-coordinate": "x", "y-coordinate": "y"}
    outvar = {"x-velocity": "u", "y-velocity": "v", "pressure": "p"}
    invar_training = LoadCsvData(data_path,invar)
    outvar_training = LoadCsvData(data_path,outvar)
    training = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar=invar_training,
        outvar=outvar_training,
        batch_size=1024,
        lambda_weighting={"u": np.full_like(outvar_training["u"], 0.1), "v": np.full_like(outvar_training["v"], 0.1), "p": np.full_like(outvar_training["p"], 0.1)},
    )
    #domain.add_constraint(training, "training")

    ########################################################
    ####            VALIDATORS AND INFERENCERS          ####
    ########################################################

    #---- Add validator 1-ON THE TRAINING POINT
    validator = PointwiseValidator(
            nodes=nodes,
            invar=invar_training,
            true_outvar=outvar_training,
            batch_size=1024,
            plotter=ValidatorPlotter(),
        )
    domain.add_validator(validator)
  
    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
