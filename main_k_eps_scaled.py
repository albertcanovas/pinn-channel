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
 * File:   main_k_eps_scaled.py
 * Author: Albert Canovas
 * Created on 10.05.2024 
 
 ### File description

 Implements the training model for the standard k-epsilon turbulence model in 2D with scaling.
 
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
from modulus.sym.geometry.parameterization import Parameterization, Parameter
from modulus.sym.utils.io.vtk import var_to_polyvtk
from modulus.sym import quantity
from modulus.sym.eq.non_dim import NonDimensionalizer, Scaler


##### IMPORT CUSTOM MODULES #####
from custom_k_ep_ls import kEpsilonInit, kEpsilon, kEpsilonLSWF
from ownAuxiliaryModulus import LineGeneric
from auxiliaryFunctions import LoadCsvData, MaxMin, NoSlipWallLSWF

@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    """ 
        #### Runs the training model

        Consists of 7 sections:

        ###### Parameter definition

        - The fluid properties are defined.

        ###### Geometry definition

        - The geometry and boundaries of the domain are defined.

        ###### Non-dimensionalisation

        - All variables defined are non-dimensionalised.

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
    rho = quantity(1.0, "kg/(m^3)")
    nu = quantity(1.0e-6, "kg/(m*s)")
    resolved_y_start = quantity(0.0018, "m")  #Resolved wall distance covered by the wall function (yplus 30) [m]
    u_inlet1 = quantity(2.0, "m/s")
    u_inlet2 = quantity(0.155, "m/s")
    u_inlet3 = quantity(1.44, "m/s")
    outlet_p = quantity(0.0, "pa")
  
    ########################################################
    ####              GEOMETRY DEFINITION               ####
    ########################################################

    channel_width = (quantity(0.0, "m"), quantity(15.15, "m"))
    channel_length = (quantity(0.00, "m"), quantity(60.78, "m"))
    point_a = (quantity(0.0, "m"), quantity(0.0, "m"))
    point_b = (quantity(0, "m"), quantity(4.63, "m"))
    point_c = (quantity(3.83, "m"), quantity(5, "m"))
    point_d = (quantity(3.83, "m"), quantity(5.381367, "m"))
    point_e = (quantity(0.0, "m"), quantity(5.02, "m"))
    point_f = (quantity(0, "m"), quantity(10.16, "m"))
    point_g = (quantity(4.78, "m"), quantity(10.16, "m"))
    point_h = (quantity(4.78, "m"), quantity(10.95, "m"))
    point_i = (quantity(0.0, "m"), quantity(10.95, "m"))
    point_j = (quantity(0.0, "m"), quantity(15.15, "m"))
    point_k = (quantity(6.887475, "m"), quantity(15.15, "m"))
    point_l = (quantity(9.78, "m"), quantity(13.48, "m"))
    point_m = (quantity(60.78, "m"), quantity(13.48, "m"))
    point_n = (quantity(60.78, "m"), quantity(2.41, "m"))
    point_o = (quantity(12.78, "m"), quantity(2.41, "m"))
    point_p = (quantity(12.57, "m"), quantity(2.84, "m"))
    point_q = (quantity(8.363993, "m"), quantity(0.789157, "m"))

    ########################################################
    ####              NON-DIMENSIONALISATION            ####
    ########################################################

    #---- Creating scales
    length_scale = quantity(11.07, "m") # Channel height
    velocity_scale = quantity(1.41, "m/s") # Average velocity
    density_scale = rho # Density

    nd = NonDimensionalizer(
        length_scale=length_scale,
        time_scale=length_scale / velocity_scale,
        mass_scale=density_scale * (length_scale**3),
    )
    # Remove scales from the boundary points
    point_a_nd = tuple(map(lambda x: nd.ndim(x), point_a))
    point_b_nd = tuple(map(lambda x: nd.ndim(x), point_b))
    point_c_nd = tuple(map(lambda x: nd.ndim(x), point_c))
    point_d_nd = tuple(map(lambda x: nd.ndim(x), point_d))
    point_e_nd = tuple(map(lambda x: nd.ndim(x), point_e))
    point_f_nd = tuple(map(lambda x: nd.ndim(x), point_f))
    point_g_nd = tuple(map(lambda x: nd.ndim(x), point_g))
    point_h_nd = tuple(map(lambda x: nd.ndim(x), point_h))
    point_i_nd = tuple(map(lambda x: nd.ndim(x), point_i))
    point_j_nd = tuple(map(lambda x: nd.ndim(x), point_j))
    point_k_nd = tuple(map(lambda x: nd.ndim(x), point_k))
    point_l_nd = tuple(map(lambda x: nd.ndim(x), point_l))
    point_m_nd = tuple(map(lambda x: nd.ndim(x), point_m))
    point_n_nd = tuple(map(lambda x: nd.ndim(x), point_n))
    point_o_nd = tuple(map(lambda x: nd.ndim(x), point_o))
    point_p_nd = tuple(map(lambda x: nd.ndim(x), point_p))
    point_q_nd = tuple(map(lambda x: nd.ndim(x), point_q))

    channel_width_nd = tuple(map(lambda x: nd.ndim(x), channel_width))
    channel_length_nd = tuple(map(lambda x: nd.ndim(x), channel_length))

    resolved_y_start_nd = nd.ndim(resolved_y_start)

    # Define the inlet and outline geometry
    inlet1 = Line(
        point_i_nd,
        point_j_nd,
        normal=1,
    )
    inlet2 = Line(
        point_e_nd,
        point_f_nd,
        normal=1,
    )
    inlet3 = Line(
        point_a_nd,
        point_b_nd,
        normal=1,
    )
    outlet = Line(
        point_n_nd,
        point_m_nd,
        normal=1,
    )

    line = [point_a_nd,point_b_nd,point_c_nd,point_d_nd,point_e_nd,point_f_nd,point_g_nd,point_h_nd,point_i_nd,point_j_nd,point_k_nd,point_l_nd,point_m_nd,point_n_nd,point_o_nd,point_p_nd,point_q_nd]
    # Define the boundary
    geo = Polygon(line)
    wallbc = LineGeneric(point_b_nd,point_c_nd,normal=1)
    wallde = LineGeneric(point_d_nd,point_e_nd,normal=1)
    wallfg = LineGeneric(point_f_nd,point_g_nd,normal=1)
    wallhi = LineGeneric(point_h_nd,point_i_nd,normal=1)
    walljk = LineGeneric(point_j_nd,point_k_nd,normal=1)
    wallkl = LineGeneric(point_k_nd,point_l_nd,normal=1)
    walllm = LineGeneric(point_l_nd,point_m_nd,normal=1)
    wallno = LineGeneric(point_n_nd,point_o_nd,normal=1)
    wallop = LineGeneric(point_o_nd,point_p_nd,normal=1)
    wallpq = LineGeneric(point_p_nd,point_q_nd,normal=1)
    wallqa = LineGeneric(point_q_nd,point_a_nd,normal=1)

    pr = {"normal_distance": resolved_y_start_nd}

    ########################################################
    ####              NN ARCHITECTURE                   ####
    ########################################################

    # List of nodes (equations) to be on the NN
    eq = kEpsilon(nu=nd.ndim(nu), rho=nd.ndim(rho))
    wf = kEpsilonLSWF(nu=nd.ndim(nu), rho=nd.ndim(rho))

    # Define the NN architecture
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v")],
        frequencies=("axis", [i / 2 for i in range(8)]),
        frequencies_params=("axis", [i / 2 for i in range(8)]),
        cfg=cfg.arch.fourier,
    )
    p_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("p")],
        frequencies=("axis", [i / 2 for i in range(8)]),
        frequencies_params=("axis", [i / 2 for i in range(8)]),
        cfg=cfg.arch.fourier,
    )
    k_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("k_star")],
        frequencies=("axis", [i / 2 for i in range(8)]),
        frequencies_params=("axis", [i / 2 for i in range(8)]),
        cfg=cfg.arch.fourier,
    )
    ep_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("ep_star")],
        frequencies=("axis", [i / 2 for i in range(8)]),
        frequencies_params=("axis", [i / 2 for i in range(8)]),
        cfg=cfg.arch.fourier,
    )
    nodes = (
        eq.make_nodes()
        + wf.make_nodes()
        + [Node.from_sympy(Min(log(1 + exp(Symbol("k_star"))) + 1e-4, 10.05), "k")]
        + [Node.from_sympy(Min(log(1 + exp(Symbol("ep_star"))) + 1e-4, 711), "ep")]
        + [flow_net.make_node(name="flow_network")]
        + [p_net.make_node(name="p_network")]
        + [k_net.make_node(name="k_network")]
        + [ep_net.make_node(name="ep_network")]
        + Scaler(
            ["u", "v", "p", "k", "ep"],
            ["u_scaled", "v_scaled", "p_scaled", "k_scaled", "ep_scaled"],
            ["m/s", "m/s", "kg/(m*s^2)", "m^2/s^2", "m^2/s^3"],
            nd,
        ).make_node()
    )

    x, y = Symbol("x"), Symbol("y")

    # make domain
    domain = Domain()
    
    ########################################################
    ####              BOUNDARY CONDITIONS               ####
    ########################################################
    # Point where wall funciton is applied

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
        outvar={"u": nd.ndim(u_inlet1)},
        lambda_weighting={"u": 1},
        batch_size=cfg.batch_size.inlet,
    )
    domain.add_constraint(inlet1, "Inlet")

    #---- Inlet 2 BC
    inlet2 = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet2,
        outvar={"u": nd.ndim(u_inlet2)},
        lambda_weighting={"u": 1},
        batch_size=cfg.batch_size.inlet,
    )
    domain.add_constraint(inlet2, "Inlet")

    #---- Inlet 3 BC
    inlet3 = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet3,
        outvar={"u": nd.ndim(u_inlet3)},
        lambda_weighting={"u": 1},
        batch_size=cfg.batch_size.inlet,
    )
    domain.add_constraint(inlet3, "Inlet")


    #---- Outlet BC
    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet,
        outvar={"p": nd.ndim(outlet_p)},
        lambda_weighting={"p": 1},
        batch_size=cfg.batch_size.outlet,
    )
    domain.add_constraint(outlet, "Outlet")

    ########################################################
    ####                  DOMAIN LOSSES                 ####
    ########################################################
    #---- Interior PDE loss
    lambda_function = 1*tanh(20*Symbol("sdf"))
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
        bounds={x: channel_length_nd, y: channel_width_nd},
    )
    domain.add_constraint(interior, "Interior")

    #---- Training Data Constraint
    # Data loader
    data_path = "Data_keps/data2D_keps_real_air-2,00_0,155_1,44.csv"

    invar = {"x-coordinate": "x", "y-coordinate": "y"}
    outvar = {"x-velocity": "u", "y-velocity": "v", "pressure": "p","turb-kinetic-energy": "k","turb-diss-rate": "ep"}
    invar_training = LoadCsvData(data_path,invar,length_scale)
    outvar_training = LoadCsvData(data_path,outvar,velocity_scale)
    training = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar=invar_training,
        outvar=outvar_training,
        batch_size=1024,
        lambda_weighting={"u": np.full_like(outvar_training["u"], cfg.batch_size.lam_u),
                          "v": np.full_like(outvar_training["v"], cfg.batch_size.lam_v),
                          "p": np.full_like(outvar_training["p"], cfg.batch_size.lam_p),
                          "k": np.full_like(outvar_training["k"], cfg.batch_size.lam_k),
                          "ep": np.full_like(outvar_training["ep"], cfg.batch_size.lam_ep)}
    )
    #domain.add_constraint(training, "training")

    ########################################################
    ####            VALIDATORS AND INFERENCERS          ####
    ########################################################
    # Add validator
    validator = PointwiseValidator(
        nodes=nodes,
        invar=invar_training,
        true_outvar=outvar_training,
        batch_size=1024,
        plotter=ValidatorPlotter(),
    )
    domain.add_validator(validator)

        # add inferencer data
    inference = PointwiseInferencer(
        nodes=nodes,
        invar=invar_training,
        output_names=["u", "v", "p","k","ep"],#,"u_a","v_a","p_a","k_a","ep_a"]
        #requires_grad=True
    )
    domain.add_inferencer(inference, "inf_interior")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
