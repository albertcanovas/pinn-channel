""" 
    * File:   ownAuxiliaryModulus.py
    * Author: Albert Canovas
    * Created on 10.04.2024

    ### File description

    Classes derived from basic modulus items, that are tailored towards our case
 """

import sys
from operator import mul
from sympy import Symbol, Abs, Max, Min, sqrt, sin, cos, acos, atan2, pi, Heaviside
from functools import reduce

pi = float(pi)
from modulus.sym.geometry.curve import SympyCurve
from modulus.sym.geometry.helper import _sympy_sdf_to_sdf
from modulus.sym.geometry.geometry import Geometry, csg_curve_naming
from modulus.sym.geometry.parameterization import Parameterization, Parameter, Bounds
from modulus.sym.eq.pde import PDE


class LineGeneric(Geometry):
    """
    2D Line that allows to be in any direction of the 2D plane (as opposed to the basic Line class that must be vertical)
    """

    def __init__(self, point_1, point_2, normal=1, parameterization=Parameterization()):
    
        """
            Initializes the line geometry element and calculates the related parameters
            :param point_1: tuple with 2 ints or floats lower bound point of line segment
            :param point_2: tuple with 2 ints or floats upper bound point of line segment
            :param normal: int or float direction of line (+1 for counterclockwise or -1 for clockwise)
            :param parameterization: Parameterization of geometry.
        """
       # calculate bounds
        l = Symbol(csg_curve_naming(0))
        x = Symbol("x")
        y = Symbol("y")

        # curves for each side
        curve_parameterization = Parameterization({l: (0, 1)})
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        dist_x = point_2[0] - point_1[0]
        dist_y = point_2[1] - point_1[1]
        length = sqrt(dist_x * dist_x + dist_y * dist_y)
        norm_x = -dist_y / length * normal
        norm_y = dist_x / length * normal
        line_1 = SympyCurve(
            functions={
                "x": point_1[0] + l * dist_x,
                "y": point_1[1] + l * dist_y,
                "normal_x":norm_x,  # TODO rm 1e-10
                "normal_y": norm_y,
            },
            parameterization=curve_parameterization,
            area=length,
        )
        curves = [line_1]

        # calculate SDF (signed distance function)
        pq_x = x - point_1[0]
        pq_y = y - point_1[1]
        aux_len = dist_y * pq_x - dist_x * pq_y
        sdf = -aux_len/length
        bounds = Bounds(
            {
                Parameter("x"): (point_1[0], point_2[0]),
                Parameter("y"): (point_1[1], point_2[1]),
            },
            parameterization=parameterization,
        )

        # initialize Line
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=2,
            bounds=bounds,
            parameterization=parameterization,
        )


class NormalDotVecMinusInlet(PDE):
    """
        Normal dot velocity minus the flow
        The integration of this variable over a vertical line should give a mass flow of 0. 
    """


    name = "NormalDotVecMinusInlet"

    def __init__(self, vec=["u", "v", "w"],u_inlet1="u_inlet1",u_inlet2="u_inlet2",u_inlet3="u_inlet1"):
        """
            Initializes the PDE
            :param vec: array with the 2 or 3 components of the velocity field
            :param u_inlet1: float or string with the velocity magnitude at the first inlet
            :param u_inlet2: float or string with the velocity magnitude at the second inlet
            :param u_inlet3: float or string with the velocity magnitude at the third inlet
        """
        # normal
        normal = [Symbol("normal_x"), Symbol("normal_y"), Symbol("normal_z")]
        h_inlet3 = 4.63 #height of the first inlet
        h_inlet2 = 5.14 #height of the second inlet
        h_inlet1 = 4.2 #height of the third inlet
        h_channel = 11.07 #height of the channel
        if isinstance(u_inlet1, (int, float, complex)):
            totalflow = (((u_inlet1)*h_inlet1 + (u_inlet2)*h_inlet2 + (u_inlet3)*h_inlet3) / h_channel)#Total mass flow per unit height
        else:
            totalflow = ((Symbol(u_inlet1)*h_inlet1 + Symbol(u_inlet2)*h_inlet2 + Symbol(u_inlet3)*h_inlet3) / h_channel)#Total mass flow per unit height
        # make input variables
        self.equations = {}
        self.equations["normal_dot_vel_minus_inlet"] = 0
        for v, n in zip(vec, normal):
            self.equations["normal_dot_vel_minus_inlet"] += Symbol(v) * n
        self.equations["normal_dot_vel_minus_inlet"] -= totalflow