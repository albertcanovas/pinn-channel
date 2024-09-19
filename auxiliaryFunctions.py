""" 
    * File:   auxiliaryFunctions.py
    * Author: Yannick Reischl
    * Created on 10.04.2024

     ### File description

    Custom functions to avoid code repetition.
    Some functions use modulus parts and some do not.
 """
import os
import warnings
import numpy as np
#from os import walk, path

from modulus.sym.hydra import to_absolute_path
from modulus.sym.utils.io import csv_to_dict
from modulus.sym.domain.constraint import PointwiseConstraint, PointwiseBoundaryConstraint


def LoadCsvData(file_path,mapping,scaling=1.0):
    """ 
    LoadCsvData Returns a CSV data file as a numpy dictionary
    :param file_path: Name of the file from the working directory
    :param mapping: Dictionary with the names of the CSV and output dict variables
    :param scaling: Scaling factor for the variables read
    :return: dictionary wth the read variables

    """
    if os.path.exists(to_absolute_path(file_path)):
        dict_var = csv_to_dict(to_absolute_path(file_path), mapping)
        var_numpy = {
            key: value for key, value in dict_var.items() if key in list(mapping.values())
            }
        for name in var_numpy.keys():
            for valor in var_numpy[name]:
                valor /= scaling

    else:
        warnings.warn(
            f"Directory {file_path} does not exist. Will skip adding validators."
            )
    return var_numpy

def MaxMin(file_path,mapping):
    """ 
        MaxMin Returns a maximum and minimum of a vaariable in a csv file
        :param file_path: Name of the file from the working directory
        :param mapping: Dictionary with the names of the CSV and output dict variables
        :return: maximum and minimum

    """
    var = LoadCsvData(file_path,mapping)
    minim = min(min(var.values()))
    maxim = max(max(var.values()))
    return maxim[0], minim[0]

def AddDataConstraints(folder,domain,nodes,invar,outvar,u_inlet1_vec,u_inlet2_vec,u_inlet3_vec):
    """ 
        AddDataConstraints Adds data constraints from all CSV files in a folder to a domain variable. 
        The parameters in the CSV need to be within the specified limits.
        :param folder: Folder where the CSV fiels are located
        :param domain: domain variable to where the constraint are added
        :param invar: Dictionary of the input variable CSV names
        :param outvar: Dictionary of the output variable CSV names
        :param u_inlet1_vec: Array with the lower and upper limits of u_inlet1 ranges
        :param u_inlet2_vec: Array with the lower and upper limits of u_inlet2 ranges
        :param u_inlet3_vec: Array with the lower and upper limits of u_inlet3 ranges
        :return: domain with the added data constraints
    
     """
    for root, dirs, files in os.walk(folder):
        for name in files:
            importvel = name.split("-")[1].split(".")[0]      
            vel_inlet1 = float(importvel.split("_")[0].replace(",","."))    
            vel_inlet2 = float(importvel.split("_")[1].replace(",","."))
            vel_inlet3 = float(importvel.split("_")[2].replace(",","."))   
            if u_inlet1_vec[0] <= vel_inlet1 <= u_inlet1_vec[1] and u_inlet2_vec[0] <= vel_inlet2 <= u_inlet2_vec[1] and u_inlet3_vec[0] <= vel_inlet3 <= u_inlet3_vec[1]:
                data_path = str(os.path.join(root, name))
                training_label = "training_" + importvel
                
                invar_training = LoadCsvData(data_path,invar)
                outvar_training = LoadCsvData(data_path,outvar)

                invar_training.update({"u_inlet1": np.full_like(invar_training["x"], vel_inlet1)})
                invar_training.update({"u_inlet2": np.full_like(invar_training["x"], vel_inlet2)})
                invar_training.update({"u_inlet3": np.full_like(invar_training["x"], vel_inlet3)})

                training = PointwiseConstraint.from_numpy(
                    nodes=nodes,
                    invar=invar_training,
                    outvar=outvar_training,
                    batch_size=1024,
                    lambda_weighting={"u": np.full_like(outvar_training["u"], 10), "v": np.full_like(outvar_training["v"], 10), "p": np.full_like(outvar_training["p"], 10)},
                    #lambda_weighting={"u": np.full_like(outvar_training["u"], 1), "v": np.full_like(outvar_training["v"], 1), "p": np.full_like(outvar_training["p"], 1), "k": np.full_like(outvar_training["k"], 1), "ep": np.full_like(outvar_training["ep"], 1)},
                )

                domain.add_constraint(training, training_label)
    return domain


def NoSlipWallLSWF(nodes,wall,lambda_weighting,batch,pr=None):
    """ 
        NoSlipWall Creates a boundary noslip condition for a LSWF model 
        :param nodes: Nodes structure of the neural network
        :param wall: wall geometry to apply the B.C.
        :param lambda_weighting: weighting of the constraint loss
        :param patch: Number of points to be randomly sampled
        :param pr: Parameterization dictionary
        :return: PointwiseBoundaryConstraint of the no-slip condition
    
     """
    wall_bc = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wall,
        outvar={
            "velocity_wall_normal_wf": 0,
            "velocity_wall_parallel_wf": 0,
            "ep_wf": 0,
            "wall_shear_stress_x_wf": 0,
            "wall_shear_stress_y_wf": 0,
        },
        lambda_weighting={
            "velocity_wall_normal_wf": lambda_weighting,
            "velocity_wall_parallel_wf": lambda_weighting,
            "ep_wf": lambda_weighting,
            "wall_shear_stress_x_wf": lambda_weighting,
            "wall_shear_stress_y_wf": lambda_weighting,
        },
        batch_size=batch,
        parameterization=pr,
    )
    return wall_bc

def NoSlipWall2D(nodes,wall,lambda_weighting,batch,pr=None):
    """ 
        NoSlipWall Creates a boundary noslip condition for u and v 
        :param nodes: Nodes structure of the neural network
        :param wall: wall geometry to apply the B.C.
        :param lambda_weighting: weighting of the constraint loss
        :param patch: Number of points to be randomly sampled
        :param pr: Parameterization dictionary
        :return: PointwiseBoundaryConstraint of the no-slip condition
    
     """
    wall_bc = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wall,
        outvar={
            "u": 0,
            "v": 0,
        },
        lambda_weighting={
            "u": lambda_weighting,
            "v": lambda_weighting,
        },
        batch_size=batch,
        parameterization=pr,
    )
    return wall_bc