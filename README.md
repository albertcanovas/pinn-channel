# PINN Channel Flow


## Include the data files


All data files should be inside the correct folder, which is not included in the repository due to its size.

The files should be found in a folder called CFD_Data. Include all files inside the channel2D/ folder. The folder structure inside the repository should look like:

### Data folder structure
- pinn-channel/
    - conf/
    - Docs/
    - Data/
    - Data_keps/
        - Data5/
        - Data10/
        - Data20/
        - Data50/
    - ValData/

Inside the /Data/ folder, the training dataset for laminar flow is found. In /Data_keps/ the turbulent simulations are located. Inside each DataXX folder (XX is 5, 10, 20 or 50) the dataset for the corresponnding amount of design points is located. This means that inside /Data_keps/Data5/ only 5 files should be located.

> [!WARNING]
> A wrong folder structure won't allow the code to run properly

The naming of the data files is the following. The file name starts with a description or identifier of the simulation conditions (RANS/laminar or reynolds, fluid etc) followed by a - and the inlet flow velocities separated by an undersore. Note that for the inlet values, a comma is used instead of a point.

`name-X,XX_Y,YY_Z,ZZ.csv`

For a simulation on 2D with a KE Realizable model with density=1 and the following parameters u1: 1.5; u2: 1.7; u3: 1.9; it would be

`data2D_keps_real_dens1-1,50_1,70_1,90.csv`



## Running the files

### Input data

Almost all configuration, specially the information realted to the modulus configuration, is located in the `config.yaml` file. In the file, under the type training, the parameters related to the training steps aer located. Under batch_size, the amount of points used for sammpling are located

### Output data

The data generated during training is stored in the pinn-channel-flow/cahnnel2D/outputs/filename/ where filename is the name of the script that is running. Inside the folder network_checkpoint, all the vtp files are generated. The files can be opened in paraview for viewing.

### Running the script

1. Open terminal in pinn-channel-flow/channel2D/ (the terminal must be inside the container).
2. Run `python3 filename.py` where filename is the name of the file.

### Monitoring the training

Tensorboard can be used to monitor the training. Losses and validation errors are shown. To monitor the training either during or after training:
1. Run `tensorboard --logir=.` either in the output folder or any directory that cointains the folder. 
2. Open [http://localhost:6006](http://localhost:6006) in a browser.

## Create documentation

Documentation can be created using the pdoc module. Inside a folder called /Docs the documentation is created in an .html format. The html files can then be opened with a browser to view the documentation in an intuitive way.

1. Open terminal in pinn-channel/Docs (the terminal must be inside the container).
2. Run `pdoc -html -o . ../*.py`
3. Open the created html files in a browser.


