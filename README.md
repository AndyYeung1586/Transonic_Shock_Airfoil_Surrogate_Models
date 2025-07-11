# Transonic Shock Airfoil Surrogate Models

Additional resources for the "Machine Learning Framework for Predicting Shock Locations on Rotorcraft Airfoils" from the AIAA AVIATION 2025

## Prerequisite

- Python 3.9 or later
- joblib
- matplotlib
- numpy
- panda
- sklearn
- torch (I am using v2.7.0)

## Get Started

To get a simple taste of the surrogate models, run `design_airfoil_gui.py`. This will launch a GUI and you can import airfoil coordinates files which will be used as input for the surrogate models. Note that the coordinates files **must** be in inverted Selig format. See `sample/NACA0012.dat` for the necessary airfoil coordinates format. 

In addition to the coordinates files, user can also manually change the Chebyshev polynomial coefficients from the airfoil Shape function to design any airfoil they want. However, doing so may generate airfoils outside of the intended design space and cause the surrogate models to make erroneous prediction. To combat potential misuse of the surrogate models, users can  . Since the GUI is built on Tkinter library, the performance is far from ideal. See the section below to call the surrogate models efficiently without the use of a GUI.

## Training Surrogate Models

### Datasets

The training datasets are included under the `Datasets` directory. File names that begin with `ma` are used to train transonic shock prediction model (TSPM); whereas, file names that begin with `M` are used to train airfoil polar model (APM). In both dataset, the input parameters are `a1` to `a19` which corresponse to the 9 lower surface Chebyshev polynomials, 9 upper surface Chebyshev polynomials, and 1 trailing edge half thickness respectively. Since TSPM are trained on discrete angles, the datasets it accompany also contain angle of attack per entry. 

### Training

Training take place in the `predict_shock.py` and `train_aerodynamics_prediction_model.py` for TSPM and APM respectively. Most of the hyperparamters, network architecture, and training data can be configured in the `main` function of their perspective Python file. 

### Light Weight Usage

To call the surrogate model 
