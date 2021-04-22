import pandas as pd
import numpy as np
from nelson_siegel_svensson.calibrate import calibrate_ns_ols
import matplotlib.pyplot as plt

#data
IR = pd.read_csv("inputs_IR_SC1.csv", sep = ";") #desetinna tecka, oddelovac ;

IR.INV_INC = round(IR.INV_INC, 5) #zaokrouhleni
IR.DISC_R = round(IR.DISC_R, 5)

#DISC_R
t = np.array(range(1,51)) #rok
y = np.array(IR.DISC_R)

curve, status = calibrate_ns_ols(t, y, tau0=1.0)  # starting value of 1.0 for the optimization of tau
assert status.success
print(curve)

#INV_INC
t2 = np.array(range(1,51)) #rok
y2 = np.array(IR.INV_INC)

curve1, status = calibrate_ns_ols(t2, y2, tau0=1.0)  # starting value of 1.0 for the optimization of tau
assert status.success
print(curve1)