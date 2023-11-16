import numpy as np
import pandas as pd
import os


print(os.getcwd())

df = pd.read_csv(r'data\URBAN-SED_v2.0.0\annotations\train\soundscape_train_unimodal1999.txt')
print(df)