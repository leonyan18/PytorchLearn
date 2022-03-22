import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils import data

d = {'a': 1, 'b': 2, 'c': 3}
ser = pd.Series(data=d, index=['x', 'y', 'z'])
print(ser)
a = torch.randn(4,4)
torch.nn.init.xavier_normal_(a,gain=1)
print(a)


