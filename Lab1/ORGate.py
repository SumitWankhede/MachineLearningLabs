import numpy as np

inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
targets = np.array([[0],[1],[1],[1]])

import pcn_logic_eg

p = pcn_logic_eg.pcn(inputs,targets)
p.pcntrain(inputs, targets, 0.25, 6)

p.confmat(inputs, targets)
