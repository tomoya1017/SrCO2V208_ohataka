import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
import numpy as np

order_8site = [2.0149e-05, 0.0012, 8.1877e-05, 0.0017, 0.0003, 3.0546e-05, 0.0010, 0.0001, 0.0001, 7.4924e-05, 0.0006]
order_8site_with32 = [0.3417, 0.3529, 0.3782, 0.3592, 0.3128, 0.0003, 0.0042,0.0017,4.4790e-05,0.0002]
order_16site = [0.0341, 0.0028, 0.3781 ,0.3582, 3.1063e-05 , 0.0099, 0.0018, 7.5329e-05, 1.9240e-05,0.0001 ,0.0024]
order_32site = [0.3417,0.3529,0.3782, 0.3592,0.3128,0.0003,0.0042,0.0017,4.4790e-05, 0.0002]
magnetic_field = []
for i in range(10):
    magnetic_field.append(i/10)


plt.plot(magnetic_field, order_8site_with32, marker="o", linestyle="dashed")
plt.xlabel("H")
plt.ylabel("|S^z|")
plt.legend()
plt.savefig("S^z_8with32site.png")
plt.show()
