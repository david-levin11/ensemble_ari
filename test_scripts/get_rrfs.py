from herbie import Herbie
from herbie.toolbox import EasyMap, pc
from herbie import paint

import matplotlib.pyplot as plt

H = Herbie(
    "2025-01-15 12:00",
    model="rrfs",
    fxx=6,
    member="control",
    product="nat",
    domain=None
)

print(H.SOURCES)
print(H.help())
ds = H.xarray("TMP:2 m above ground")
print(ds)