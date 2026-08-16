import os
import pytest
import time
from ncu_salsa_rt4 import FitsSpectrum, FitsCube
from .data import fits_files_cepa

def test_light_curve_extraction() -> None:
    cube  = FitsCube(
        fits_files=fits_files_cepa
    )
    light_curve = cube.get_light_curve(velocity = -2.3, df=False)
    light_curve_df = cube.get_light_curve(velocity = -2.3, df=True)
    print(len(fits_files_cepa))
    print(len(light_curve))
    assert len(fits_files_cepa) == len(light_curve)
    assert light_curve_df.shape[0] == len(fits_files_cepa)

if __name__ == "__main__":
    test_light_curve_extraction()