import glob
import os

directory = os.path.dirname(__file__)
data_archives = sorted(
    glob.glob(os.path.join(directory, "archives", "*.tar.bz2"))
)

