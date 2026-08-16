import glob
import os

directory = os.path.dirname(os.path.abspath(__file__))
data_archives = sorted(
    glob.glob(os.path.join(directory, "archives", "*.tar.bz2"))
)

fits_files_directory = os.path.join(directory, "fits_files")


fits_files = sorted(
    glob.glob(os.path.join(fits_files_directory, "*.fits"))
)

fits_files_cepa = sorted(
    glob.glob(os.path.join(fits_files_directory, "cepa*.fits"))
)
