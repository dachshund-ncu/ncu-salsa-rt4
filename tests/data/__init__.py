import os
import glob
directory = os.path.dirname(__file__)
test_data_filenames = glob.glob(os.path.join(directory, "*.tar.bz2"))