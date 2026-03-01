import glob
import os

directory = os.path.dirname(__file__)
test_data_filenames = sorted(
    glob.glob(os.path.join(directory, "archives", "*.tar.bz2"))
)
target_data_filenames = sorted(
    glob.glob(os.path.join(directory, "target_json_data", "*.json"))
)
