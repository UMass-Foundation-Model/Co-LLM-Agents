from pathlib import Path
from pkg_resources import resource_filename
import os

# The path to the data files.
DATA_DIRECTORY: Path = Path(resource_filename(__name__, "data"))
# The path to the list of target objects.
TARGET_OBJECTS_PATH: Path = DATA_DIRECTORY.joinpath("target_objects.csv")
# The path to the list of target object materials.
TARGET_OBJECT_MATERIALS_PATH: Path = DATA_DIRECTORY.joinpath("target_object_materials.txt")
# The path to the list of containers.
CONTAINERS_PATH: Path = DATA_DIRECTORY.joinpath("containers.txt")
# The path to the default config file.
DEFAULT_CONFIG_PATH: Path = DATA_DIRECTORY.joinpath("default_config.ini")
# The path to the user-defined config file.
USER_CONFIG_PATH: Path =  Path(os.path.join(os.getcwd(), "config.ini"))