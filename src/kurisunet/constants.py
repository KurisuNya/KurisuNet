PYTHON_SUFFIX = [".py"]
CONFIG_SUFFIX = [".yaml", ".yml"]

AUTO_REGISTER_KEY = "auto_register"
GLOBAL_IMPORTS_KEY = "global_imports"
GLOBAL_EXEC_KEY = "global_exec"
GLOBAL_VARS_KEY = "global_vars"

IMPORTS_KEY = "imports"
ARGS_KEY = "args"
PRE_EXEC_KEY = "pre_exec"
BUFFERS_KEY = "buffers"
PARAMS_KEY = "params"
VARS_KEY = "vars"

CONVERTERS_KEY = "converters"
LAYERS_KEY = "layers"
POST_EXEC_KEY = "post_exec"

LAYER_START_INDEX = 1  # INFO: should be positive(exclude 0)
MODULE_START_INDEX = 1
DROP_FROM = "drop"
ALL_FROM = "all"

STR_PREFIX = "~"
OUTPUT_MODULE_NAME = "Output"
BUILD_IN_IMPORT = [
    "import torch",
    "import torch.nn as nn",
]
