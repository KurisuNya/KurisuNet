from kurisunet.module import register_converter
from itertools import pairwise


@register_converter
def BackboneConverter():
    def converter(config, arg_dict):
        dims = arg_dict["dims"]
        if len(dims) < 2:
            raise ValueError("Backbone requires at least two dimensions")

        module = arg_dict["module"]
        in_out_pairs = list(pairwise(dims))
        layers = [[-1, module, [i, o]] for i, o in in_out_pairs]
        config["layers"] = layers
        return config

    return converter
