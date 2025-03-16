from kurisunet.module import register_converter


@register_converter
def ResizeConverter(width):
    def make_divisible(v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def resize_backbone(args, resize_in_ch=True):
        _in, out, *args = args
        if resize_in_ch:
            return make_divisible(_in * width, 8), make_divisible(out * width, 8), *args
        return _in, make_divisible(out * width, 8), *args

    def resize_out(args):
        _in, *args = args
        return make_divisible(_in * width, 8), *args

    def converter(config):
        for i, layer in enumerate(config["layers"], start=1):
            if i == 1:
                layer[2] = resize_backbone(layer[2], resize_in_ch=False)
            elif i < len(config["layers"]):
                layer[2] = resize_backbone(layer[2])
            elif i == len(config["layers"]):
                layer[2] = resize_out(layer[2])
        return config

    return converter
