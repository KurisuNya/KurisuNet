def converter(config, *args, **kwargs):
    __import__("pprint").pprint(config)
    __import__("pprint").pprint(args)
    __import__("pprint").pprint(kwargs)
    return config
