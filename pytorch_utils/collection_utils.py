def keychain_value_iter(d, key_chain=None, allowed_values=None):
    key_chain = [] if key_chain is None else list(key_chain).copy()

    if not isinstance(d, dict):
        if allowed_values is not None:
            assert isinstance(d, allowed_values), 'Value needs to be of type {}!'.format(
                allowed_values)
        yield key_chain, d
    else:
        for k, v in d.items():
            yield from keychain_value_iter(
                v,
                key_chain + [k],
                allowed_values=allowed_values)