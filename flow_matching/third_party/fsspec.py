from multiprocessing.pool import ThreadPool

import fsspec


def copy(src_uri: str, dst_uri: str):
    a = fsspec.get_mapper(src_uri)
    b = fsspec.get_mapper(dst_uri)

    def f(k):
        b[k] = a[k]

    with ThreadPool(8) as p:
        for _ in p.imap_unordered(f, a.keys()):
            pass
