"""Experimental bounded source read/hash overlap; no cross-process trust."""
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor


def sha256_prefetched(path):
    """Hash every byte in order with at most one 8MiB read ahead.

    Caller retains original before/after immutable-file identity checks.
    Never substitutes a prior digest or a different digest algorithm.
    """
    digest = hashlib.sha256()
    offset = 0
    with path.open('rb') as stream:
        with ThreadPoolExecutor(max_workers=1) as reader:
            pending = reader.submit(stream.read, 8 << 20)
            while True:
                block = pending.result()
                if not block:
                    break
                pending = reader.submit(stream.read, 8 << 20)
                digest.update(block)
                advise = getattr(os, 'posix_fadvise', None)
                dontneed = getattr(os, 'POSIX_FADV_DONTNEED', None)
                if advise is not None and dontneed is not None:
                    try:
                        advise(stream.fileno(), offset, len(block), dontneed)
                    except OSError:
                        pass
                offset += len(block)
    return digest.hexdigest()
