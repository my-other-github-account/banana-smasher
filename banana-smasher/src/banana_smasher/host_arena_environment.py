"""Opt-in environment for a NEW glibc child process, never a live allocator fix.

Pass the returned mapping to process creation before Python/Torch startup.
The experimentally exercised arena count is two. No numerical options, process
caps, host reserves or production group-size authority are changed here.
"""

def host_arena_environment(environment, *, enabled=False):
    if type(enabled) is not bool:
        raise ValueError('enabled must be bool')
    result = dict(environment)
    if enabled:
        result['MALLOC_ARENA_MAX'] = '2'
    return result
