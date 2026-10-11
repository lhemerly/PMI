"""Candidate-side bootstrap; expected outputs/evaluator remain on the host."""
import builtins
import json
from pathlib import Path
import resource
import sys


def main():
    resource.setrlimit(resource.RLIMIT_AS,(512*1048576,512*1048576))
    resource.setrlimit(resource.RLIMIT_CPU,(5,5))
    resource.setrlimit(resource.RLIMIT_FSIZE,(65536,65536))
    resource.setrlimit(resource.RLIMIT_NOFILE,(32,32))
    data=json.loads(sys.stdin.buffer.read(262145))
    names=('sorted','len','min','max','list','tuple','range','enumerate','zip','reversed')
    scope={'__builtins__':{name:getattr(builtins,name) for name in names}}
    source=Path('/candidate/source.py').read_text()
    # The controller validates the source AST before snapshotting it. This worker
    # has no reference answers, controller imports, credentials or host mounts.
    exec(compile(source,'/candidate/source.py','exec'),scope)
    outputs=[scope['merge_intervals'](tuple(tuple(pair) for pair in case)) for case in data['cases']]
    encoded=json.dumps({'outputs':outputs},separators=(',',':'),allow_nan=False).encode()
    if len(encoded)>65536:raise ValueError('candidate output bound')
    sys.stdout.buffer.write(encoded)


if __name__=='__main__':main()
