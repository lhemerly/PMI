"""Explicit minimal-runtime packaging; no dependency execution or installation."""
import hashlib
import json
from pathlib import Path,PurePosixPath
import shutil
import stat


def build_runtime(destination,inventory,worker="interval",*,authority=None):
    from .qualification_authority import gate_cpu
    gate_cpu(authority,'package',destination=destination,inventory=inventory,worker=worker)
    if worker not in ("interval","qualification"):raise ValueError("unknown fixed worker role")
    destination=Path(destination)
    if not destination.is_absolute() or not isinstance(inventory,dict) or not inventory:raise ValueError('explicit absolute bundle/inventory required')
    prepared=[]
    for target,row in inventory.items():
        relative=PurePosixPath(target)
        if relative.is_absolute() or '..' in relative.parts or '\\' in target or (relative.parts[0] not in ('usr','lib','lib64') and target!='worker.py'):raise ValueError('runtime destination outside whitelist')
        if not isinstance(row,dict) or set(row)!={'source','sha256'}:raise ValueError('pinned runtime source required')
        source=Path(row['source'])
        if not source.is_absolute() or not source.is_file() or not stat.S_ISREG(source.stat().st_mode):raise ValueError('runtime source must be a regular file')
        data=source.read_bytes()
        if hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('runtime source identity changed')
        prepared.append((target,source,data))
    if not {'usr/bin/python3','worker.py'}<=set(inventory):raise ValueError('interpreter/bootstrap inventory missing')
    destination.mkdir(mode=0o700)
    try:
        manifest={}
        for target,source,data in prepared:
            path=destination/target;path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as file:file.write(data)
            path.chmod(0o500 if data.startswith(b'\x7fELF') or target=='usr/bin/python3' else 0o400);manifest[target]=hashlib.sha256(data).hexdigest()
        from .executor import verify_runtime
        # Bind destinations must already exist under the read-only runtime root.
        for name in ('candidate','work','proc','dev','tmp'):(destination/name).mkdir(mode=0o500)
        for path in sorted((p for p in destination.rglob('*') if p.is_dir()),reverse=True):path.chmod(0o500)
        destination.chmod(0o500)
        identity=verify_runtime(destination,manifest,worker=worker)
        return {'root':str(destination),'files':manifest,'runtime_sha256':identity,'primitive_sha256':hashlib.sha256(Path('/usr/bin/bwrap').read_bytes()).hexdigest(),'packaging':'explicit copied regular files; no ldd/interpreter execution'}
    except BaseException:
        for path in destination.rglob('*'):
            if path.is_dir():path.chmod(0o700)
        destination.chmod(0o700);shutil.rmtree(destination);raise
