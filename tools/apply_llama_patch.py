"""Apply PMI's local graph/loader extension to its llama.cpp pin, without a fetch."""
from pathlib import Path
import subprocess
import sys

PIN = '4ebdf2c74acce30883d8e34b7c70b3eb8146f2fe'
root = Path(__file__).resolve().parents[1]
repo = root / 'vendor/llama.cpp'
patch = root / 'patches/llama-pmi-external-tensors.patch'

def git(*args):
    return subprocess.run(['git', '-C', str(repo), *args], text=True, capture_output=True)

head = git('rev-parse', 'HEAD')
if head.returncode or head.stdout.strip() != PIN:
    sys.exit(f'PMI requires llama.cpp at {PIN}; initialize the pinned submodule first.')
if git('apply', '--reverse', '--check', str(patch)).returncode == 0:
    print('PMI graph/loader extension patch already applied')
    sys.exit(0)
check = git('apply', '--check', str(patch))
if check.returncode:
    sys.exit('PMI extension patch conflicts with the vendor checkout; preserve/review local edits.\n' + check.stderr)
result = git('apply', str(patch))
if result.returncode:
    sys.exit('Could not apply PMI extension patch: ' + result.stderr)
print('Applied PMI graph/loader extension patch to pinned llama.cpp')
