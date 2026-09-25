import struct
from pathlib import Path

def s(x):
    b=x.encode(); return struct.pack('<Q',len(b))+b
# GGUF v3, two tensors, metadata architecture/alignment.
out=bytearray(struct.pack('<IIQQ',0x46554747,3,2,2))
out += s('general.architecture') + struct.pack('<I',8) + s('toy_moe')
out += s('general.alignment') + struct.pack('<I',4) + struct.pack('<I',32)
out += s('blk.0.ffn_up_exps.weight') + struct.pack('<I',3)+struct.pack('<QQQ',32,1,4)+struct.pack('<I',8)+struct.pack('<Q',0)
out += s('blk.0.attn_norm.weight') + struct.pack('<I',2)+struct.pack('<QQ',2,2)+struct.pack('<I',0)+struct.pack('<Q',160)
while len(out)%32: out.append(0)
out += bytes([x%256 for x in range(34*4)]) + bytes([0]) * (160-34*4) + bytes(range(16))
Path('tests/fixtures/toy.gguf').write_bytes(out)
