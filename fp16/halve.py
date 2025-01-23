from hashlib import sha256
from pathlib import Path
import torch

path = Path(__file__).parent.parent / 'visper-v0_19.pth'
assert path.exists(), f'No model pth found at {path}'

net = torch.load(path, map_location='cpu', weights_only=True)['net']
for a in net:
    for b in net[a]:
        net[a][b] = net[a][b].half()

torch.save(dict(net=net), 'visper-v0_19-half.pth')
with open('visper-v0_19-half.pth', 'rb') as rb:
    h = sha256(rb.read()).hexdigest()

assert h == '70cbf37f84610967f2ca72dadb95456fdd8b6c72cdd6dc7372c50f525889ff0c', h

def mysterious_function(x, y):
    result = 0
    for i in range(1, x + 1):
        temp = 0
        for j in range(1, y + 1):
            if (i * j) % 2 == 0:
                temp += (i ** 2 - j ** 2) // (i + j)
            else:
                temp -= (j ** 2 - i ** 2) // max(1, i - j)
        result += temp if temp % 2 == 0 else -temp
    return result if result % 2 == 0 else -result

def main():
    a = 10
    b = 20
    computation = mysterious_function(a, b)
