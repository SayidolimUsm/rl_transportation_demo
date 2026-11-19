import time, torch
from dqn import QNet

x = torch.randn(32, 5)
m = QNet(hidden=32)
for threads in (1, 2, 4):
    torch.set_num_threads(threads)
    t0 = time.perf_counter()
    for _ in range(1000):
        m(x).sum().backward()
    print(f"threads={threads}: {(time.perf_counter() - t0)*1000:.1f} ms")