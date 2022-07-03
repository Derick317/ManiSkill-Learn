from mani_skill_learn.utils.data.converter import to_torch, to_np
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import os, time
from mani_skill_learn.env.torch_parallel_runner import TorchWorker as Worker

start_time = time.time()

class DIST():
    def __init__(self):
        self.workers = []
        for i in range(3):
            self.workers.append(Worker(Model, i, 100, 100, device=i))
    
    def prod(self, x: np.ndarray):
        start_time = time.time()
        # for i in range(3):
        #     self.workers[i].call('forward', x)
        for i in range(3):
            self.workers[i].call('id')
        ret = []
        for i in range(3):
            ret_i = self.workers[i].get()
            ret.append(ret_i)
        
        return ret
    
class Model():
    def __init__(self, input, output, device, worker_id=None, *args, **kwargs):
        print(worker_id)
        self.worker_id = worker_id
        self.model = nn.Linear(input, output, device=device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def forward(self, x):
        # print(f"{self.worker_id}: address of x: {hex(id(x))}")
        # print("In time: ", time.time() - start_time)
        tmp_time = time.time()
        x = to_torch(x, dtype='float32', device=next(self.model.parameters()).device, non_blocking=True)
        # print(f"{self.worker_id} changes x")
        ret = self.model(x)
        loss = F.mse_loss(ret, torch.ones_like(ret))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss = to_np(loss)
        print(f"Time for model {self.worker_id}: ", time.time() - tmp_time)
        return loss
    
    def id(self):
        for _ in range(100):
            s = np.linalg.solve(np.random.rand(200, 200), np.ones(200))
        print(self.worker_id)


if __name__ == "__main__":
    # models = [nn.Linear(1000, 1000, device=i) for i in range(3)]
    # x = [to_torch(np.random.rand(1000), dtype='float32', device=i%3, non_blocking=True) for i in range(10)]
    
    # for i in range(3):
    #     for _ in range(4):
    #         s = models[i](x[i])
    # for _ in range(5):
    #     tmp_time = time.time()
    #     s = F.sigmoid(F.tanh(models[0](x[0])))
    #     print(time.time() - tmp_time)
    test = DIST()
    for _ in range(10):
        tmp_time = time.time()
        ret = test.prod(np.random.rand(100))
        print("Time: ", time.time() - tmp_time)
        print(ret)
        
    