from torchrl.data import PrioritizedReplayBuffer, ReplayBuffer
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
"""def collate_fn(batch):
    print(type(batch))
    print(len(batch))"""
collate_fn = torch.stack
rb = ReplayBuffer(collate_fn=collate_fn,size=10)
rb.add(TensorDict({"a": torch.randn(3)}, batch_size=[]))
rb.extend(TensorDict({"a": torch.randn(2, 3)}, batch_size=[2]))
print(len(rb))
print(rb.sample(10))
print(rb.sample(2).contiguous())