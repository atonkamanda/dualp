import torch
import torch.nn as nn

class ShapeRecorderHook:
    def __init__(self):
        self.outputs = []

    def hook(self, module, input, output):
        self.outputs.append((module, output.shape))

    def attach(self, module):
        handles = []
        for m in module.children():
            handles.extend(self.attach(m))
        if not hasattr(module, "children") or len(list(module.children())) == 0:
            handle = module.register_forward_hook(self.hook)
            handles.append(handle)
        return handles

    def remove(self):
        for handle in self.handles:
            handle.remove()

def record_shapes(model):
    hook = ShapeRecorderHook()
    hook.handles = hook.attach(model)
    return hook

# Example usage
model = nn.Sequential(
                        nn.Linear(320, 50),
                        nn.ReLU(),
                        nn.Linear(50, 10),
                        nn.Softmax(dim=1))

hook = record_shapes(model)
_ = model(torch.randn(64, 320))
for module, shape in hook.outputs:
    print(f"Module: {module} - Output Shape: {shape}")
hook.remove()
