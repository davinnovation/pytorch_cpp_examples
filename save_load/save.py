import torch
import torchvision.models as models

resnet18 = models.resnet18()

torch.save(resnet18, "resnet18.pt") # can't load in c++

# TorchScript
input_sample = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(resnet18, input_sample)
traced_script_module.save("resnet18_torchscript.pt")

