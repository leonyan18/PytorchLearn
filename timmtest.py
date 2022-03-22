import timm
import torch

# Metric
# Augmentation

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Using device: {device}')

print(timm.list_models("resnet*"))
model = timm.create_model('convit_tiny', pretrained=True, num_classes=10, in_chans=1)
print(model)
