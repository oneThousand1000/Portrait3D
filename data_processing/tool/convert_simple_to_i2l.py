import torch

model = torch.load('pose_resnet_50_256x192.pth.tar')  # load simple
model_save = {'network': {}, 'epoch': 0}
for k,v in model.items():
    save_k = 'module.backbone.' + k
    model_save['network'][save_k] = v.cpu()

torch.save(model_save, 'snapshot_0.pth.tar')
