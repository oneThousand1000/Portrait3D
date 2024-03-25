import torch
from PIL import Image
from torchvision.transforms import ToPILImage
import glob
import os
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision import transforms, utils
from tqdm import tqdm
import tempfile
import dnnlib
from torch_utils import training_stats
from torch_utils import custom_ops

from torch.utils.data import dataset


class LoadData(dataset.Dataset):

    def __init__(self, base_path):
        super(LoadData, self).__init__()
        #base_path = 'F:/high_quality_3DPortraitGAN/exp/stable-dreamfusion/output/2023-10-28-with-inversion-initialization/samples_new_crop'
        paths = sorted(glob.glob(f'{base_path}/aligned_images/*'))
        os.makedirs(f'{base_path}/mask', exist_ok=True)
        self.paths = paths

    def __getitem__(self,idx):
        image_path =self.paths[idx]
        image = Image.open(image_path)
        # Define the preprocessing transformation
        preprocess = transforms.Compose([
            transforms.ToTensor()
        ])

        # Apply the transformation to the image
        input_tensor = preprocess(image)

        return input_tensor, image_path

    def __len__(self):
        return len(self.paths)


def get_mask(model, batch, cid):
    normalized_batch = transforms.functional.normalize(
        batch, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    output = model(normalized_batch)['out']
    # sem_classes = [
    #     '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    #     'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    #     'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    # ]
    # sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
    # cid = sem_class_to_idx['car']

    normalized_masks = torch.nn.functional.softmax(output, dim=1)

    boolean_car_masks = (normalized_masks.argmax(1) == cid)
    return boolean_car_masks.float()


def get_and_save_mask( device,base_path):
    # data loder
    batch_size = 8
    dataset = torch.utils.data.DataLoader(
        dataset=LoadData(base_path),
        batch_size=batch_size,
        shuffle=False
    )
    for input_tensor, image_paths in tqdm(dataset):
        input_batch = input_tensor.to(device)  # batxh, 3, 256, 256

        # load segmentation net
        seg_net = deeplabv3_resnet101(pretrained=True, progress=False).to(device)
        seg_net.requires_grad_(False)
        seg_net.eval()

        # 15 means human mask
        mask = get_mask(seg_net, input_batch, 15)
        print(mask.shape) # 16, 256, 256

        mask = mask.unsqueeze(1) # 16, 1, 256, 256

        for i in range(mask.shape[0]):
            # Squeeze the tensor to remove unnecessary dimensions and convert to PIL Image
            mask0 = mask[i:i+1]
            mask_squeezed = torch.squeeze(mask0)
            mask_image = ToPILImage()(mask_squeezed)
            image_path = image_paths[i]
            # Save as PNG
            mask_path = image_path.replace('aligned_images', 'mask')
            # /home/zjucadjin/dataset/pexels-256-new/0000000053/0000053992.png
            # mask_dir = mask_path[:-len('/0000053992.png')]
            # os.makedirs(mask_dir, exist_ok=True)
            mask_image.save(mask_path)


def run(rank,base_path):
    rank = rank
    device = torch.device('cuda', rank)
    get_and_save_mask(device,base_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--mask_path', type=str, required=True)
    run(0, parser.parse_args().base_path)