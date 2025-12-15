import torch
import lpips
from torchvision import transforms

def calculate_lpips(img1, img2, crop_border, input_order="HWC", model_type='alex'):
    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    model = lpips.LPIPS(net=model_type)  # 加载指定的模型
    if type(img1) == torch.Tensor:
        if len(img1.shape) == 3:
            img1 = img1.unsqueeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1,2,0)
    if type(img2) == torch.Tensor:
        if len(img2.shape) == 3:
            img2 = img2.unsqueeze(0)



    transform = transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1, 1]
    ])

    trans = transforms.Compose([
        transforms.ToTensor()
    ])

    img1 = trans(img1)
    img2 = trans(img2)

    if img1.max() > 1:
        img1 = transform(img1).unsqueeze(0)
        img2 = transform(img2).unsqueeze(0)


    with torch.no_grad():
        lpips_distance = model(img1, img2)

    return lpips_distance.item()
