import oneflow as flow
import numpy as np


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = flow.sqrt(flow.sum(in_feat ** 2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)


def l2_norm(p0, p1, range=255.0):
    return 0.5 * np.mean((p0 / range - p1 / range) ** 2)


def tensor2np(tensor_obj):
    # change dimension of a tensor object into a numpy array
    return tensor_obj[0].cpu().detach().float().numpy().transpose((1, 2, 0))


def tensor2im(image_tensor, imtype=np.uint8, cent=1.0, factor=255.0 / 2.0):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + cent) * factor
    return image_numpy.astype(imtype)


def tensor2tensorlab(image_tensor, to_norm=True, mc_only=False):
    # image tensor to lab tensor
    from skimage import color

    img = tensor2im(image_tensor)
    img_lab = color.rgb2lab(img)
    if mc_only:
        img_lab[:, :, 0] = img_lab[:, :, 0] - 50
    if to_norm and not mc_only:
        img_lab[:, :, 0] = img_lab[:, :, 0] - 50
        img_lab = img_lab / 100.0

    return np2tensor(img_lab)


def dssim(p0, p1, range=255.0):
    from skimage.measure import compare_ssim

    return (1 - compare_ssim(p0, p1, data_range=range, multichannel=True)) / 2.0
