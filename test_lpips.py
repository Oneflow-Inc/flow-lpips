import lpips
import flow_lpips
from PIL import Image

import numpy as np
import oneflow.mock_torch as mock


loss_torch = lpips.LPIPS(net="vgg")
loss_flow = flow_lpips.LPIPS(net="vgg")

pred_image = np.array(Image.open("./imgs/pred.png"))
ref_image = np.array(Image.open("./imgs/ref.png"))

lpips_result = {
    "pytorch": 0.0,
    "oneflow": 0.0,
}


def loss_fn(pred, ref, mock: bool = True):
    import torch

    fn = loss_flow if mock else loss_torch
    return fn(
        torch.from_numpy(pred.transpose(2, 0, 1)).unsqueeze(0),
        torch.from_numpy(ref.transpose(2, 0, 1)).unsqueeze(0),
    )


for mode in ["pytorch", "oneflow"]:
    if mode == "pytorch":
        with mock.disable():
            lpips_result[mode] = loss_fn(pred_image, ref_image, False)
    elif mode == "oneflow":
        with mock.enable():
            lpips_result[mode] = loss_fn(pred_image, ref_image, True)

print(lpips_result["pytorch"].detach().numpy())
print(lpips_result["oneflow"].detach().numpy())
