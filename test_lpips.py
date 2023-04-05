import lpips
import flow_lpips
from PIL import Image

import numpy as np
import oneflow.mock_torch as mock

import unittest


def _test_lpips(test_case, net, version):
    loss_torch = lpips.LPIPS(net=net, version=version)
    loss_flow = flow_lpips.LPIPS(net=net, version=version)

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
            torch.from_numpy(pred.transpose(2, 0, 1)).unsqueeze(0).float(),
            torch.from_numpy(ref.transpose(2, 0, 1)).unsqueeze(0).float(),
        )

    for mode in ["pytorch", "oneflow"]:
        if mode == "pytorch":
            with mock.disable():
                lpips_result[mode] = loss_fn(pred_image, ref_image, False).item()
        elif mode == "oneflow":
            with mock.enable():
                lpips_result[mode] = loss_fn(pred_image, ref_image, True).item()

    test_case.assertTrue(
        np.allclose(
            np.array(lpips_result["pytorch"]),
            np.array(lpips_result["oneflow"]),
        )
    )


class TestLPIPS(unittest.TestCase):
    def test_lpips(self):
        nets = ["alex", "vgg", "squeeze"]
        versions = ["0.0", "0.1"]
        for net in nets:
            for version in versions:
                _test_lpips(self, net, version)


if __name__ == "__main__":
    unittest.main()
