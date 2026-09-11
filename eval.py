from data.thermal_dataset import ThermalDataset
from data.flir_dataset import FlirDataset
import torch.utils.data
from pytorch_msssim import SSIM as my_ssim, MS_SSIM
from lpips.lpips import LPIPS
import tensorflow as tf
from collections import OrderedDict


class Evalulate:
    def __init__(self, opt):
        mode = "test"  # or val
        if opt.dataset_mode == 'VEDAI':
            dataset = ThermalDataset()
            dataset.initialize(opt, mode="test")  # or mode="val"
        elif opt.dataset_mode == 'KAIST':
            dataset = ThermalDataset()
            # mode = '/cta/users/mehmet/rgbt-ped-detection/data/scripts/imageSets/test-all-20.txt'
            dataset.initialize(opt, mode=mode)
        elif opt.dataset_mode == 'FLIR':
            dataset = FlirDataset()
            dataset.initialize(opt, test=True)
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,  # opt.batchSize,
            shuffle=False,
            num_workers=int(opt.nThreads))

        self.lpips_obj = LPIPS(net='alex')
        self.mssim_obj = MS_SSIM(channel=1, size_average=True, data_range=2.)
        self.ssim_obj = my_ssim(channel=1, size_average=True, data_range=2.)
        self.L1_obj = torch.nn.L1Loss()
        # TODO: No flip

    def eval(self, model, visualizer):

        mssim, ssim, i, lpips, psnr, l1 = 0, 0, 0, 0, 0, 0
        with torch.no_grad():
            for i, data in enumerate(self.dataloader):
                model.set_input(data)
                model.test()
                visualizer.add_errors(model.get_current_errors())
                l1 = (l1 * i + self.L1_obj(model.real_B, model.fake_B).item()) / (i + 1)
                mssim = (mssim * i + self.mssim_obj(model.real_B, model.fake_B).item()) / (i + 1)
                ssim = (ssim * i + self.ssim_obj(model.real_B, model.fake_B).item()) / (i + 1)
                lpips = (lpips * i + self.lpips_obj(model.real_B.cpu(), model.fake_B.cpu()).mean().item()) / (i + 1)
                psnr = (psnr * i + tf.image.psnr(tf.convert_to_tensor(model.real_B.detach().cpu().numpy()),
                                                 tf.convert_to_tensor(model.fake_B.detach().cpu().numpy()), 2).numpy()) / (i + 1)
        visualizer.append_error_hist(i, val=True)

        return ssim, mssim, l1, psnr, lpips

    def get_eval_metrix(self, ssim, mssim, l1, psnr, lpips):
        ret_metrix = OrderedDict([('SSIM', ssim), ('MSSIM', mssim), ('L1', l1),
                                  ('PSNR', psnr), ('LPIPS', lpips)])
        return ret_metrix

# mssim /= (i + 1)
# ssim /= (i + 1)
# print("ok,mssim:{},ssim{}".format(mssim, ssim))
