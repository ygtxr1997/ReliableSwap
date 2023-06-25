import os
import numpy as np
import torch
from PIL import Image

from infoswap.preprocess.mtcnn import MTCNN
from deep3d.util.preprocess import align_img
from deep3d.util.load_mats import load_lm3d
from deep3d.models.bfm import ParametricFaceModel
from deep3d.models.networks import define_net_recon

make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))

class OPT(object):
    def __init__(self):
        pass

class Deep3DImageInfer(object):
    def __init__(self):
        self.bfm_folder = make_abs_path('./BFM')

        self.mtcnn = MTCNN()
        self.lm3d_std = load_lm3d(self.bfm_folder)

        self._load_model()

    def _load_model(self):
        opt = OPT()

        ''' base '''
        opt.name = 'face_recon'
        opt.model = 'facerecon'
        opt.epoch = 20

        ''' facerecon_model '''
        opt.net_recon = 'resnet50'
        opt.init_path = None  # no need to init if having pre-trained model
        opt.bfm_folder = make_abs_path('./BFM')
        opt.bfm_model = 'BFM_model_front.mat'

        opt.focal = 1015.
        opt.center = 112.
        opt.camera_d = 10.
        opt.z_near = 5.
        opt.z_far = 15.
        opt.use_opengl = False

        self.net_recon = define_net_recon(
            net_recon=opt.net_recon,
            use_last_fc=False,
            init_path=opt.init_path
        ).cuda().eval()

        self.facemodel = ParametricFaceModel(
            bfm_folder=opt.bfm_folder,
            camera_distance=opt.camera_d,
            focal=opt.focal,
            center=opt.center,
            is_train=False,
            default_name=opt.bfm_model
        )
        self.facemodel.to(0)

        load_filepath = make_abs_path('./checkpoints/face_recon/epoch_20.pth')
        state_dict = torch.load(load_filepath, map_location='cpu')
        self.net_recon.load_state_dict(state_dict['net_recon'])
        print('[Deep3D] model loaded from %s.' % load_filepath)

    @torch.no_grad()
    def infer_image(self, img_pil: Image):
        """ """
        ''' get 5 landmarks '''
        W, H = img_pil.size
        _, landmarks = self.mtcnn.detect_faces(img_pil)
        if len(landmarks) == 0:
            print('no face here, using default ffhq aligned coordinates.')
            landmarks = np.array([
                [192.98138, 239.94708],
                [318.90277, 240.1936],
                [256.63416, 314.01935],
                [201.26117, 371.41043],
                [313.08905, 371.15118],
            ]) / 512 * H  # default 512x512 ffhq aligned coordinates
            facial5points = landmarks
        else:
            facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
        facial5points_deep3d = np.array(facial5points)
        facial5points_deep3d[:, -1] = H - 1 - facial5points_deep3d[:, -1]

        _, im, lm, _ = align_img(img_pil, facial5points_deep3d, self.lm3d_std)

        ''' to tensor '''
        im = torch.tensor(np.array(im) / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).cuda()
        # lm = torch.tensor(lm).unsqueeze(0).cuda()

        ''' forward '''
        output_coeff = self.net_recon(im)
        # pred_vertex, pred_tex, pred_color, pred_lm = \
        #     self.facemodel.compute_for_render(output_coeff)
        # self.pred_mask, _, self.pred_face = self.renderer(
        #     self.pred_vertex, self.facemodel.face_buf, feat=self.pred_color)

        pred_coeffs_dict = self.facemodel.split_coeff(output_coeff)
        return pred_coeffs_dict


if __name__ == "__main__":
    deep3d_image_infer = Deep3DImageInfer()

    X1 = Image.open("./datasets/examples/result.jpg")
    X2 = Image.open("./datasets/examples/target.jpg")

    params1 = deep3d_image_infer.infer_image(X1)
    params2 = deep3d_image_infer.infer_image(X2)
    exp_dim = params1['exp'].shape[-1]

    l2_mean = torch.nn.functional.mse_loss(params1['exp'], params2['exp'], reduction='mean')
    l2_sum = torch.nn.functional.mse_loss(params1['exp'], params2['exp'], reduction='sum')
    print(torch.sqrt(l2_mean * exp_dim))
    print(torch.sqrt(l2_sum))
