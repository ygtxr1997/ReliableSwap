'''
The code for calculating mutual information refers to the implementation of this paper:
@inproceedings{
    schulz2020iba,
    title={Restricting the Flow: Information Bottlenecks for Attribution},
    author={Schulz, Karl and Sixt, Leon and Tombari, Federico and Landgraf, Tim},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=S1xWh1rYwB}
}
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _kl_div(r, lambda_, mean_r, std_r):
    """ Return the feature-wise KL-divergence of p(z|x) and q(z)
        # The equation in the paper is:
        # Z = λ * R + (1 - λ) * ε)
        # where ε ~ N(μ_r, σ_r**2),
        #  and given R the distribution of Z ~ N(λ * R, ((1 - λ)*σ_r)**2) (λ * R is constant variable)
        #
        # As the KL-Divergence stays the same when both distributions are scaled,
        # normalizing Z such that Q'(Z) = N(0, 1) by using σ(R), μ(R).
        # Then for Gaussian Distribution:
        #   I(R, z) = KL[P(z|R)||Q(z)] = KL[P'(z|R)||N(0, 1)]
        #           = 0.5 * ( - log[det(noise)] - k + tr(noise_cov) + μ^T·μ )
    """
    r_norm = (r - mean_r) / std_r
    var_z = (1 - lambda_) ** 2
    log_var_z = torch.log(var_z)
    mu_z = r_norm * lambda_

    capacity = -0.5 * (1 + log_var_z - mu_z ** 2 - var_z)
    return capacity


class SpatialGaussianKernel(nn.Module):
    """ A simple convolutional layer with fixed gaussian kernels, used to smoothen the input """
    def __init__(self, kernel_size, sigma, channels, device):
        super().__init__()
        self.sigma = sigma
        self.kernel_size = kernel_size
        assert kernel_size % 2 == 1, "kernel_size must be an odd number (for padding), {} given".format(self.kernel_size)
        variance = sigma ** 2.
        x_cord = torch.arange(kernel_size, dtype=torch.float, device=device)  # 1, 2, 3, 4
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)  # 1, 2, 3 \ 1, 2, 3 \ 1, 2, 3
        y_grid = x_grid.t()  # 1, 1, 1 \ 2, 2, 2 \ 3, 3, 3
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        mean_xy = (kernel_size - 1) / 2.
        kernel_2d = (1. / (2. * math.pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean_xy) ** 2., dim=-1) /
            (2 * variance)
        )
        kernel_2d = kernel_2d / kernel_2d.sum()
        kernel_3d = kernel_2d.expand(channels, 1, -1, -1)  # expand in channel dimension
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels,
                              padding=0, kernel_size=kernel_size,
                              groups=channels, bias=False)
        self.conv.weight.data.copy_(kernel_3d)
        self.conv.to(device)
        self.conv.weight.requires_grad = False
        self.pad = nn.ReflectionPad2d(int((kernel_size - 1) / 2))

    def forward(self, x):
        return self.conv(self.pad(x))


class IBLayer(nn.Module):
    def __init__(self, in_c, out_c, device, smooth=True, kernel_size=1, sigma=1.):
        """
        Insert Information Bottleneck at one inter-feature
        :param in_c: sum of the channels of ‘readout_feats’
        :param out_c: the channel of ‘attach_feat’
        :param kernel_size: param of Gaussian Smooth
        """
        super(IBLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=in_c//2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=in_c//2, out_channels=out_c*2, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=out_c*2, out_channels=out_c, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()

        with torch.no_grad():
            nn.init.constant_(self.conv3.bias, 5.0)
            self.conv3.weight *= 1e-3

        self._alpha_bound = 5

        # Smoothing layer
        if smooth:
            if kernel_size is not None:
                # Construct static convolution layer with gaussian kernel
                sigma = kernel_size * 0.25  # Cover 2 stds in both directions
                self.smooth = SpatialGaussianKernel(kernel_size, sigma, out_c, device)
            elif sigma is not None and sigma > 0:
                # Construct static conv layer with gaussian kernel
                kernel_size = int(round(2 * sigma)) * 2 + 1  # Cover 2.5 stds in both directions
                self.smooth = SpatialGaussianKernel(kernel_size, sigma, out_c, device)
        else:
            self.smooth = None

        self.to(device)

    def forward(self, R, readout_feats, active_neurons, m_r=None, std_r=None, info_mean=True):
        # channel-wise mean and std: [Cr, Hr, Wr]
        m_r = torch.mean(R, dim=0) if m_r is None else m_r
        std_r = torch.std(R, dim=0) if std_r is None else std_r

        # --- 1. Get the smoothed mask 'lambda' for one attach feature
        readout = []
        for idx, feat in enumerate(readout_feats):
            # Reshape as spatial shape of the attach feature: [B, Cf, Hr, Wr]
            feat = F.interpolate(feat, size=(R.shape[-2], R.shape[-1]), mode='bilinear', align_corners=True)
            readout.append(feat)
        readout = torch.cat(readout, dim=1)  # [B, sum(Cf), Hr, Wr]

        # Pass through the readout network to obtain alpha
        alpha = self.conv1(readout)
        alpha = self.relu(alpha)
        alpha = self.conv2(alpha)
        alpha = self.relu(alpha)
        alpha = self.conv3(alpha)  # [B, Cr, Hr, Wr]

        # Keep alphas in a meaningful range during training
        alpha = alpha.clamp(-self._alpha_bound, self._alpha_bound)
        lambda_ = self.sigmoid(alpha)  # TODO:

        # Smoothing step
        lambda_ = self.smooth(lambda_) if self.smooth is not None else lambda_

        # --- 2. Get restricted latent feature Z
        eps = torch.randn(size=R.shape).to(R.device) * std_r + m_r  # [B, Cr, Hr, Wr]
        Z = R * lambda_ + (torch.ones_like(R).to(R.device) - lambda_) * eps

        info_capacity = _kl_div(R, lambda_, m_r, std_r) * active_neurons  # [Cr, Hr, Wr]
        Z *= active_neurons

        info = info_capacity.mean(dim=0) if info_mean else info_capacity

        return Z, lambda_, info


class IIB(nn.Module):
    def __init__(self, in_c, out_c_list, device, smooth=True, kernel_size=1):
        super(IIB, self).__init__()
        self.N = len(out_c_list)
        for i in range(self.N):
            setattr(self, f'iba_{i}', IBLayer(in_c, out_c=out_c_list[i], device=device,
                                              smooth=smooth, kernel_size=kernel_size))

        self.to(device)
        self.device = device

    def forward(self, model, B, N, readout_feats, param_dict=None):
        """
        Multi-average information bottleneck
        :param model:
        :param B: first [:B] samples are for Xs, last [B:] samples are for Xt
        :param N:
        :param readout_feats:
        :param param_dict:
        :return:
        """
        Info = 0.
        # X_id_restrict = torch.zeros_like(X_id).to(self.device)  # [2*B, 512]
        X_id_restrict = torch.zeros([2*B, 512]).to(self.device)
        Xt_feats = []
        Xs_feats = []
        X_lambda = []

        Rs_params = []
        Rt_params = []

        for i in range(N):  # multi-average information bottleneck
            R = readout_feats[i]  # [2*B, Cr, Hr, Wr]
            Z, lambda_, info = getattr(IIB, f'iba_{i}')(R, readout_feats,
                                                         m_r=param_dict[i][0], std_r=param_dict[i][1])

            # (1) loss
            X_id_restrict += model.restrict_forward(Z, i)  # [2*B, 512]
            Info += info.mean()

            # (2) attributes
            Rs, Rt = R[:B], R[B:]
            lambda_s, lambda_t = lambda_[:B], lambda_[B:]

            m_s = torch.mean(Rs, dim=0)  # [C, H, W]
            std_s = torch.mean(Rs, dim=0)
            Rs_params.append([m_s, std_s])
            eps_s = torch.randn(size=Rt.shape).to(Rt.device) * std_s + m_s
            feat_t = Rt * (1. - lambda_t) + lambda_t * eps_s
            Xt_feats.append(feat_t)  # only related with lambda

            m_t = torch.mean(Rt, dim=0)  # [C, H, W]
            std_t = torch.mean(Rt, dim=0)
            Rt_params.append([m_t, std_t])
            eps_t = torch.randn(size=Rs.shape).to(Rs.device) * std_t + m_t
            feat_s = Rs * (1. - lambda_s) + lambda_s * eps_t
            Xs_feats.append(feat_s)  # only related with lambda

            X_lambda.append(lambda_)

        X_id_restrict /= float(N)
        Info /= float(N)

        return X_id_restrict, Info, Xs_feats, Xt_feats, X_lambda, Rs_params, Rt_params


def get_restricted_id_attrs(model, readout_feats, lambdas, R_params, lamb_detach=True, calc_id=True, calc_Zid=False):
    feats = []
    Zs = []
    N = len(readout_feats)
    id = None
    for i, R in enumerate(readout_feats):
        R = readout_feats[i]
        lamb = lambdas[i].detach() if lamb_detach else lambdas[i]
        m, std = R_params[i]

        eps = torch.randn(size=R.shape).to(R.device) * std + m
        feat = R * (1. - lamb) + lamb * eps  # to prevent updating iba via lambda
        feats.append(feat)
        if calc_Zid:
            Z = R * lamb + eps * (1. - lamb)
            Zs.append(Z)

        if calc_id:
            if id is None:
                id = model.restrict_forward(feat, i)
            else:
                id += model.restrict_forward(feat, i)
    if calc_id:
        id /= float(N)
    if calc_Zid:
        return id, feats, Zs
    else:
        return id, feats
