#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
# from scipy.misc import imresize
# from imageio import imread
# from PIL import Image
import cv2

def RGB2YCbCr(rgb):
    R = rgb[:,:,0]
    G = rgb[:,:,1]
    B = rgb[:,:,2]


    Y = 0.257*R+0.504*G+0.098*B+16
    Cb = -0.148*R-0.291*G+0.439*B+128
    Cr = 0.439*R-0.368*G-0.071*B+128

    return np.dstack([Y, Cb, Cr])

def RGB2Lab(rgb):
    R = rgb[:,:,0] / 255.0
    G = rgb[:,:,1] / 255.0
    B = rgb[:,:,2] / 255.0
    T = 0.008856
    M, N = R.shape
    s = M * N
    RGB = np.r_[R.reshape((1, s)), G.reshape((1, s)), B.reshape((1, s))]
    MAT = np.array([[0.412453,0.357580,0.180423],
           [0.212671,0.715160,0.072169],
           [0.019334,0.119193,0.950227]])
    XYZ = np.dot(MAT, RGB)
    X = XYZ[0,:] / 0.950456
    Y = XYZ[1,:]
    Z = XYZ[2,:] / 1.088754


    XT = X > T
    YT = Y > T
    ZT = Z > T

    Y3 = np.power(Y, 1.0/3)
    fX = np.zeros(s)
    fY = np.zeros(s)
    fZ = np.zeros(s)

    fX[XT] = np.power(X[XT], 1.0 / 3)
    fX[~XT] = 7.787 * X[~XT] + 16.0 / 116

    fY[YT] = Y3[YT]
    fY[~YT] = 7.787 * Y[~YT] + 16.0 / 116

    fZ[ZT] = np.power(Z[ZT], 1.0 / 3)
    fZ[~ZT] = 7.787 * Z[~ZT] + 16.0 / 116

    L = np.zeros(YT.shape)
    a = np.zeros(fX.shape)
    b = np.zeros(fY.shape)

    L[YT] = Y3[YT] * 116 - 16.0
    L[~YT] = 903.3 * Y[~YT]

    a = 500 * (fX - fY)
    b = 200 * (fY - fZ)

    return np.dstack([L.reshape(R.shape), a.reshape(R.shape), b.reshape(R.shape)])

def Lab2RGB(Lab):
    M, N, C = Lab.shape
    s = M * N

    L = Lab[:,:,0].reshape((1, s)).astype(np.double)
    a = Lab[:,:,1].reshape((1, s)).astype(np.double)
    b = Lab[:,:,2].reshape((1, s)).astype(np.double)

    T1 = 0.008856
    T2 = 0.206893

    fY = np.power((L + 16.0) / 116, 3.0)
    YT = fY > T1
    fY[~YT] = L[~YT] / 903.3
    Y = fY.copy()

    fY[YT] = np.power(fY[YT], 1.0 / 3)
    fY[~YT] = 7.787 * fY[~YT] + 16.0 / 116

    fX = a / 500.0 + fY
    XT = fX > T2
    X = np.zeros((1, s))
    X[XT] = np.power(fX[XT], 3)
    X[~XT] = (fX[~XT] - 16.0 / 116) / 7.787

    fZ = fY - b / 200.0
    ZT = fZ > T2
    Z = np.zeros((1, s))
    Z[ZT] = np.power(fZ[ZT], 3)
    Z[~ZT] = (fZ[~ZT] - 16.0 / 116) / 7.787

    X = X * 0.950456
    Z = Z * 1.088754
    MAT = np.array([[ 3.240479,-1.537150,-0.498535],
       [-0.969256, 1.875992, 0.041556],
        [0.055648,-0.204043, 1.057311]])
    RGB = np.dot(MAT, np.r_[X,Y,Z])
    R = RGB[0, :].reshape((M,N))
    G = RGB[1, :].reshape((M,N))
    B = RGB[2, :].reshape((M,N))
    return np.clip(np.round(np.dstack([R,G,B]) * 255), 0, 255).astype(np.uint8)



def count(w):
    return dict(zip(*np.unique(w, return_counts = True)))
def count_array(w, size):
    d = count(w)
    return np.array([d.get(i, 0) for i in range(size)])

def get_border(Sa):
    si = np.argmax(Sa)
    t1 = si - 1
    t2 = si + 1
    diff = 0
    while t1 >= 0 and t2 <= 255:
        diff += (Sa[t1] - Sa[t2])
        if abs(diff) > 2 * max(Sa[t1], Sa[t2]) or Sa[t1] == 0 or Sa[t2] == 0:
            print("Sa", Sa[t1], Sa[t2])
            return [t1, t2]
        t1 -= 1
        t2 += 1
    t1 = max(0, t1)
    t2 = min(255, t2)
    return [t1, t2]


def deal(rgb):
    y = RGB2YCbCr(rgb)
    b = (y[:,:,1] >= 77) & (y[:,:,1] <= 127) & (y[:,:,2] >= 133) & (y[:,:,2] <= 173)
    lab = np.round(RGB2Lab(rgb)).astype(np.int)
    # a, b += 128
    lab[:,:,1:3] += 128
    # 0 ~ 255
    Sa = count_array(lab[:,:,1][b], 256)
    Sb = count_array(lab[:,:,2][b], 256)
    SaBorder = get_border(Sa)
    SbBorder = get_border(Sb)
    b2 = (((lab[:,:,1] >= SaBorder[0]) & (lab[:,:,1] <= SaBorder[1])) | ((lab[:,:,2] >= SbBorder[0]) & (lab[:,:,2] <= SbBorder[1])))
    plt.subplot(121)
    plt.imshow(b, "gray")
    plt.subplot(122)
    plt.imshow(b2, "gray")
    plt.show()
    return lab, b2, Sa, Sb, SaBorder, SbBorder, np.mean(lab[:,:,1][b2]), np.mean(lab[:,:,2][b2])


import cv2
import numexpr as ne
import numpy as np
import scipy as sp
from numpy import linalg as npla


def color_transfer_sot(src, trg, steps=10, batch_size=5, reg_sigmaXY=16.0, reg_sigmaV=5.0):
    """
    Color Transform via Sliced Optimal Transfer
    ported by @iperov from https://github.com/dcoeurjo/OTColorTransfer
    src         - any float range any channel image
    dst         - any float range any channel image, same shape as src
    steps       - number of solver steps
    batch_size  - solver batch size
    reg_sigmaXY - apply regularization and sigmaXY of filter, otherwise set to 0.0
    reg_sigmaV  - sigmaV of filter
    return value - clip it manually
    """
    if not np.issubdtype(src.dtype, np.floating):
        raise ValueError("src value must be float")
    if not np.issubdtype(trg.dtype, np.floating):
        raise ValueError("trg value must be float")

    if len(src.shape) != 3:
        raise ValueError("src shape must have rank 3 (h,w,c)")

    if src.shape != trg.shape:
        raise ValueError("src and trg shapes must be equal")

    src_dtype = src.dtype
    h, w, c = src.shape
    new_src = src.copy()

    advect = np.empty((h * w, c), dtype=src_dtype)
    for step in range(steps):
        advect.fill(0)
        for batch in range(batch_size):
            dir = np.random.normal(size=c).astype(src_dtype)
            dir /= npla.norm(dir)

            projsource = np.sum(new_src * dir, axis=-1).reshape((h * w))
            projtarget = np.sum(trg * dir, axis=-1).reshape((h * w))

            idSource = np.argsort(projsource)
            idTarget = np.argsort(projtarget)

            a = projtarget[idTarget] - projsource[idSource]
            for i_c in range(c):
                advect[idSource, i_c] += a * dir[i_c]
        new_src += advect.reshape((h, w, c)) / batch_size

    if reg_sigmaXY != 0.0:
        src_diff = new_src - src
        src_diff_filt = cv2.bilateralFilter(src_diff, 0, reg_sigmaV, reg_sigmaXY)
        if len(src_diff_filt.shape) == 2:
            src_diff_filt = src_diff_filt[..., None]
        new_src = src + src_diff_filt
    return new_src


def color_transfer_mkl(x0, x1):
    eps = np.finfo(float).eps

    h, w, c = x0.shape
    h1, w1, c1 = x1.shape

    x0 = x0.reshape((h * w, c))
    x1 = x1.reshape((h1 * w1, c1))

    a = np.cov(x0.T)
    b = np.cov(x1.T)

    Da2, Ua = np.linalg.eig(a)
    Da = np.diag(np.sqrt(Da2.clip(eps, None)))

    C = np.dot(np.dot(np.dot(np.dot(Da, Ua.T), b), Ua), Da)

    Dc2, Uc = np.linalg.eig(C)
    Dc = np.diag(np.sqrt(Dc2.clip(eps, None)))

    Da_inv = np.diag(1. / (np.diag(Da)))

    t = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(Ua, Da_inv), Uc), Dc), Uc.T), Da_inv), Ua.T)

    mx0 = np.mean(x0, axis=0)
    mx1 = np.mean(x1, axis=0)

    result = np.dot(x0 - mx0, t) + mx1
    return np.clip(result.reshape((h, w, c)).astype(x0.dtype), 0, 1)


def color_transfer_idt(i0, i1, bins=256, n_rot=20):
    import scipy.stats

    relaxation = 1 / n_rot
    h, w, c = i0.shape
    h1, w1, c1 = i1.shape

    i0 = i0.reshape((h * w, c))
    i1 = i1.reshape((h1 * w1, c1))

    n_dims = c

    d0 = i0.T
    d1 = i1.T

    for i in range(n_rot):

        r = sp.stats.special_ortho_group.rvs(n_dims).astype(np.float32)

        d0r = np.dot(r, d0)
        d1r = np.dot(r, d1)
        d_r = np.empty_like(d0)

        for j in range(n_dims):
            lo = min(d0r[j].min(), d1r[j].min())
            hi = max(d0r[j].max(), d1r[j].max())

            p0r, edges = np.histogram(d0r[j], bins=bins, range=[lo, hi])
            p1r, _ = np.histogram(d1r[j], bins=bins, range=[lo, hi])

            cp0r = p0r.cumsum().astype(np.float32)
            cp0r /= cp0r[-1]

            cp1r = p1r.cumsum().astype(np.float32)
            cp1r /= cp1r[-1]

            f = np.interp(cp0r, cp1r, edges[1:])

            d_r[j] = np.interp(d0r[j], edges[1:], f, left=0, right=bins)

        d0 = relaxation * np.linalg.solve(r, (d_r - d0r)) + d0

    return np.clip(d0.T.reshape((h, w, c)).astype(i0.dtype), 0, 1)


def reinhard_color_transfer(target: np.ndarray, source: np.ndarray, target_mask: np.ndarray = None,
                            source_mask: np.ndarray = None, mask_cutoff=0.5) -> np.ndarray:
    """
    Transfer color using rct method.
        target      np.ndarray H W 3C   (BGR)   np.float32
        source      np.ndarray H W 3C   (BGR)   np.float32
        target_mask(None)   np.ndarray H W 1C  np.float32
        source_mask(None)   np.ndarray H W 1C  np.float32

        mask_cutoff(0.5)    float
    masks are used to limit the space where color statistics will be computed to adjust the target
    reference: Color Transfer between Images https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf
    """
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)

    source_input = source
    if source_mask is not None:
        source_input = source_input.copy()
        source_input[source_mask[..., 0] < mask_cutoff] = [0, 0, 0]

    target_input = target
    if target_mask is not None:
        target_input = target_input.copy()
        target_input[target_mask[..., 0] < mask_cutoff] = [0, 0, 0]

    target_l_mean, target_l_std, target_a_mean, target_a_std, target_b_mean, target_b_std, \
        = target_input[..., 0].mean(), target_input[..., 0].std(), target_input[..., 1].mean(), target_input[
        ..., 1].std(), target_input[..., 2].mean(), target_input[..., 2].std()

    source_l_mean, source_l_std, source_a_mean, source_a_std, source_b_mean, source_b_std, \
        = source_input[..., 0].mean(), source_input[..., 0].std(), source_input[..., 1].mean(), source_input[
        ..., 1].std(), source_input[..., 2].mean(), source_input[..., 2].std()

    # not as in the paper: scale by the standard deviations using reciprocal of paper proposed factor
    target_l = target[..., 0]
    target_l = ne.evaluate('(target_l - target_l_mean) * source_l_std / target_l_std + source_l_mean')

    target_a = target[..., 1]
    target_a = ne.evaluate('(target_a - target_a_mean) * source_a_std / target_a_std + source_a_mean')

    target_b = target[..., 2]
    target_b = ne.evaluate('(target_b - target_b_mean) * source_b_std / target_b_std + source_b_mean')

    np.clip(target_l, 0, 100, out=target_l)
    np.clip(target_a, -127, 127, out=target_a)
    np.clip(target_b, -127, 127, out=target_b)

    return cv2.cvtColor(np.stack([target_l, target_a, target_b], -1), cv2.COLOR_LAB2BGR)


def linear_color_transfer(target_img, source_img, mode='pca', eps=1e-5):
    '''
    Matches the colour distribution of the target image to that of the source image
    using a linear transform.
    Images are expected to be of form (w,h,c) and float in [0,1].
    Modes are chol, pca or sym for different choices of basis.
    '''
    mu_t = target_img.mean(0).mean(0)
    t = target_img - mu_t
    t = t.transpose(2, 0, 1).reshape(t.shape[-1], -1)
    Ct = t.dot(t.T) / t.shape[1] + eps * np.eye(t.shape[0])
    mu_s = source_img.mean(0).mean(0)
    s = source_img - mu_s
    s = s.transpose(2, 0, 1).reshape(s.shape[-1], -1)
    Cs = s.dot(s.T) / s.shape[1] + eps * np.eye(s.shape[0])
    if mode == 'chol':
        chol_t = np.linalg.cholesky(Ct)
        chol_s = np.linalg.cholesky(Cs)
        ts = chol_s.dot(np.linalg.inv(chol_t)).dot(t)
    if mode == 'pca':
        eva_t, eve_t = np.linalg.eigh(Ct)
        Qt = eve_t.dot(np.sqrt(np.diag(eva_t))).dot(eve_t.T)
        eva_s, eve_s = np.linalg.eigh(Cs)
        Qs = eve_s.dot(np.sqrt(np.diag(eva_s))).dot(eve_s.T)
        ts = Qs.dot(np.linalg.inv(Qt)).dot(t)
    if mode == 'sym':
        eva_t, eve_t = np.linalg.eigh(Ct)
        Qt = eve_t.dot(np.sqrt(np.diag(eva_t))).dot(eve_t.T)
        Qt_Cs_Qt = Qt.dot(Cs).dot(Qt)
        eva_QtCsQt, eve_QtCsQt = np.linalg.eigh(Qt_Cs_Qt)
        QtCsQt = eve_QtCsQt.dot(np.sqrt(np.diag(eva_QtCsQt))).dot(eve_QtCsQt.T)
        ts = np.linalg.inv(Qt).dot(QtCsQt).dot(np.linalg.inv(Qt)).dot(t)
    matched_img = ts.reshape(*target_img.transpose(2, 0, 1).shape).transpose(1, 2, 0)
    matched_img += mu_s
    matched_img[matched_img > 1] = 1
    matched_img[matched_img < 0] = 0
    return np.clip(matched_img.astype(source_img.dtype), 0, 1)


def lab_image_stats(image):
    # compute the mean and standard deviation of each channel
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())

    # return the color statistics
    return (lMean, lStd, aMean, aStd, bMean, bStd)


def _scale_array(arr, clip=True):
    if clip:
        return np.clip(arr, 0, 255)

    mn = arr.min()
    mx = arr.max()
    scale_range = (max([mn, 0]), min([mx, 255]))

    if mn < scale_range[0] or mx > scale_range[1]:
        return (scale_range[1] - scale_range[0]) * (arr - mn) / (mx - mn) + scale_range[0]

    return arr


def channel_hist_match(source, template, hist_match_threshold=255, mask=None):
    # Code borrowed from:
    # https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
    masked_source = source
    masked_template = template

    if mask is not None:
        masked_source = source * mask
        masked_template = template * mask

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()
    masked_source = masked_source.ravel()
    masked_template = masked_template.ravel()
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles = hist_match_threshold * s_quantiles / s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles = 255 * t_quantiles / t_quantiles[-1]
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def color_hist_match(src_im, tar_im, hist_match_threshold=255):
    h, w, c = src_im.shape
    matched_R = channel_hist_match(src_im[:, :, 0], tar_im[:, :, 0], hist_match_threshold, None)
    matched_G = channel_hist_match(src_im[:, :, 1], tar_im[:, :, 1], hist_match_threshold, None)
    matched_B = channel_hist_match(src_im[:, :, 2], tar_im[:, :, 2], hist_match_threshold, None)

    to_stack = (matched_R, matched_G, matched_B)
    for i in range(3, c):
        to_stack += (src_im[:, :, i],)

    matched = np.stack(to_stack, axis=-1).astype(src_im.dtype)
    return matched


def color_transfer_mix(img_src, img_trg):
    img_src = np.clip(img_src * 255.0, 0, 255).astype(np.uint8)
    img_trg = np.clip(img_trg * 255.0, 0, 255).astype(np.uint8)

    img_src_lab = cv2.cvtColor(img_src, cv2.COLOR_BGR2LAB)
    img_trg_lab = cv2.cvtColor(img_trg, cv2.COLOR_BGR2LAB)

    rct_light = np.clip(linear_color_transfer(img_src_lab[..., 0:1].astype(np.float32) / 255.0,
                                              img_trg_lab[..., 0:1].astype(np.float32) / 255.0)[..., 0] * 255.0,
                        0, 255).astype(np.uint8)

    img_src_lab[..., 0] = (np.ones_like(rct_light) * 100).astype(np.uint8)
    img_src_lab = cv2.cvtColor(img_src_lab, cv2.COLOR_LAB2BGR)

    img_trg_lab[..., 0] = (np.ones_like(rct_light) * 100).astype(np.uint8)
    img_trg_lab = cv2.cvtColor(img_trg_lab, cv2.COLOR_LAB2BGR)

    img_rct = color_transfer_sot(img_src_lab.astype(np.float32), img_trg_lab.astype(np.float32))
    img_rct = np.clip(img_rct, 0, 255).astype(np.uint8)

    img_rct = cv2.cvtColor(img_rct, cv2.COLOR_BGR2LAB)
    img_rct[..., 0] = rct_light
    img_rct = cv2.cvtColor(img_rct, cv2.COLOR_LAB2BGR)

    return (img_rct / 255.0).astype(np.float32)

def face_color_transfer(source, target):
    slab, sb, Sa, Sb, [sab, sae],[sbb, sbe], sam, sbm = deal(source)
    tlab, tb, Ta, Tb, [tab, tae],[tbb, tbe], tam, tbm = deal(target)
    print("[sab, sae] = [%d, %d], [sbb, sbe] = [%d, %d]" % (sab,sae,sbb,sbe))
    print("[tab, tae] = [%d, %d], [tbb, tbe] = [%d, %d]" % (tab,tae,tbb,tbe))

    print(sam, sbm, tam, tbm)
    sam = (sab + sae) / 2.0
    sbm = (sbb + sbe) / 2.0
    tam = (tab + tae) / 2.0
    tbm = (tbb + tbe) / 2.0
    print(sam, sbm, tam, tbm)

    plt.plot(Sa, 'r.')
    plt.plot(Ta, 'r*')
    plt.plot(Sb, 'b.')
    plt.plot(Tb, 'b*')
    plt.show()

    rsa1 = (sam - sab) * 1.0 / (tam - tab)
    rsa2 = (sae - sam) * 1.0 / (tae - tam)
    rsb1 = (sbm - sbb) * 1.0 / (tbm - tbb)
    rsb2 = (sbe - sbm) * 1.0 / (tbe - tbm)
    print(rsa1, rsa2, rsb1, rsb2)

    def transfer(a, sam, tam, rsa1, rsa2, sab, sae):
        aold = a.copy()
        b = a < tam
        a[b] = rsa1 * (a[b] - tam) + sam
        a[~b] = rsa2 * (a[~b] - tam) + sam
        # Correction
        b1 = (a < sab) & (a > sab - 2)
        b2 = (a > sae) & (a < 2 + sae)
        b3 = (a > sab) & (a < sae)
        b4 = ~(b1 | b2 | b3)
        a[b1] = sab
        a[b2] = sae
        print(np.sum(b1), np.sum(b2), np.sum(b3), np.sum(b4))
        #a[b4] = aold[b4]
        return a

    plt.subplot(121)
    plt.imshow(sb, "gray")
    plt.subplot(122)
    plt.imshow(tb, "gray")
    plt.show()

    tlab[:,:,1][tb] = transfer(tlab[:,:,1][tb], sam, tam, rsa1, rsa2, sab, sae)
    tlab[:,:,2][tb] = transfer(tlab[:,:,2][tb], sbm, tbm, rsb1, rsb2, sbb, sbe)
    tlab[:,:,1:3] -= 128
    tlab[:,:,1:3] = np.clip(tlab[:,:,1:3], -128, 128)
    return Lab2RGB(tlab)

def imread(filename):
    im = mpimg.imread(filename)
    if im.dtype != np.uint8:
        im = np.clip(np.round(im * 255), 0, 255).astype(np.uint8)
    return im

# RGB

def skin_color_transfer(img_src, img_trg, ct_mode='mix'):
    """
    color transfer for [0,1] float32 inputs
    """
    if ct_mode == 'lct':
        out = linear_color_transfer(img_src, img_trg)
    elif ct_mode == 'rct':
        out = reinhard_color_transfer(img_src, img_trg)
    elif ct_mode == 'mkl':
        out = color_transfer_mkl(img_src, img_trg)
    elif ct_mode == 'idt':
        out = color_transfer_idt(img_src, img_trg)
    elif ct_mode == 'sot':
        out = color_transfer_sot(img_src, img_trg)
        out = np.clip(out, 0.0, 1.0)
    elif ct_mode == 'mix':
        out = color_transfer_mix(img_src, img_trg)
    elif ct_mode == 'adaptive':
        out = face_color_transfer(img_trg * 255, img_src * 255) / 255.
    else:
        raise ValueError(f"unknown ct_mode {ct_mode}")
    return out * 255


if __name__ == '__main__':
    target = imread("D:\\research\code\\face swapping\images\source4.png")
    source = imread("D:\\research\code\\face swapping\images\\2.jpg")
    '''
    source = imread("pic/boy5.png")
    target = imread("pic/girl2.png")
    '''
    res = face_color_transfer(source, target)

    cv2.imwrite('res.jpg', cv2.cvtColor(res, cv2.COLOR_RGB2BGR))

    plt.subplot(131)
    plt.title("source")
    plt.imshow(source)

    plt.subplot(132)
    plt.title("target")
    plt.imshow(target)

    plt.subplot(133)
    plt.title("Transfered")
    plt.imshow(res)

    plt.show()
