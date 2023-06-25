import torch
import glob
import os
import sys
import cv2
import tqdm
import shutil
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import torch.nn.functional as F
from torchvision import transforms
from multiprocessing.pool import Pool

sys.path.append("../")

from modules.networks.faceshifter import FSGenerator
#from faceswap.networks.faceshifter_mod import FSGenerator as FSGenerator_v2
from inference.alignment import (
    norm_crop,
    norm_crop_with_M,
    paste_back,
)
from utils import save, get_5_from_98, get_detector, get_lmk

from PIPNet.lib.tools import get_lmk_model, demo_image
from landmark_smooth import kalman_filter_landmark, savgol_filter_landmark

make_abs_path = lambda fn: os.path.join(os.path.dirname(os.path.realpath(__file__)), fn)

parser = argparse.ArgumentParser(description="face swap")
parser.add_argument("-s", "--source", type=str, required=True)
parser.add_argument("-t", "--target", type=str, required=True)
parser.add_argument("-o", "--out", type=str, required=True, help='output folder')
parser.add_argument("-n", "--out_name", type=str, default="output.mp4")
parser.add_argument("-f", "--frames", type=int, default=0)
parser.add_argument("-pp", "--pool_process", type=int, default=170)
parser.add_argument("--align_source", type=str, default="ffhq")
parser.add_argument("--align_target", type=str, default="ffhq")
parser.add_argument(
    "-cp",
    "--ckpt_path",
    type=str,
    default="../../trainer/faceshifter/extracted_ckpt/G_step_14999_v2.pt",
)
parser.add_argument("--video", dest="video", action="store_true")
parser.add_argument("--no-video", dest="video", action="store_false")
parser.add_argument("--audio", dest="audio", action="store_true")
parser.add_argument("--no-audio", dest="audio", action="store_false")
parser.add_argument("--concat", dest="concat", action="store_true")
parser.add_argument("--no-concat", dest="concat", action="store_false")
parser.add_argument("--tddfav2", dest="tddfav2", action="store_true")
parser.add_argument("--no-tddfav2", dest="tddfav2", action="store_false")
parser.set_defaults(tddfav2=False)
parser.set_defaults(video=True)
parser.set_defaults(audio=True)
parser.set_defaults(concat=True)

landmark_smooth_choices = ['kalman', 'savgol', 'cancel']
parser.add_argument("--landmark_smooth", type=str, default="kalman",
                    choices=landmark_smooth_choices)
parser.add_argument("--vis_landmarks", type=bool, default=False)

args = parser.parse_args()


def load_extracted(path="./extracted_ckpt/G_test.pt"):
    if 'infoswap' in path:
        print('[Loading pretrained InfoSwap model]')
        from modules.third_party.infoswap.inference_model import InfoSwapInference
        G = InfoSwapInference()
        G.load_model()
        return G

    G = FSGenerator(
        "../modules/third_party/arcface/weights/ms1mv3_arcface_r100_fp16/backbone.pth"
    )
    if 'hanbang' in path:
        print('[Loading cached hanbang model]')
        G.load_state_dict(
            torch.load('../trainer/faceshifter/extracted_ckpt/G_step14999_v2.pth', 'cpu'),
            strict=False
        )
        args.align_source = 'arcface'
        args.align_target = 'set1'
    else:
        G.load_state_dict(torch.load(path, "cpu"), strict=False)
    G.eval()
    return G


def get_model(model):
    G = model.G
    iresnet = model.iresnet
    return G, iresnet


def swap_image_multiple_source(
    G,
    source,
    target,
    out_path,
    align_source="arcface",
    align_target="set1",
    gpu_mode=True,
    T=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
    ),
):
    G, iresnet = get_model(G)
    G.eval()
    iresnet.eval()
    net, detector = get_lmk_model()

    target = np.array(Image.open(target))
    original_target = target.copy()

    lmk = get_5_from_98(demo_image(target, net, detector)[0])
    target, M = norm_crop_with_M(target, lmk, 256, mode=align_target, borderValue=0.0)
    target = T(target).unsqueeze(0)

    source_imgs = []
    for x in glob.glob(os.path.join(source, "*")):
        if os.path.isdir(x):
            continue
        source_img = np.array(Image.open(x).convert("RGB"))
        lmk = get_5_from_98(demo_image(source_img, net, detector)[0])
        source_img = norm_crop(source_img, lmk, 256, mode=align_source, borderValue=0.0)
        source_imgs.append(T(source_img).unsqueeze(0))
    source = torch.cat(source_imgs, dim=0)
    name = "out.png"

    if gpu_mode:
        G = G.cuda()
        iresnet = iresnet.cuda()
        source = source.cuda()
        target = target.cuda()

    with torch.no_grad():
        id_vector = F.normalize(
            iresnet(F.interpolate(source, size=112, mode="bilinear")),
            dim=-1,
            p=2,
        )
        id_vector = id_vector.mean(0, keepdim=True)
        target, _ = G(target, id_vector, infer=True)
    target = np.array(tensor2pil_transform(target[0] * 0.5 + 0.5))
    os.makedirs(out_path, exist_ok=True)
    Image.fromarray(target.astype(np.uint8)).save(os.path.join(out_path, name))
    save((target, M, original_target, os.path.join(out_path, "paste_back_" + name)))


def swap_image(
    source_image,
    target_path,
    out_path,
    transform,
    G,
    align_source="arcface",
    align_target="set1",
    gpu_mode=True,
    paste_back=True,
):
    name = target_path.split("/")[-1]
    name = "out_" + name
    G.eval()
    if gpu_mode:
        G = G.cuda()
    source_img = np.array(Image.open(source_image).convert("RGB"))
    net, detector = get_lmk_model()
    lmk = get_5_from_98(demo_image(source_img, net, detector)[0])
    source_img = norm_crop(source_img, lmk, 256, mode=align_source, borderValue=0.0)
    source_img = transform(source_img).unsqueeze(0)

    target = np.array(Image.open(target_path).convert("RGB"))
    original_target = target.copy()
    lmk = get_5_from_98(demo_image(target, net, detector)[0])
    target, M = norm_crop_with_M(target, lmk, 256, mode=align_target, borderValue=0.0)
    target = transform(target).unsqueeze(0)
    if gpu_mode:
        target = target.cuda()
        source_img = source_img.cuda()
    with torch.no_grad():
        output = G(source_img, target, infer=True)
        if isinstance(output, tuple):
            target = output[0][0] * 0.5 + 0.5
        else:
            target = output[0] * 0.5 + 0.5
    target = np.array(tensor2pil_transform(target))
    os.makedirs(out_path, exist_ok=True)
    Image.fromarray(target.astype(np.uint8)).save(os.path.join(out_path, name))
    save((target, M, original_target, os.path.join(out_path, "paste_back_" + name)))


def process_video(
    source_image,
    target_path,
    out_path,
    transform,
    G,
    align_source="arcface",
    align_target="set1",
    gpu_mode=True,
    frames=9999999,
    use_tddfav2=False,
):
    G.eval()
    if gpu_mode:
        G = G.cuda()

    fps = 25.0
    if not os.path.isdir(target_path):
        vidcap = cv2.VideoCapture(target_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        try:
            for match in glob.glob(os.path.join("./tmp/", "*.png")):
                os.remove(match)
            for match in glob.glob(os.path.join(out_path, "*.png")):
                os.remove(match)
        except Exception as e:
            print(e)
        os.makedirs("./tmp/", exist_ok=True)
        os.system(
            f"ffmpeg -i {target_path} -qscale:v 1 -qmin 1 -qmax 1 -vsync 0  ./tmp/frame_%d.png"
        )
        target_path = "./tmp/"

    globbed_images = sorted(glob.glob(os.path.join(target_path, "*.png")))
    source_img = np.array(Image.open(source_image).convert("RGB"))
    if not use_tddfav2:
        net, detector = get_lmk_model()
        lmk = demo_image(source_img, net, detector)
        lmk = lmk[0]
        lmk = get_5_from_98(lmk)
    else:
        net, detector = get_detector(gpu_mode=gpu_mode)
        lmk = get_lmk(source_img, net, detector)

    source_img = norm_crop(source_img, lmk, 256, mode=align_source, borderValue=0.0)
    source_img = transform(source_img).unsqueeze(0)
    source_img = source_img.cuda() if gpu_mode else source_img

    targets = []
    target_lmks = []
    Ms = []
    original_frames = []
    names = []

    """ Get target landmarks """
    print('[Extracting target landmarks...]')
    count = 0
    for image in tqdm.tqdm(globbed_images):
        target = np.array(Image.open(image).convert("RGB"))
        if not use_tddfav2:
            lmk = demo_image(target, net, detector)
            lmk = lmk[0]
            target_lmks.append(lmk)
        else:
            raise NotImplementedError('tddfa not used!')
            lmk = get_lmk(target, net, detector)
        count += 1
        if count > frames:
            break

    """ Landmark smoothing """
    target_lmks = np.array(target_lmks, np.float32)  # (#frames, 98, 2)
    if args.landmark_smooth == 'kalman':
        target_lmks = kalman_filter_landmark(target_lmks).astype(np.int)
    elif args.landmark_smooth == 'savgol':
        target_lmks = savgol_filter_landmark(target_lmks).astype(np.int)
    elif args.landmark_smooth == 'cancel':
        target_lmks = target_lmks.astype(np.int)
    else:
        raise KeyError('Not supported landmark_smooth choice')

    """ Crop target images according to the landmarks """
    print('[Cropping targets and inferring...]')
    count = 0
    for image in tqdm.tqdm(globbed_images):
        names.append(os.path.join(out_path, Path(image).name))
        target = np.array(Image.open(image).convert("RGB"))
        original_frames.append(target)
        try:
            if not use_tddfav2:
                num_keypoints = 98 if args.vis_landmarks else 0
                for point in range(num_keypoints):
                    cv2.circle(target, target_lmks[count][point], 1, (0, 0, 255), 1,)
                lmk = get_5_from_98(target_lmks[count])
            else:
                raise NotImplementedError('tddfa not used!')
                lmk = get_lmk(target, net, detector)
            target, M = norm_crop_with_M(
                target,
                lmk, 256, mode=align_target, borderValue=0.0
            )
            target = transform(target).unsqueeze(0)
        except Exception as e:
            print(e)
            targets.append(None)
            Ms.append(None)
            continue

        target = target.cuda() if gpu_mode else target

        """ Inference results or Visualize landmarks """
        with torch.no_grad():
            if not args.vis_landmarks:
                output = G(source_img, target, infer=True)
                if isinstance(output, tuple):
                    target = output[0][0] * 0.5 + 0.5
                else:
                    target = output[0] * 0.5 + 0.5
            else:
                target = target[0] * 0.5 + 0.5

        targets.append(np.array(tensor2pil_transform(target)))
        Ms.append(M)

        count += 1
        if count > frames:
            break

    os.makedirs(out_path, exist_ok=True)
    return targets, Ms, original_frames, names, fps


if __name__ == "__main__":
    T = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
    )
    tensor2pil_transform = transforms.ToPILImage()
    G = load_extracted(path=args.ckpt_path)

    frames = args.frames if args.frames > 0 else 9999999
    use_gpu = torch.cuda.device_count() > 0

    ext_ = args.target.split("/")[-1]
    if any(x in ext_ for x in ["jpg", "png", "jpeg"]):
        swap_image(
            args.source,
            args.target,
            args.out,
            T,
            G,
            gpu_mode=use_gpu,
            align_target=args.align_target,
            align_source=args.align_source,
        )
        exit()

    targets, Ms, original_frames, names, fps = process_video(
        args.source,
        args.target,
        args.out,
        T,
        G,
        gpu_mode=use_gpu,
        frames=frames,
        align_target=args.align_target,
        align_source=args.align_source,
        use_tddfav2=args.tddfav2,
    )

    if args.pool_process <= 1:
        for target, M, original_target, name in tqdm.tqdm(
            zip(targets, Ms, original_frames, names)
        ):
            if M is None or target is None:
                Image.fromarray(original_target.astype(np.uint8)).save(name)
                continue
            Image.fromarray(paste_back(np.array(target), M, original_target)).save(name)
    else:
        with Pool(args.pool_process) as pool:
            pool.map(save, zip(targets, Ms, original_frames, names))

    # merge frames to video
    if args.video:
        video_save_path = os.path.join(args.out, args.out_name)
        if args.audio:
            print("use audio")
            os.system(
                f"ffmpeg  -y -r {fps} -i {args.out}/frame_%d.png -i {args.target}"
                f" -map 0:v:0 -map 1:a:0? -c:a copy -c:v libx264 -crf 10 -r {fps} -pix_fmt yuv420p {video_save_path}"
            )
        else:
            print("no audio")
            os.system(
                f"ffmpeg  -y -r {fps} -i ./tmp/frame_%d.png "
                f"-c:v libx264 -crf 10 -r {fps} -pix_fmt yuv420p {video_save_path}"
            )

        # ffmpeg -i left.mp4 -i right.mp4 -filter_complex hstack output.mp4
        if args.concat:
            concat_video_save_path = os.path.join(args.out, "concat_" + args.out_name)
            os.system(
                f"ffmpeg -y  -i {args.target}  -i {video_save_path} -filter_complex hstack {concat_video_save_path}"
            )

        # delete tmp file
        shutil.rmtree("./tmp/")
        for match in glob.glob(os.path.join(args.out, "*.png")):
            os.remove(match)
