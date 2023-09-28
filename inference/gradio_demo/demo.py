import os
import uuid
import glob
import shutil
from pathlib import Path
from multiprocessing.pool import Pool

import gradio as gr
import torch
from torchvision import transforms

import cv2
import numpy as np
from PIL import Image
import tqdm

from modules.networks.faceshifter import FSGenerator
from inference.alignment import norm_crop, norm_crop_with_M, paste_back
from inference.utils import save, get_5_from_98, get_detector, get_lmk
from inference.PIPNet.lib.tools import get_lmk_model, demo_image
from inference.landmark_smooth import kalman_filter_landmark, savgol_filter_landmark
from tricks import Trick

make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))


fs_model_name = 'faceshifter'
in_size = 512

mouth_net_param = {
    "use": True,
    "feature_dim": 128,
    "crop_param": (28, 56, 84, 112),
    "weight_path": "../../modules/third_party/arcface/weights/mouth_net_28_56_84_112.pth",
}
trick = Trick()

T = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
    )
tensor2pil_transform = transforms.ToPILImage()


def extract_generator(ckpt: str, pt: str):
    print(f'[extract_generator] loading ckpt...')
    from trainer.faceshifter.faceshifter_pl import FaceshifterPL512, FaceshifterPL
    import yaml
    with open(make_abs_path('../../trainer/faceshifter/config.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['mouth_net'] = mouth_net_param

    if in_size == 256:
        net = FaceshifterPL(n_layers=3, num_D=3, config=config)
    elif in_size == 512:
        net = FaceshifterPL512(n_layers=3, num_D=3, config=config, verbose=False)
    else:
        raise ValueError('Not supported in_size.')
    checkpoint = torch.load(ckpt, map_location="cpu", )
    net.load_state_dict(checkpoint["state_dict"], strict=False)
    net.eval()

    G = net.generator
    torch.save(G.state_dict(), pt)
    print(f'[extract_generator] extracted from {ckpt}, pth saved to {pt}')


''' load model '''
if fs_model_name == 'faceshifter':
    # pt_path = make_abs_path("../ffplus/extracted_ckpt/G_mouth1_t38.pth")
    # pt_path = make_abs_path("../ffplus/extracted_ckpt/G_mouth1_t512_6.pth")
    # ckpt_path = "/apdcephfs/share_1290939/gavinyuan/out/triplet512_6/epoch=3-step=128999.ckpt"
    pt_path = make_abs_path("../ffplus/extracted_ckpt/G_mouth1_t512_4.pth")
    ckpt_path = "/apdcephfs/share_1290939/gavinyuan/out/triplet512_4/epoch=2-step=185999.ckpt"
    if not os.path.exists(pt_path) or 't512' in pt_path:
        extract_generator(ckpt_path, pt_path)
    fs_model = FSGenerator(
        make_abs_path("../../modules/third_party/arcface/weights/ms1mv3_arcface_r100_fp16/backbone.pth"),
        mouth_net_param=mouth_net_param,
        in_size=in_size,
        downup=in_size == 512,
    )
    fs_model.load_state_dict(torch.load(pt_path, "cpu"), strict=True)
    fs_model.eval()

    @torch.no_grad()
    def infer_batch_to_img(i_s, i_t, post: bool = False):
        i_r = fs_model(i_s, i_t)[0]  # x, id_vector, att

        if post:
            target_hair_mask = trick.get_any_mask(i_t, par=[0, 17])
            target_hair_mask = trick.smooth_mask(target_hair_mask)
            i_r = target_hair_mask * i_t + (target_hair_mask * (-1) + 1) * i_r
            i_r = trick.finetune_mouth(i_s, i_t, i_r) if in_size == 256 else i_r

        img_r = trick.tensor_to_arr(i_r)[0]
        return img_r

elif fs_model_name == 'simswap_triplet' or fs_model_name == 'simswap_vanilla':
    from modules.networks.simswap import Generator_Adain_Upsample
    sw_model = Generator_Adain_Upsample(
        input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=False,
        mouth_net_param=mouth_net_param
    )
    if fs_model_name == 'simswap_triplet':
        pt_path = make_abs_path("../ffplus/extracted_ckpt/G_mouth1_st5.pth")
        ckpt_path = make_abs_path("/apdcephfs/share_1290939/gavinyuan/out/"
                                  "simswap_triplet_5/epoch=12-step=782999.ckpt")
    elif fs_model_name == 'simswap_vanilla':
        pt_path = make_abs_path("../ffplus/extracted_ckpt/G_tmp_sv4_off.pth")
        ckpt_path = make_abs_path("/apdcephfs/share_1290939/gavinyuan/out/"
                                  "simswap_vanilla_4/epoch=694-step=1487999.ckpt")
    else:
        pt_path = None
        ckpt_path = None
    sw_model.load_state_dict(torch.load(pt_path, "cpu"), strict=False)
    sw_model.eval()
    fs_model = sw_model

    from trainer.simswap.simswap_pl import SimSwapPL
    import yaml
    with open(make_abs_path('../../trainer/simswap/config.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['mouth_net'] = mouth_net_param
    net = SimSwapPL(config=config, use_official_arc='off' in pt_path)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    net.load_state_dict(checkpoint["state_dict"], strict=False)
    net.eval()
    sw_mouth_net = net.mouth_net  # maybe None
    sw_netArc = net.netArc
    fs_model = fs_model.cuda()
    sw_mouth_net = sw_mouth_net.cuda() if sw_mouth_net is not None else sw_mouth_net
    sw_netArc = sw_netArc.cuda()

    @torch.no_grad()
    def infer_batch_to_img(i_s, i_t, post: bool = False):
        i_r = fs_model(source=i_s, target=i_t, net_arc=sw_netArc, mouth_net=sw_mouth_net,)
        if post:
            target_hair_mask = trick.get_any_mask(i_t, par=[0, 17])
            target_hair_mask = trick.smooth_mask(target_hair_mask)
            i_r = target_hair_mask * i_t + (target_hair_mask * (-1) + 1) * i_r
        i_r = i_r.clamp(-1, 1)
        i_r = trick.tensor_to_arr(i_r)[0]
        return i_r

elif fs_model_name == 'simswap_official':
    from simswap.image_infer import SimSwapOfficialImageInfer
    fs_model = SimSwapOfficialImageInfer()
    pt_path = 'Simswap Official'
    mouth_net_param = {
        "use": False
    }

    @torch.no_grad()
    def infer_batch_to_img(i_s, i_t):
        i_r = fs_model.image_infer(source_tensor=i_s, target_tensor=i_t)
        i_r = i_r.clamp(-1, 1)
        return i_r

else:
    raise ValueError('Not supported fs_model_name.')


print(f'[demo] model loaded from {pt_path}')


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
    use_post=False,
    use_gpen=False,
    in_size=256,
):
    name = target_path.split("/")[-1]
    name = "out_" + name
    if isinstance(G, torch.nn.Module):
        G.eval()
        if gpu_mode:
            G = G.cuda()
    source_img = np.array(Image.open(source_image).convert("RGB"))
    net, detector = get_lmk_model()
    lmk = get_5_from_98(demo_image(source_img, net, detector)[0])
    source_img = norm_crop(source_img, lmk, in_size, mode=align_source, borderValue=0.0)
    source_img = transform(source_img).unsqueeze(0)

    target = np.array(Image.open(target_path).convert("RGB"))
    original_target = target.copy()
    lmk = get_5_from_98(demo_image(target, net, detector)[0])
    target, M = norm_crop_with_M(target, lmk, in_size, mode=align_target, borderValue=0.0)
    target = transform(target).unsqueeze(0)
    if gpu_mode:
        target = target.cuda()
        source_img = source_img.cuda()

    cv2.imwrite('cropped_source.png', trick.tensor_to_arr(source_img)[0, :, :, ::-1])
    cv2.imwrite('cropped_target.png', trick.tensor_to_arr(target)[0, :, :, ::-1])

    # both inputs should be 512
    result = infer_batch_to_img(source_img, target, post=use_post)

    cv2.imwrite('result.png', result[:, :, ::-1])

    os.makedirs(out_path, exist_ok=True)
    Image.fromarray(result.astype(np.uint8)).save(os.path.join(out_path, name))
    save((result, M, original_target, os.path.join(out_path, "paste_back_" + name), None),
         trick=trick, use_post=use_gpen)


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
    landmark_smooth="kalman",
):
    if isinstance(G, torch.nn.Module):
        G.eval()
        if gpu_mode:
            G = G.cuda()
    ''' Target video to frames (.png) '''
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
            f"ffmpeg -i {target_path} -qscale:v 1 -qmin 1 -qmax 1 -vsync 0  ./tmp/frame_%05d.png"
        )
        target_path = "./tmp/"
    globbed_images = sorted(glob.glob(os.path.join(target_path, "*.png")))
    ''' Get target landmarks '''
    print('[Extracting target landmarks...]')
    if not use_tddfav2:
        align_net, align_detector = get_lmk_model()
    else:
        align_net, align_detector = get_detector(gpu_mode=gpu_mode)
    target_lmks = []
    for frame_path in tqdm.tqdm(globbed_images):
        target = np.array(Image.open(frame_path).convert("RGB"))
        lmk = demo_image(target, align_net, align_detector)
        lmk = lmk[0]
        target_lmks.append(lmk)
    ''' Landmark smoothing '''
    target_lmks = np.array(target_lmks, np.float32)  # (#frames, 98, 2)
    if landmark_smooth == 'kalman':
        target_lmks = kalman_filter_landmark(target_lmks,
                                             process_noise=0.01,
                                             measure_noise=0.01).astype(np.int)
    elif landmark_smooth == 'savgol':
        target_lmks = savgol_filter_landmark(target_lmks).astype(np.int)
    elif landmark_smooth == 'cancel':
        target_lmks = target_lmks.astype(np.int)
    else:
        raise KeyError('Not supported landmark_smooth choice')
    ''' Crop source image '''
    source_img = np.array(Image.open(source_image).convert("RGB"))
    if not use_tddfav2:
        lmk = get_5_from_98(demo_image(source_img, align_net, align_detector)[0])
    else:
        lmk = get_lmk(source_img, align_net, align_detector)
    source_img = norm_crop(source_img, lmk, in_size, mode=align_source, borderValue=0.0)
    source_img = transform(source_img).unsqueeze(0)
    if gpu_mode:
        source_img = source_img.cuda()
    ''' Process by frames '''
    targets = []
    t_facial_masks = []
    Ms = []
    original_frames = []
    names = []
    count = 0
    for image in tqdm.tqdm(globbed_images):
        names.append(os.path.join(out_path, Path(image).name))
        target = np.array(Image.open(image).convert("RGB"))
        original_frames.append(target)
        ''' Crop target frames '''
        lmk = get_5_from_98(target_lmks[count])
        target, M = norm_crop_with_M(target, lmk, in_size, mode=align_target, borderValue=0.0)
        target = transform(target).unsqueeze(0)  # in [-1,1]
        if gpu_mode:
            target = target.cuda()
        ''' Finetune paste masks '''
        target_facial_mask = trick.get_any_mask(target,
                                                par=[1, 2, 3, 4, 5, 6, 10, 11, 12, 13]).squeeze()  # in [0,1]
        target_facial_mask = target_facial_mask.cpu().numpy().astype(np.float32)
        target_facial_mask = trick.finetune_mask(target_facial_mask, target_lmks)  # in [0,1]
        t_facial_masks.append(target_facial_mask)
        ''' Face swapping '''
        with torch.no_grad():
            if 'faceshifter' in fs_model_name:
                output = G(source_img, target)
                target_hair_mask = trick.get_any_mask(target, par=[0, 17])
                target_hair_mask = trick.smooth_mask(target_hair_mask)
                output = target_hair_mask * target + (target_hair_mask * (-1) + 1) * output
                output = trick.finetune_mouth(source_img, target, output)
            elif 'simswap' in fs_model_name and 'official' not in fs_model_name:
                output = fs_model(source=source_img, target=target,
                                  net_arc=sw_netArc, mouth_net=sw_mouth_net,)
                if 'vanilla' not in fs_model_name:
                    target_hair_mask = trick.get_any_mask(target, par=[0, 17])
                    target_hair_mask = trick.smooth_mask(target_hair_mask)
                    output = target_hair_mask * target + (target_hair_mask * (-1) + 1) * output
                    output = trick.finetune_mouth(source_img, target, output)
                output = output.clamp(-1, 1)
            elif 'simswap_official' in fs_model_name:
                output = fs_model.image_infer(source_tensor=source_img, target_tensor=target)
                output = output.clamp(-1, 1)
            if isinstance(output, tuple):
                target = output[0][0] * 0.5 + 0.5
            else:
                target = output[0] * 0.5 + 0.5
        targets.append(np.array(tensor2pil_transform(target)))
        Ms.append(M)
        count += 1
        if count > frames:
            break
    os.makedirs(out_path, exist_ok=True)
    return targets, t_facial_masks, Ms, original_frames, names, fps


def swap_image_gr(img1, img2, use_post=False, use_gpen=False, gpu_mode=True):
    root_dir = make_abs_path("./online_data")
    req_id = uuid.uuid1().hex
    data_dir = os.path.join(root_dir, req_id)
    os.makedirs(data_dir, exist_ok=True)
    source_path = os.path.join(data_dir, "source.png")
    target_path = os.path.join(data_dir, "target.png")
    filename = "paste_back_out_target.png"
    out_path = os.path.join(data_dir, filename)
    cv2.imwrite(source_path, img1[:, :, ::-1])
    cv2.imwrite(target_path, img2[:, :, ::-1])
    swap_image(
        source_path,
        target_path,
        data_dir,
        T,
        fs_model,
        gpu_mode=gpu_mode,
        align_target='ffhq',
        align_source='ffhq',
        use_post=use_post,
        use_gpen=use_gpen,
        in_size=in_size,
    )
    out = cv2.imread(out_path)[..., ::-1]
    return out


def swap_video_gr(img1, target_path, use_gpu=True, frames=9999999):
    root_dir = make_abs_path("./online_data")
    req_id = uuid.uuid1().hex
    data_dir = os.path.join(root_dir, req_id)
    os.makedirs(data_dir, exist_ok=True)
    source_path = os.path.join(data_dir, "source.png")
    cv2.imwrite(source_path, img1[:, :, ::-1])
    out_dir = os.path.join(data_dir, "out")
    out_name = "output.mp4"
    targets, t_facial_masks, Ms, original_frames, names, fps = process_video(
        source_path,
        target_path,
        out_dir,
        T,
        fs_model,
        gpu_mode=use_gpu,
        frames=frames,
        align_target='ffhq',
        align_source='ffhq',
        use_tddfav2=False,
    )

    pool_process = 170
    audio = True
    concat = False

    if pool_process <= 1:
        for target, M, original_target, name, t_facial_mask in tqdm.tqdm(
                zip(targets, Ms, original_frames, names, t_facial_masks)
        ):
            if M is None or target is None:
                Image.fromarray(original_target.astype(np.uint8)).save(name)
                continue
            Image.fromarray(paste_back(np.array(target), M, original_target, t_facial_mask)).save(name)
    else:
        with Pool(pool_process) as pool:
            pool.map(save, zip(targets, Ms, original_frames, names, t_facial_masks))

    video_save_path = os.path.join(out_dir, out_name)
    if audio:
        print("use audio")
        os.system(
            f"ffmpeg  -y -r {fps} -i {out_dir}/frame_%05d.png -i {target_path}"
            f" -map 0:v:0 -map 1:a:0? -c:a copy -c:v libx264 -r {fps} -crf 10 -pix_fmt yuv420p  {video_save_path}"
        )
    else:
        print("no audio")
        os.system(
            f"ffmpeg  -y -r {fps} -i ./tmp/frame_%05d.png "
            f"-c:v libx264 -r {fps} -crf 10 -pix_fmt yuv420p {video_save_path}"
        )
    # ffmpeg -i left.mp4 -i right.mp4 -filter_complex hstack output.mp4
    if concat:
        concat_video_save_path = os.path.join(out_dir, "concat_" + out_name)
        os.system(
            f"ffmpeg -y  -i {target_path}  -i {video_save_path} -filter_complex hstack {concat_video_save_path}"
        )
    # delete tmp file
    shutil.rmtree("./tmp/")
    for match in glob.glob(os.path.join(out_dir, "*.png")):
        os.remove(match)
    print(video_save_path)
    return video_save_path


if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("SuperSwap")

        with gr.Tab("Image"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=3):
                    image1_input = gr.Image()
                    image2_input = gr.Image()
                    use_post = gr.Checkbox(label="后处理")
                    use_gpen = gr.Checkbox(label="超分增强")
                with gr.Column(scale=2):
                    image_output = gr.Image()
                    image_button = gr.Button("换脸")
        with gr.Tab("Video"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=3):
                    image3_input = gr.Image()
                    video_input = gr.Video()
                with gr.Column(scale=2):
                    video_output = gr.Video()
                    video_button = gr.Button("换脸")
        image_button.click(
            swap_image_gr,
            inputs=[image1_input, image2_input, use_post, use_gpen],
            outputs=image_output,
        )
        video_button.click(
            swap_video_gr,
            inputs=[image3_input, video_input],
            outputs=video_output,
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)
