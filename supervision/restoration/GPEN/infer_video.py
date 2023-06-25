import os
import cv2
import numpy as np
import glob
import tqdm
import shutil
import argparse
from face_enhancement import FaceEnhancement


def process_video(target_path, out_path, faceenhancer):
    fps = 25.0
    os.makedirs(out_path, exist_ok=True)
    original_vid_path = target_path
    vid_name = "out.mp4"
    if not os.path.isdir(target_path):
        vid_name = target_path.split("/")[-1]
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
    else:
        print("folder not implemented.")
        exit()

    globbed_images = sorted(glob.glob(os.path.join(target_path, "*.png")))
    for image in tqdm.tqdm(globbed_images):
        name = image.split("/")[-1]
        filename = os.path.join(out_path, name)
        im = cv2.imread(image, cv2.IMREAD_COLOR)  # BGR
        h, w, _ = im.shape
        # im = cv2.resize(im, (0,0), fx=2, fy=2) #optional
        img, orig_faces, enhanced_faces = faceenhancer.process(im)
        img = cv2.resize(img, (w, h))
        cv2.imwrite(filename, img)

    # merge frames to video
    video_save_path = os.path.join(out_path, vid_name)

    os.system(
        f"ffmpeg  -y -r {fps} -i {out_path}/frame_%d.png -i {original_vid_path}"
        f" -map 0:v:0 -map 1:a? -c:a copy -c:v libx264 -r {fps} -pix_fmt yuv420p  {video_save_path}"
    )

    # delete tmp file
    shutil.rmtree("./tmp/")
    for match in glob.glob(os.path.join(out_path, "*.png")):
        os.remove(match)


if __name__ == "__main__":
    model = {
        "name": "GPEN-BFR-512",
        "in_size": 512,
        "out_size": 512,
        "channel_multiplier": 2,
        "narrow": 1,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, required=True, help="input file")
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Please provide output folder which has no more than one parent dir that has not been created.",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    faceenhancer = FaceEnhancement(
        use_sr=True,
        in_size=model["in_size"],
        out_size=model["out_size"],
        model=model["name"],
        channel_multiplier=model["channel_multiplier"],
        narrow=model["narrow"],
    )

    process_video(
        args.indir,
        args.outdir,
        faceenhancer,
    )
