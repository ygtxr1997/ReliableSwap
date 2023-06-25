import torch
import argparse
import yaml
from trainer.faceshifter.faceshifter_pl import FaceshifterPL


def extract_generator(
    load_path="/gavin/code/FaceSwapping/trainer/faceshifter/trainer/out/hello/epoch=6-step=98999.ckpt",
    path="./extracted_ckpt/G_v2.pth",
    n_layers=3,
    num_D=3,
):
    with open('/gavin/code/FaceSwapping/trainer/faceshifter/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    net = FaceshifterPL(n_layers=n_layers, num_D=num_D, config=config)
    checkpoint = torch.load(
        load_path,
        map_location="cpu",
    )
    net.load_state_dict(checkpoint["state_dict"], strict=False)
    net.eval()

    G = net.generator
    torch.save(G.state_dict(), path)

    return checkpoint["state_dict"]


def load_extracted(path="./extracted_ckpt/G_test.pt"):
    net = FaceshifterPL()
    G = net.generator
    G.load_state_dict(torch.load(path, "cpu"))
    G.eval()
    return G


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="face swap")
    parser.add_argument("-o", "--out", type=str, required=True)
    parser.add_argument("-nd", "--num_D", type=int, default=3)
    parser.add_argument("-nl", "--n_layers", type=int, default=3)
    parser.add_argument(
        "-p",
        "--ckpt_path",
        type=str,
        default="/gavin/code/FaceSwapping/trainer/faceshifter/trainer/out/hello/epoch=6-step=98999.ckpt",
    )
    parser.add_argument("--v2", dest="use_v2", action="store_true")
    parser.add_argument("--no-v2", dest="use_v2", action="store_false")
    parser.set_defaults(use_v2=False)
    args = parser.parse_args()

    extract_generator(
        args.ckpt_path,
        path=args.out,
        n_layers=args.n_layers,
        num_D=args.num_D,
    )
    # python3 extract_ckpt.py -p out/hello/epoch-xxxx -o extracted_ckpt/G_yg_vxx.pth
