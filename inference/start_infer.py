import argparse
import pytorch_lightning as pl

from inference.infer_tools import get_test_interator


def get_cmd_args():
    parser = argparse.ArgumentParser(description="benchmark evaluation")
    parser.add_argument("-m", "--models", type=str,
                        help="model names seperated by \',\'.")
    parser.add_argument("-i", "--in_folder", type=str, help='input folder path')
    parser.add_argument("-o", "--out_folder", type=str, help='output folder path')

    parser.add_argument("-bs", "--batch_size", type=int,
                        default=1, help='testing batch size')

    args = parser.parse_args()

    args.models = args.models.split(",")

    return args


def get_iterators(args):
    model_names = args.models
    testers = []
    for name in model_names:
        test_iterator = get_test_interator(
            name, args.in_folder, args.out_folder, args.batch_size
        )
        testers.append(test_iterator)
    return testers


if __name__ == "__main__":
    """
    python start_infer.py -m faceshifter,reliable_faceshifter,simswap,reliable_simswap,infoswap,hires,megafs  \
        -i /home/yuange/program/PyCharmRemote/e4s_v2/outputs  \
        -o /home/yuange/Documents/E4S_v2/sota_method_results
    """
    args = get_cmd_args()
    testers = get_iterators(args)
    pl_tester = pl.Trainer(
        logger=False,
        gpus=1,
        strategy='dp',
        benchmark=True,
    )
    for tester in testers:
        pl_tester.test(tester)
