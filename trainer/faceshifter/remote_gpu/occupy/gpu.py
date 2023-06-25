import torch
import time
import os
import argparse
import shutil
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='Matrix multiplication')
    parser.add_argument('--gpus', help='gpu amount', default=-1, type=int)
    parser.add_argument('--size', help='matrix size', default=20000, type=int)
    parser.add_argument('--interval', help='sleep interval', default=0.005, type=float)
    parser.add_argument('--minute', help='total time (minutes)', default=500, type=int)
    args = parser.parse_args()
    return args


def matrix_multiplication(args):
    a_list, b_list, result = [], [], []
    size = (args.size, args.size)

    for i in range(args.gpus):
        a_list.append(torch.rand(size, device=i))
        b_list.append(torch.rand(size, device=i))
        result.append(torch.rand(size, device=i))

    start = time.time()
    elapse = 0
    old_time_remaining = 0
    while elapse < args.minute * 60:
        for i in range(args.gpus):
            result[i] = a_list[i] * b_list[i]
        time.sleep(args.interval)

        elapse = time.time() - start
        time_remaining = int(args.minute * 60 - elapse)  # seconds
        if time_remaining % 60 == 0 and time_remaining != old_time_remaining:
            print('Time remaining: %d minutes' % (time_remaining // 60))
            old_time_remaining = time_remaining


if __name__ == "__main__":
    # usage: python matrix_multiplication_gpus.py --size 20000 --gpus 2 --interval 0.01
    args = parse_args()
    if args.gpus == -1:
        args.gpus = torch.cuda.device_count()
    matrix_multiplication(args)
