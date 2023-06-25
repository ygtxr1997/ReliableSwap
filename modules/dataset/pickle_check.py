import argparse
import pickle


def check(pickle_path: str):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    print('[data type]: %s, [data len]: %d' % (type(data), len(data)))
    print('[item 0]:', data[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_path', type=str, help='/gavin/datasets/triplet_xxx.pickle')
    args = parser.parse_args()

    check(args.pickle_path)
