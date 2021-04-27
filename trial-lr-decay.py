import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = args_parser()
    for round in range(1,11):
        print(f'round {round}, learning rate: {args.lr}')
        args.lr *= 0.99