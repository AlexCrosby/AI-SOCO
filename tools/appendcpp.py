import argparse
import os


# This script is used to append and remove the '.cpp' file extension from the data. This is because for some parsers,
# '.cpp' is required for files to be identified as code.

def main(args):
    train_dir = os.path.join(args.data, 'train')
    dev_dir = os.path.join(args.data, 'dev')
    test_dir = os.path.join(args.data, 'test')
    dirs = [train_dir, dev_dir, test_dir]
    if not args.remove:
        for d in dirs:
            files = os.listdir(d)
            for file in files:
                if '.cpp' in file:
                    continue
                old_file = os.path.join(d, file)
                new_name = file + '.cpp'
                new_file = os.path.join(d, new_name)
                os.rename(old_file, new_file)
    elif args.remove:
        for d in dirs:
            files = os.listdir(d)
            for file in files:
                if '.cpp' not in file:
                    continue
                old_file = os.path.join(d, file)
                new_name = file.replace('.cpp', '')
                new_file = os.path.join(d, new_name)
                os.rename(old_file, new_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('-d',
                        '--data',
                        type=str,
                        default=r'../data_dir/pre',
                        help='Path to the data directory.')
    parser.add_argument('-r',
                        '--remove',
                        help='Remove with remove the .cpp extension. Default = False',
                        action='store_true')
    args = parser.parse_args()
    print(args)
    main(args)
