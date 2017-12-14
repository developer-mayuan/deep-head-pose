import os, argparse


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Generate Image File List for a given Dataset.')
    parser.add_argument('--data_dir', dest='data_dir',
                        help='Directory path for data.',
                        default='', type=str)
    parser.add_argument('--output_string', dest='output_string',
                        help='String appended to output snapshots.',
                        default='', type=str)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    data_dir = args.data_dir
    file_list = []

    with open('output/filelist.txt', 'w') as file:

        for root, directories, filenames in os.walk(data_dir):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    filename = ('.').join(filename.split('.')[:-1])
                    file.write(os.path.relpath(os.path.join(root,
                                                            filename),
                                               data_dir))
                    file.write('\n')

    print len(file_list)

    file.close()


if __name__ == '__main__':
    main()
