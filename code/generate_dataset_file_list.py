import os, argparse
import utils
import numpy as np


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

    with open('output/filelist.txt', 'w') as file:

        for root, directories, filenames in os.walk(data_dir):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    filename = ('.').join(filename.split('.')[:-1])
                    # load the .mat data and filer the data
                    mat_path = os.path.join(root, filename + '.mat')
                    # We get the pose in radians
                    pose = utils.get_ypr_from_mat(mat_path)
                    # And convert to degrees.
                    pitch = pose[0] * 180 / np.pi
                    yaw = pose[1] * 180 / np.pi
                    roll = pose[2] * 180 / np.pi

                    if yaw < -99 or yaw > 99:
                        print("ignore file: " + filename)
                        continue

                    if pitch < -99 or pitch > 99:
                        print("ignore file: " + filename)
                        continue

                    if roll < -99 or roll > 99:
                        print("ignore file: " + filename)
                        continue

                    file.write(os.path.relpath(os.path.join(root, filename),
                                               data_dir))
                    file.write('\n')

    file.close()


if __name__ == '__main__':
    main()
