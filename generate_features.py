import argparse
import logging
import os
import pprint

from features.config import config, update_config
from features.video.extract import extract_video_features


def parse_args():

    parser = argparse.ArgumentParser(description='Generate Features')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)

    args = parser.parse_args()
    update_config(config, args)
    return args


def main():

    args = parse_args()
    head = '%(asctime)-15s %(message)s'
    os.makedirs(os.path.join('features/runs', 'logs'), exist_ok=True)
    logging.basicConfig(filename=os.path.join('features/runs', 'logs', os.path.basename(args.cfg[:-5]) + '.txt'), format=head) # todo proper logging path

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    extract_video_features(config, logger)


if __name__ == '__main__':
    main()
