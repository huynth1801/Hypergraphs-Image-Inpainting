import argparse
import os
import time

class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser ()
        self.initialized = False

    def initialized(self):
        self.parser.add_argument('--dataset', type=str, default='celeba-hq', help="Dataset name for training")
        self.parser.add_argument('--train_dir', type=str, default='', help="directory where all images are stored")
        self.parser.add_argument('--train_file_path', type=str, default='', help='The file storing the names of the file for training (If not provided training will happen for all images in train_dir)')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='GPU to be used e.g. 0, 1, 2')
        