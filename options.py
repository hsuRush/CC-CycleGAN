from argparse import ArgumentParser
from argparse import ArgumentTypeError
import logging
import os

import numpy as np
import common

def train_options():
    parser = ArgumentParser(description='parser for cond-cyclegan model')
    parser.add_argument(
        '--model', type=str, default='resnet18_2222_16',
        help='select pretrained model: '
            '  resnet50'
            '  resnet18'
        )
    parser.add_argument(
        '--stn', action='store_true', default=False,
        help='use stn net')

    parser.add_argument(
        '--ctc_weights', type=str, default='./lpGen_exp/1_exp1/best-model.h5',
        help='weight file active at resume')

    parser.add_argument(
        '--ctc_resume', action='store_true', default=False,
        help='resume')

    parser.add_argument(
        '--ctc_condition', action='store_true', default=False,
        help='ctc_condition')

    parser.add_argument(
        '--test', action='store_true', default=False,
        help='test or train')

    parser.add_argument(
        '--batch', type=common.positive_int, default=16,
        help='test or train')

    parser.add_argument(
        '--epoch', type=common.positive_int, default=100,
        help='test or train')
    parser.add_argument(
        '--exp_dir', type=str, default='exp1',
        help='where weights saved')
    parser.add_argument(
        '--dataset', type=str, default='lpgen2aolp',
        help='which dataset to use')
    
    args = parser.parse_args()
    return args


def test_options():
    parser = ArgumentParser(description='parser for cond-cyclegan model')
    parser.add_argument(
        '--model', type=str, default='resnet18_2222_16',
        help='select pretrained model: '
            '  resnet50'
            '  resnet18'
        )
    parser.add_argument(
        '--stn', action='store_true', default=False,
        help='use stn net')

    parser.add_argument(
        '--resume_epoch', type=int, default=245,
        help='resume which weight file by epoch')

    parser.add_argument(
        '--ctc_condition', action='store_true', default=False,
        help='ctc_condition')

    parser.add_argument(
        '--batch', type=common.positive_int, default=16,
        help='test or train')

    parser.add_argument(
        '--iteration', type=int, default=64,
        help='if 0 means test_all_of_dataset , test how many times per "batchsize"')

    parser.add_argument(
        '--test', action='store_true', default=True,
        help='test or train')
    parser.add_argument(
        '--set', type=str, default='test',
        help='use which set to test')

    parser.add_argument(
        '--exp_dir', type=str, default='exp1',
        help='where weights saved')

    parser.add_argument(
        '--dataset', type=str, default='lpgen2aolp',
        help='which dataset to use')

    parser.add_argument(
        '--verbose', action='store_true', default=False,
        help='print lots of stuffs')

    parser.add_argument(
        '--direction',  type=str, default='both',
        help='choose: A2B, B2A, both')
    args = parser.parse_args()
    return args