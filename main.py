from lm_dataset import LMDataset
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = LMDataset.add_model_specific_args(ArgumentParser())
    h_params = parser.parse_args()
    ds = LMDataset(h_params)
    _ = ds[0]
    x = 0
