from dataset_utils import BaseDataset
from argparse import Namespace
import csv


class PairsDS(BaseDataset):

    def __init__(self, h_params: Namespace, file_path):
        self.winners = []
        self.losers = []
        super(PairsDS, self).__init__(h_params, file_path)
        self.winner_ids, self.winner_masks = self.preprocess(self.winners, True)
        self.loser_ids, self.loser_masks = self.preprocess(self.losers, True)

    def read_file(self, file_path):
        with open(file_path, newline='', encoding="UTF-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
            for idx, row in enumerate(csv_reader):
                if idx == 0:
                    continue
                winner, loser, _ = row
                self.winners.append(winner)
                self.losers.append(loser)

    def __getitem__(self, item):
        winner_id = self.winner_ids[item]
        winner_mask = self.winner_masks[item]
        loser_id = self.loser_ids[item]
        loser_mask = self.loser_masks[item]
        return winner_id, winner_mask, loser_id, loser_mask

    def __len__(self):
        return len(self.winners)

