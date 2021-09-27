# from pytorch_lightning import Trainer
# from argparse import ArgumentParser
# from lm_model import T5ForSentenceClassification, LoggingCallback
# from dataset_utils import BaseDataset
# from lm_dataset import LMDataset
# from sentence_pairs_dataset import PairsDS
import os
# from utils import set_seed
import pandas as pd
from matplotlib import pyplot as plt


def get_last_version(folder):
    files = list(os.listdir(folder))
    files = [(int(file.split("_")[1]), file) for file in files]
    files = sorted(files, reverse=True)[0][1]
    cp_file = list(os.listdir(f"{folder}{files}/checkpoints"))
    if len(cp_file) != 1:
        raise NotImplementedError
    cp_file = cp_file[0]
    cp_file = f"{folder}{files}/checkpoints/{cp_file}"
    return cp_file


def get_trainer(folder_name):
    parser = ArgumentParser()
    parser = BaseDataset.add_model_specific_args(parser)
    # parser = Trainer.add_argparse_args(parser)
    h_params = parser.parse_args()
    ds = PairsDS(h_params, h_params.train_path)

    trainer_params = {"gpus": h_params.gpus, "num_sanity_val_steps": 5, "min_epochs": h_params.num_train_epochs,
                      "max_epochs": h_params.num_train_epochs, "callbacks": [LoggingCallback()],
                      "default_root_dir": f"trained_models/{folder_name}",
                      "accumulate_grad_batches": h_params.gradient_accumulation_steps}
    trainer = Trainer(**trainer_params)
    return trainer, h_params


def do_single_run(seed=42, folder_name=""):
    set_seed(seed)

    trainer, h_params = get_trainer(folder_name)
    h_params.output_dir = f"results/{folder_name}"
    model = T5ForSentenceClassification(h_params)
    trainer.fit(model)
    logs_folder = f"trained_models/{folder_name}/lightning_logs/"
    model_loc = get_last_version(logs_folder)

    model = T5ForSentenceClassification.load_from_checkpoint(model_loc, h_params=h_params)

    trainer.test(model, test_dataloaders=model.data_loaders["dev"])
    dev_loss = model.cur_loss
    return dev_loss


def get_score_graph(score, score_name, axis):
    scores = []
    for epoch in range(10):
        epoch_file = f'results/model_results_with_evaluation_t5_base_epoch_{epoch}.csv'
        epoch_data = pd.read_csv(epoch_file)
        epoch_scores = float(epoch_data[score].mean())
        scores.append(epoch_scores)
    axis.plot(range(1, 11), scores)
    # axis.set(xlabel='Epochs', ylabel='Score', xlim=(0, 10), y_lim=(0, 1))
    axis.set_xlabel('Epoch')
    axis.set_ylabel(f'{score_name}')
    axis.set_xlim(1, 10)
    axis.set_ylim(0.2, 0.8)
    axis.set_title(score_name)
    # axis.y_lim(0, 1)
    # axis.x_lim(0, 10)
    axis.label_outer()


def get_all_score_graphs():
    scores = [f'score_{x}' for x in range(5)]
    scores.remove('score_2')
    scores_names = ['Grammar', 'Repetition', 'Focus', 'Structure']
    fig, ax = plt.subplots(1, 4)
    for idx, (score, score_name) in enumerate(zip(scores, scores_names)):
        axis = ax[idx]
        get_score_graph(score, score_name, axis)
    fig.set_size_inches(15, 5)
    fig.tight_layout()
    fig.savefig('results/scores.jpg')
    plt.show()


if __name__ == '__main__':
    os.environ['KMP_WARNINGS'] = 'off'
    # do_single_run(folder_name="testing")
    get_all_score_graphs()
