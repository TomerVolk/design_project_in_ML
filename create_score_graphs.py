import pandas as pd
from matplotlib import pyplot as plt


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
    get_all_score_graphs()
