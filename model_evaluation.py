# scr/examples.py
import nltk
import numpy as np
from typing import List

from keras.models import load_model
from nltk.tokenize import sent_tokenize

from src.BERT_experiments.BERT_model import BERT, custom_loss, set_quality_index
from src.BERT_experiments.BERT_model import set_quality_index
from src.vectorizer import BERTVectorizer
import warnings
warnings.filterwarnings("ignore")

MODE = 'Multi Task-5'
QUALITY = 'Q5'
YEAR = 'all'

MODEL_PATH = './sumqe/BERT_DUC_{}_{}_{}.h5'.format(YEAR, QUALITY, MODE)

METRICS_WEIGHTS = np.array([0.15, 0.15, 0.15, 0.05, 0.5])


def init_evaluator():
    nltk.download('punkt')

    # Set the quality index used in custom_loss
    set_quality_index(mode=MODE, quality=QUALITY)

    # Load the model
    model = load_model(MODEL_PATH, custom_objects={'BERT': BERT, 'custom_loss': custom_loss})

    # Define the vectorizer
    vectorizer = BERTVectorizer()
    return model, vectorizer


def get_scores_single_sentence(sentence: str, model, vectorizer):
    token_ids = []
    for i, sen in enumerate(sent_tokenize(sentence)):
        sentence_tok = vectorizer.vectorize_inputs(sequence=sen, i=i)
        token_ids = token_ids + sentence_tok

    # Transform the summary_tokens_ids into inputs --> (bpe_ids, mask, segments)
    inputs = vectorizer.transform_to_inputs(token_ids)

    # Construct the dict that you will feed on your network. If you have multiple summaries,
    # you can update the lists and feed all of them together.
    test_dict = {
        'word_inputs': np.asarray([inputs[0, 0]]),
        'pos_inputs': np.asarray([inputs[1, 0]]),
        'seg_inputs': np.asarray([inputs[2, 0]])
    }

    output = model.predict(test_dict, batch_size=1)

    score = output @ METRICS_WEIGHTS
    score = float(score)
    return score


def get_scores(sentences, model, vectorizer):
    scores = []
    for sen in sentences:
        score = get_scores_single_sentence(sen, model, vectorizer)
        scores.append(score)
    return scores


if __name__ == '__main__':
    text = [
        "I love apples",
        "Bob are good",
        "please don't suck",
        "targeted killing is an invasive and dangerous method that can eliminate at least one enemy person. it is a "
        "good thing. it should be allowed. "
        "polygamy is a place for wives to decide whether to use it. it is a good idea to have a husband in the world."
    ]
    bert, vec = init_evaluator()
    print("created model")
    get_scores(text, bert, vec)
