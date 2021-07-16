from transformers import T5ForConditionalGeneration, T5TokenizerFast
from torch import nn, Tensor


class KeyWordGeneration(nn.Module):

    def __init__(self):
        super(KeyWordGeneration, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small")

    def forward(self, input_ids: Tensor, attention_masks: Tensor, labels: Tensor = None):
        """
        the forward in case of training, created another function for testing
        :param input_ids: the IDs of the input. from the tokenizer. should be of shape batch X sequence
        :param attention_masks: the masks of the input. from the tokenizer. should be of shape batch X sequence
        :param labels: the labels for the model. if given, will calculate the loss. should be of shape batch X sequence
        :return: if given labels, returns predictions (logits) and loss. if labels are not given, returns only logits.
        logits are of shape batch X sequence X vocab size
        """
        out_model = self.model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
        logits = out_model["logits"]
        if labels is None:
            return logits
        return logits, out_model["loss"]

    def eval_forward(self, input_ids: Tensor, attention_masks: Tensor):
        """
        A forward pass for the eval, doesn't calculate the loss, but gives a better output
        :param input_ids: same as in forward
        :param attention_masks: same as in forward
        :return: prediction of shape batch X sequence
        """
        out_model = self.model.generate(input_ids=input_ids, attention_mask=attention_masks,
                                        max_length=input_ids.shape[1], min_length=input_ids.shape[1])
        return out_model


if __name__ == '__main__':
    model = KeyWordGeneration()
    sen = "Hello World"
    labels = "How are you doing?"
    tokenizer = T5TokenizerFast.from_pretrained("t5-small")
    model_in = tokenizer.batch_encode_plus([sen], return_tensors="pt", max_length=10, padding="max_length")
    model_labels = tokenizer.batch_encode_plus([labels], return_tensors="pt", max_length=10, padding="max_length")
    model.eval_forward(model_in["input_ids"], model_in["attention_mask"])
