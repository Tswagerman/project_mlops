import torch
from transformers import BertForSequenceClassification

# Define the model
class FakeRealClassifier(torch.nn.Module):
    def __init__(self, pretrained_model_name='bert-base-cased', num_labels=2):
        super(FakeRealClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=num_labels)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask=attention_mask)
        return output.logits