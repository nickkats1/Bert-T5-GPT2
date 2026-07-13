import pytest
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

from headlines.bert.model import BertClassifier


@pytest.fixture
def bert_classifier():
    return BertClassifier()


@pytest.fixture
def bert_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")


class TestBertModel:
    """Test for bert model"""

    def test_attributes(self, bert_classifier):
        """test attributes of bert model"""

        assert hasattr(bert_classifier, "bert")
        assert hasattr(bert_classifier, "drop")
        assert hasattr(bert_classifier, "out")
        assert hasattr(bert_classifier, "forward")

    def test_bert_model_is_loaded(self, bert_classifier):
        """test that BertModel from transformers is loaded when called"""

        assert isinstance(bert_classifier.bert, BertModel)

    def test_dropout(self, bert_classifier):
        """test dropout is set"""
        assert isinstance(bert_classifier.drop, nn.Dropout)
        assert bert_classifier.drop.p == 0.1

    def test_output(self, bert_classifier):
        """test output is set"""
        assert isinstance(bert_classifier.out, nn.Linear)
        assert bert_classifier.out.in_features == 768
        assert bert_classifier.out.out_features == 3

    def test_forward(self, bert_classifier, bert_tokenizer):
        """test forward returns correct output shape"""

        bert_classifier.eval()

        inputs = bert_tokenizer("market rally on earnings says crawford", return_tensors="pt")

        with torch.no_grad():
            output = bert_classifier(inputs["input_ids"], inputs["attention_mask"])
            assert output.shape == (1, 3)

    def test_model_to_device(self, bert_classifier):
        """test if bert model is moved to device"""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        bert_classifier.to(device)

        for param in bert_classifier.parameters():
            assert param.device == device
