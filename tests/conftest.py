import csv
import tempfile
from pathlib import Path

import pytest

from bert.config import ClassificationDataArguments, ClassificationModelArguments
from bert.train import build_model, build_tokenizer
from gpt2.config import ClmDataArguments, ClmModelArguments
from gpt2.tokenizer import load_tokenizer
from gpt2.train import build_model as build_gpt2_model
from t5.config import SummarizationDataArguments, SummarizationModelArguments
from t5.train import build_model as build_t5_model
from t5.train import build_tokenizer as build_t5_tokenizer


TINY_BERT = "hf-internal-testing/tiny-random-BertModel"
TINY_T5 = "hf-internal-testing/tiny-random-T5ForConditionalGeneration"
TINY_GPT2 = "hf-internal-testing/tiny-random-GPT2LMHeadModel"

GUARDIAN_COLUMNS = ["Time", "Headlines"]

GUARDIAN_ROWS = [
    ("18-Jul-20", "What will changes to England's lockdown rules mean for me?"),
    ("17-Jul-20", "Johnson's coronavirus workplace guidance confused, warn unions"),
    ("17-Jul-20", "Lloyds to increase number of black staff in senior roles"),
    ("17-Jul-20", "Boohoo shares rise 12% after co-founders pump money into firm"),
    ("16-Jul-20", "Royalties investment firm buys catalogue of Lady Gaga collaborator"),
    ("16-Jul-20", "Next in final stages of buying UK arm of Victoria's Secret"),
    ("16-Jul-20", "UK's hidden problem of rising unemployment may soon be exposed"),
    ("15-Jul-20", "UK inflation rises as game console prices increase in lockdown"),
    ("15-Jul-20", "Governments put 'green recovery' on the backburner"),
    ("15-Jul-20", "The green recovery  UK government planning new green investment bank"),
    ("15-Jul-20", "Treasury forecaster's three stark predictions for Britain's economy"),
    ("14-Jul-20", "UK's expected U-turn on Huawei fails to satisfy Tory rebels"),
    ("18-Jul-20", "Johnson is asking Santa for a Christmas recovery"),
    ("18-Jul-20", "English councils call for smoking ban outside pubs and cafes"),
    ("18-Jul-20", "Can Tesla justify a valuation of three hundred billion?"),
    ("18-Jul-20", "Atol protection to be extended to vouchers on Covid-19 cancellations"),
    ("18-Jul-20", "Ask and Zizzi to close 75 outlets, threatening up to 1,200 jobs"),
    ("18-Jul-20", "World Bank calls on creditors to cut poorest nations' debt payments"),
    ("18-Jul-20", "British Airways retires Boeing 747 fleet as Covid-19 hits travel"),
    ("17-Jul-20", "BA begins to carry out its 'fire and rehire' threat to jobs"),
    ("17-Jul-20", "What will be changing in Boris Johnson's 'return to normality'"),
    ("17-Jul-20", "A fifth of Brazilian soy in Europe is result of deforestation"),
    ("17-Jul-20", "Public sector pension discrimination could cost UK taxpayer billions"),
    ("16-Jul-20", "Canary Wharf traders and landlord bank on return to offices"),
    ("18-Jul-20", "Five key areas Sunak must tackle to serve up economic recovery"),
    ("18-Jul-20", "British Airways retires iconic Boeing 747 fleet in pictures"),
    ("17-Jul-20", "In search of a new economics for Covid-19 era"),
    ("17-Jul-20", "Battery firm chooses Welsh site for Britain's first gigafactory"),
    ("17-Jul-20", "Netflix shares drop despite positive second-quarter earnings"),
    ("16-Jul-20", "More than 1,600 UK jobs at risk at casino firm Genting"),
    ("16-Jul-20", "More people file for unemployment as US economy continues to reel"),
    ("15-Jul-20", "Asos considers action against Leicester supplier in ethical audit"),
    ("15-Jul-20", "UK launches first online service for groceries in reusable packing"),
    ("15-Jul-20", "Virgin Atlantic: a successful non-intervention by the Treasury"),
    ("14-Jul-20", "UK's growth figures dim hopes of V-shaped recovery from Covid-19"),
    ("14-Jul-20", "Face masks 'will deter young shoppers' says JD Sports chairman"),
]

REUTERS_COLUMNS = ["Headlines", "Time", "Description"]

REUTERS_ROWS = [
    ("Google bars ads on virus misinformation", "Jul 18 2020", "Ads may not run against claims that defy science."),
    ("Facebook executives face antitrust questions", "Jul 18 2020", "Executives were subpoenaed in a monopoly probe."),
    ("Former Pemex boss extradited to Mexico", "Jul 18 2020", "He faces corruption charges after leaving Spain."),
    ("TikTok considers London for headquarters", "Jul 18 2020", "Talks with the UK government ran for months."),
    ("Disney cuts ad spending on Facebook", "Jul 18 2020", "The studio joins a boycott over hate speech."),
    ("Bank of England warns of slower recovery", "Jul 19 2020", "Output will regain its old size later than hoped."),
    ("Oil prices steady as demand worries persist", "Jul 19 2020", "Supply cuts offset fears of weaker fuel use."),
    ("Airlines call for scrapping quarantine", "Jul 20 2020", "Carriers want airport testing to replace it."),
    ("Retail sales rebound but stay below normal", "Jul 20 2020", "Shops reopened yet takings trail last year."),
    ("Chipmaker shares slide after weak outlook", "Jul 21 2020", "Guidance missed estimates on factory delays."),
    ("Carmaker to cut thousands of jobs", "Jul 21 2020", "Restructuring follows a collapse in sales."),
    ("Trade talks stall over fishing and state aid", "Jul 22 2020", "Both sides report wide gaps on quotas."),
]


def write_csv(columns: list[str], rows: list[tuple]) -> Path:
    """Write rows to a temporary CSV and return its path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)

        return Path(f.name)


@pytest.fixture
def temp_guardian_file():
    """Guardian CSV shaped like the real one: headlines with no label column."""
    path = write_csv(GUARDIAN_COLUMNS, GUARDIAN_ROWS)

    yield path

    path.unlink(missing_ok=True)


@pytest.fixture
def temp_reuters_file():
    """Reuters CSV carrying headlines, timestamps and article descriptions."""
    path = write_csv(REUTERS_COLUMNS, REUTERS_ROWS)

    yield path

    path.unlink(missing_ok=True)


@pytest.fixture(scope="session")
def bert_model_args():
    """Model arguments pointing at the tiny BERT checkpoint."""
    return ClassificationModelArguments(model_name_or_path=TINY_BERT)


@pytest.fixture(scope="session")
def bert_tokenizer(bert_model_args):
    """Tokenizer for the tiny BERT checkpoint, loaded once per run."""
    return build_tokenizer(bert_model_args)


@pytest.fixture(scope="session")
def bert_model(bert_model_args):
    """Untrained classification model sized to the three sentiment labels."""
    return build_model(bert_model_args)


@pytest.fixture
def bert_data_args(temp_guardian_file):
    """Data arguments reading the temporary Guardian CSV."""
    return ClassificationDataArguments(data_path=str(temp_guardian_file), max_length=32)


@pytest.fixture(scope="session")
def t5_model_args():
    """Model arguments pointing at the tiny T5 checkpoint."""
    return SummarizationModelArguments(model_name_or_path=TINY_T5)


@pytest.fixture(scope="session")
def t5_tokenizer(t5_model_args):
    """Tokenizer for the tiny T5 checkpoint, loaded once per run."""
    return build_t5_tokenizer(t5_model_args)


@pytest.fixture(scope="session")
def t5_model(t5_model_args):
    """Untrained summarization model, loaded once per run."""
    return build_t5_model(t5_model_args)


@pytest.fixture
def t5_data_args(temp_reuters_file):
    """Data arguments reading the temporary Reuters CSV."""
    return SummarizationDataArguments(
        data_path=str(temp_reuters_file),
        max_source_length=32,
        max_target_length=16,
    )


@pytest.fixture(scope="session")
def gpt2_model_args():
    """Model arguments pointing at the tiny GPT-2 checkpoint."""
    return ClmModelArguments(model_name_or_path=TINY_GPT2)


@pytest.fixture(scope="session")
def gpt2_tokenizer(gpt2_model_args):
    """Tokenizer for the tiny GPT-2 checkpoint, carrying its own pad token."""
    return load_tokenizer(gpt2_model_args)


@pytest.fixture(scope="session")
def gpt2_model(gpt2_model_args, gpt2_tokenizer):
    """Untrained causal model sized to the padded tokenizer."""
    return build_gpt2_model(gpt2_model_args, gpt2_tokenizer)


@pytest.fixture
def gpt2_data_args(temp_reuters_file):
    """Data arguments reading the temporary Reuters CSV."""
    return ClmDataArguments(data_path=str(temp_reuters_file), max_length=32)
