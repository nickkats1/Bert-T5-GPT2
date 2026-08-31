import csv
import tempfile
from pathlib import Path

import pytest

from headlines.bert.train import build_model
from headlines.bert.train import build_tokenizer as build_bert_tokenizer
from headlines.gpt2.train import build_model as build_gpt2_model
from headlines.gpt2.train import build_tokenizer as build_gpt2_tokenizer
from headlines.t5.train import build_model as build_t5_model
from headlines.t5.train import build_tokenizer as build_t5_tokenizer


TINY_BERT = "hf-internal-testing/tiny-random-BertModel"
TINY_T5 = "hf-internal-testing/tiny-random-T5ForConditionalGeneration"
TINY_GPT2 = "hf-internal-testing/tiny-random-GPT2LMHeadModel"


@pytest.fixture(scope="session")
def bert_tokenizer():
    """Tokenizer for the tiny BERT checkpoint, loaded once per run."""
    return build_bert_tokenizer(TINY_BERT)


@pytest.fixture(scope="session")
def bert_label_maps():
    """Label lookups matching the labels used in GUARDIAN_ROWS."""
    labels = sorted({label for _, label in GUARDIAN_ROWS})

    return (
        {label: i for i, label in enumerate(labels)},
        dict(enumerate(labels)),
    )


@pytest.fixture(scope="session")
def bert_model(bert_label_maps):
    """Untrained classification model, loaded once per run."""
    label_to_id, id_to_label = bert_label_maps

    return build_model(label_to_id, id_to_label, TINY_BERT)


@pytest.fixture(scope="session")
def t5_tokenizer():
    """Tokenizer for the tiny T5 checkpoint, loaded once per run."""
    return build_t5_tokenizer(TINY_T5)


@pytest.fixture(scope="session")
def t5_model():
    """Untrained summarization model, loaded once per run."""
    return build_t5_model(TINY_T5)


@pytest.fixture(scope="session")
def gpt2_tokenizer():
    """Tokenizer for the tiny GPT-2 checkpoint, carrying its pad token."""
    return build_gpt2_tokenizer(TINY_GPT2)


@pytest.fixture(scope="session")
def gpt2_model(gpt2_tokenizer):
    """Untrained causal model sized to the padded tokenizer, loaded once per run."""
    return build_gpt2_model(gpt2_tokenizer, TINY_GPT2)


GUARDIAN_COLUMNS = ["Headlines", "labels"]

GUARDIAN_ROWS = [
    ("Johnson is asking Santa for a Christmas recovery", 1),
    ("‘I now fear the worst’: four grim tales of working life upended by Covid-19", 0),
    ("Five key areas Sunak must tackle to serve up economic recovery", 2),
    ("Covid-19 leaves firms ‘fatally ill-prepared’ for no-deal Brexit", 1),
    (
        "The Week in Patriarchy  \n\n\n  Bacardi's 'lady vodka': the latest in a long line of depressing gendered products",
        0,
    ),
    ("English councils call for smoking ban outside pubs and cafes", 1),
    ("Can Tesla justify a $300bn valuation?", 1),
    ("Empty city centres: 'I’m not sure it will ever be the same again'", 0),
    ("Democratising finance for all? An investment app for amateurs and a student trader's death", 1),
    ("Homebuyer loses £300,000 to fraudsters – but gets it back after we step in", 0),
    ("St Mawes named UK’s top seaside resort in Which? poll", 2),
    ("Atol protection to be extended to vouchers on Covid-19 cancellations", 1),
]


@pytest.fixture
def guardian_columns():
    """Column names of the temporary Guardian CSV."""
    return GUARDIAN_COLUMNS


@pytest.fixture
def guardian_rows():
    """Headline and label pairs written into the temporary Guardian CSV."""
    return GUARDIAN_ROWS


@pytest.fixture
def temp_guardian_file(guardian_columns, guardian_rows):
    """Create a temporary Guardian CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(guardian_columns)
        writer.writerows(guardian_rows)

        temp_path = Path(f.name)

    yield temp_path

    if temp_path.exists():
        temp_path.unlink()


REUTERS_COLUMNS = ["Headlines", "Time", "Description"]

REUTERS_ROWS = [
    (
        "St Mawes named UK's top seaside resort in Which? poll",
        "Jul 18 2020",
        "Alphabet Inc's Google said on Friday it would prohibit websites and apps that use its advertising "
        "technology from running ads on dangerous content that goes against scientific consensus during the "
        "coronavirus pandemic",
    ),
    (
        "Key areas Sunak must tackle to serve up economic recovery",
        "18-Jul-20",
        "Top executives Mark Zuckerberg and Sheryl Sandberg as a part of its probe into whether the company "
        "has engaged in unlawful monopolistic practices, the Wall Street Journal reported on Friday",
    ),
    (
        "Ask and Zizzi to close 75 outlets, threatening up to 1,200 jobs",
        "18-Jul-20",
        "A former boss of Mexico's state oil company Petroleos Mexicanos facing corruption charges that could "
        "envelop leaders of the last government was taken to a hospital early on Friday shortly after his "
        "overnight extradition to Mexico from Spain",
    ),
    (
        "TikTok considers London and other locations for headquarters",
        "Jul 18 2020",
        "TikTok has been in discussions with the UK government over the past few months to locate its "
        "headquarters in London, a source familiar with the matter said, as part of a strategy to distance "
        "itself from its Chinese ownership",
    ),
    (
        "Disney cuts ad spending on Facebook amid growing boycott",
        "Jul 18 2020",
        "Walt Disney has become the latest company to slash its advertising spending on Facebook Inc as the "
        "social media giant faces an ad boycott over its handling of hate speech and controversial content",
    ),
    (
        "Bank of England warns of slower recovery than hoped",
        "Jul 19 2020",
        "The Bank of England said the economy would take longer to return to its pre-pandemic size than it "
        "had forecast in May, pointing to weak business investment and a rise in precautionary saving among "
        "households",
    ),
    (
        "Oil prices steady as demand concerns offset supply cuts",
        "Jul 19 2020",
        "Oil prices held broadly steady on Sunday as worries that a resurgence in coronavirus infections "
        "would sap fuel demand balanced the effect of production cuts agreed by OPEC and its allies",
    ),
    (
        "Airline group calls for scrapping of quarantine rules",
        "Jul 20 2020",
        "An industry body representing European carriers urged governments to replace blanket quarantine "
        "requirements with airport testing, warning that bookings had stalled since the restrictions were "
        "reimposed",
    ),
    (
        "Retail sales rebound but remain below pre-crisis levels",
        "Jul 20 2020",
        "Retail sales rose for a second month in June as non-essential shops reopened, though the total "
        "value of goods sold stayed well short of where it stood before lockdown measures began",
    ),
    (
        "Chipmaker shares slide after weak quarterly outlook",
        "Jul 21 2020",
        "Shares in the semiconductor group fell in extended trading after it forecast quarterly revenue "
        "below analyst estimates, blaming delays to a new manufacturing process and softer data centre "
        "orders",
    ),
    (
        "Carmaker to cut thousands of jobs in restructuring",
        "Jul 21 2020",
        "The carmaker said it would reduce its global workforce as part of a restructuring plan aimed at "
        "cutting fixed costs, after a collapse in vehicle sales during the first half of the year",
    ),
    (
        "Trade talks stall over fishing rights and state aid",
        "Jul 22 2020",
        "Negotiators ended another round of talks without agreement, with both sides saying significant "
        "differences remained on fishing quotas and the rules governing subsidies to domestic industry",
    ),
]


@pytest.fixture
def reuters_columns():
    """Column names of the temporary Reuters CSV."""
    return REUTERS_COLUMNS


@pytest.fixture
def reuters_rows():
    """Headline, time, and description triples written into the temporary Reuters CSV."""
    return REUTERS_ROWS


@pytest.fixture
def temp_reuters_headlines(reuters_columns, reuters_rows):
    """Create a temporary Reuters CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(reuters_columns)
        writer.writerows(reuters_rows)

        temp_path = Path(f.name)

    yield temp_path

    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_reuters_with_labels(reuters_columns, reuters_rows):
    """Reuters CSV carrying an extra labels column GPT-2 must not train on."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([*reuters_columns, "labels"])
        writer.writerows([*row, 1] for row in reuters_rows)

        temp_path = Path(f.name)

    yield temp_path

    if temp_path.exists():
        temp_path.unlink()
