import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_guardian_file():
    """Create a temporary Guardian CSV file for testing."""
    import csv

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Time", "Headlines"])
        writer.writeheader()
        writer.writerow(
            {
                "Time": "18-Jul-20",
                "Headlines": "St Mawes named UK's top seaside resort in Which? poll",
            }
        )
        writer.writerow(
            {
                "Time": "18-Jul-20",
                "Headlines": "key areas Sunak must tackle to serve up economic recovery",
            }
        )
        writer.writerow(
            {
                "Time": "18-Jul-20",
                "Headlines": "Ask and Zizzi to close 75 outlets, threatening up to 1,200 jobs",
            }
        )
        writer.writerow(
            {
                "Time": "17-Jul-20",
                "Headlines": "Number of UK problem gamblers seeking help soars in lockdown",
            }
        )

        temp_path = Path(f.name)

    yield temp_path

    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_reuters_headlines():
    """Create a temporary reuters CSV file for testing."""
    import csv

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Headlines", "Time", "Description"])
        writer.writeheader()
        writer.writerow(
            {
                "Headlines": "St Mawes named UK's top seaside resort in Which? poll",
                "Time": "Jul 18 2020",
                "Description": "Alphabet Inc's Google said on Friday it would prohibit websites and apps that use its advertising technology from running ads on "
                "dangerous content"
                " that goes against scientific consensus during the coronavirus pandemic",
            }
        )
        writer.writerow(
            {
                "Headlines": "key areas Sunak must tackle to serve up economic recovery",
                "Time": "18-Jul-20",
                "Description": "top executives Mark Zuckerberg and Sheryl Sandberg as a part of its probe into whether the company has engaged in unlawful monopolistic practices, the Wall Street Journal reported on Friday",
            }
        )
        writer.writerow(
            {
                "Headlines": "Ask and Zizzi to close 75 outlets, threatening up to 1,200 jobs",
                "Time": "18-Jul-20",
                "Description": "A former boss of Mexico's state oil company Petroleos Mexicanos facing corruption charges that could envelop leaders of the last government was taken to a hospital early on Friday shortly after his overnight extradition to Mexico from Spain",
            }
        )

        temp_path = Path(f.name)

    yield temp_path

    if temp_path.exists():
        temp_path.unlink()
