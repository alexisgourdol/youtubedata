import pytest
from youtubedata.preproc import get_data
from youtubedata.preproc import add_country_col
from youtubedata.preproc import merge_all_df
from youtubedata.preproc import get_youtube_categories
from youtubedata.preproc import clean_data
from youtubedata.preproc import mem_report


@pytest.fixture
def load_dataset():
    return get_data()


def test_get_data_shape(load_dataset):
    assert load_dataset[0][0].shape == (40451, 16)


def test_get_data_shape(load_dataset):
    assert load_dataset[1] == [
        "MX",
        "IN",
        "DE",
        "JP",
        "KR",
        "CA",
        "RU",
        "FR",
        "US",
        "GB",
    ]


"""
def test_add_country_col():
    pass


def test_merge_all_df():
    pass


def test_get_youtube_categories():
    pass


def test_clean_data():
    pass


def test_mem_report():
    pass
"""
