import pytest
from youtubedata.preproc import get_data
from youtubedata.preproc import add_country_col
from youtubedata.preproc import merge_all_df
from youtubedata.preproc import get_youtube_categories
from youtubedata.preproc import clean_data
from youtubedata.preproc import mem_report
from more_itertools import pairwise


@pytest.fixture
def load_dataset():
    # load_dataset[0] is df_list, load_dataset[1] is country_id_list
    return get_data()


@pytest.fixture
def df_list_with_country(load_dataset):
    return add_country_col(load_dataset[0], load_dataset[1])


def test_get_data_shape(load_dataset):
    assert load_dataset[0][0].shape == (40451, 16)


def test_get_data_countries(load_dataset):
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


def test_get_data_columns_length(load_dataset):
    assert [len(df.columns) for df in load_dataset[0]] == [
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
    ]


def test_get_data_column_names(load_dataset):
    # check all dataframes have the same column names
    assert [
        a == b for a, b in pairwise([set(df.columns) for df in load_dataset[0]])
    ] == [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]


def test_add_country_col(df_list_with_country):
    assert ["country" in df.columns for df in df_list_with_country] == [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]


def test_merge_all_df(df_list_with_country):
    all_df = merge_all_df(df_list_with_country)
    assert set(all_df.country.value_counts()) == {
        df.country.shape[0] for df in df_list_with_country
    }


"""

def test_get_youtube_categories():
    pass


def test_clean_data():
    pass


def test_mem_report():
    pass
"""
