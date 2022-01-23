import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cookbook_eda as eda

from pathlib import Path
from pprint import pprint
from more_itertools import pairwise
from sklearn.feature_extraction.text import CountVectorizer


def get_data(output=None):
    """returns a tuple of (list of dataframes, list of matching countries ID)"""
    current_path = Path.cwd()
    p = current_path.parent
    country_id_list = [f.stem[:2] for f in p.glob("**/*.csv")]
    paths_list = [f for f in p.glob("**/*.csv")]
    df_list = [pd.read_csv(path) for path in paths_list]

    if output != None:
        pprint(f"{country_id_list=}")

    if output != None:
        pprint(f"{paths_list=}")

    return df_list, country_id_list


def add_country_col(df_list, country_id_list):
    return [
        df_list[idx].assign(country=country_id)
        for (idx, df), country_id in zip(enumerate(df_list), country_id_list)
    ]


def merge_all_df(df_list_with_country):
    return pd.concat(df_list_with_country)


def get_youtube_categories():

    source_url = "https://gist.githubusercontent.com/dgp/1b24bf2961521bd75d6c/raw/9abca29c436218972e6e28de7c03060845ed85d9/youtube%2520api%2520video%2520category%2520id%2520list"

    return (
        pd.read_csv(source_url, header=None, names=["category_id"])
        .category_id.str.split(" - ", expand=True)
        .rename(columns={0: "category_id", 1: "category"})
        .loc[:31]
        .assign(category_id=lambda df_: df_.category_id.astype("int64"))
        .assign(category_id=lambda df_: df_.category_id.astype("category"))
        .assign(category=lambda df_: df_.category.str.strip())
    )


def clean_data(df):
    return (
        df.drop_duplicates()
        .reset_index(drop=True)
        .assign(category_id=lambda df_: pd.Categorical(df_.category_id))
        .assign(country=lambda df_: pd.Categorical(df_.country))
        .assign(
            trending_date=lambda df_: pd.to_datetime(
                df_.trending_date, format="%y.%d.%m"
            )
        )
        .assign(publish_time=lambda df_: pd.to_datetime(df_.publish_time))
        .assign(views=lambda df_: df_.views.astype(np.int32))
        .assign(likes=lambda df_: df_.likes.astype(np.int32))
        .assign(dislikes=lambda df_: df_.dislikes.astype(np.int32))
        .assign(comment_count=lambda df_: df_.comment_count.astype(np.int32))
        .assign(
            video_id=lambda df_: df_.video_id.mask(
                (df_.video_id == "#NAME?") | (df_.video_id == "#VALUE!"),
                df_.thumbnail_link.str.replace("/default.jpg", "")
                .str.replace("/default_live.jpg", "")
                .str[23:],
            )
        )
        .assign(
            title=lambda df_: df_.title.str.lower(),
            channel_title=lambda df_: df_.channel_title.str.lower(),
            tags=lambda df_: df_.tags.str.lower(),
            description=lambda df_: df_.description.str.lower(),
        )
        .merge(
            get_youtube_categories(),
            how="left",
            left_on="category_id",
            right_on="category_id",
        )
    )


def mem_report(all_df, optimized_df):
    original_size = all_df.memory_usage(deep=True).sum() // 1_000_000
    optimized_size = optimized_df.memory_usage(deep=True).sum() // 1_000_000

    print(f"{original_size} MB ==> {optimized_size} MB")
    print(f"{(1 - ((optimized_size / original_size))) * 100:.2f}% saved")


def main():
    df_list, country_id_list = get_data()
    df_list_with_country = add_country_col(df_list, country_id_list)
    all_df = merge_all_df(df_list_with_country)
    all_df.head(5)
    clean_df = clean_data(all_df)
    clean_df.head(5)
    mem_report(all_df, clean_df)


if __name__ == "__main__":
    main()
