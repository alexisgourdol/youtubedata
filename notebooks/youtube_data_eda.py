# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# # Youtube Data EDA

# <markdowncell>

# ## Project Description

# <markdowncell>

# **Goal:** practice EDA
# 
# **Context:** YouTube (the world-famous video sharing website) maintains a list of the top trending videos on the platform. According to Variety magazine, “To determine the year’s top-trending videos, YouTube uses a combination of factors including measuring users interactions (number of views, shares, comments and likes). Note that they’re not the most-viewed videos overall for the calendar year”. Top performers on the YouTube trending list are music videos (such as the famously virile “Gangam Style”), celebrity and/or reality TV performances, and the random dude-with-a-camera viral videos that YouTube is well-known for.
# 
# This dataset is a daily record of the top trending YouTube videos.
# 
# Note that this dataset is a structurally improved version of this dataset.
# 
# **Data Source:** https://www.kaggle.com/datasnaek/youtube-new
# 
# **Author:** https://www.kaggle.com/datasnaek

# <markdowncell>

# _______________

# <markdowncell>

# **Data cleaning and exploration notes:**
# 


# <markdowncell>

# - Merge 10 csv files into an `all_df` pandas DataFrame
#     - utf-8 default results in Error    
#     `#UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc3 in position 130670: invalid continuation byte`
#     - Solve encoding issues for MX, RU, JP, KR by opening the csv files in an editor, make a small change, save.
#     - Regular utf-8 encoding should work now
# - Create `country` column
# - Drop duplicates & Reset index
# - Make `country` , `category_id` columns a categorical dtype
# - Turn into datetime `trending_date`, `publish_time`
# - Make `views`, `likes`, `dislikes`, `comment_count`,   as np.int32
# - Process `video_id`
#     - About 2100 rows have a `video_id` equal to `#NAME?` (probaly beacause they all start with a `-`).
#     - A few more have `#VALUE!`. 
#     - We can extract the correct id value from the `thumbnail_link` URL
# - Lowercase `title`, `channel_title`,  `tags`, `description`
# - Enrich with [`category`](https://gist.githubusercontent.com/dgp/1b24bf2961521bd75d6c/raw/9abca29c436218972e6e28de7c03060845ed85d9/youtube%2520api%2520video%2520category%2520id%2520list) 
# - Process `tags`

# <markdowncell>

# ## Imports

# <markdowncell>

# [`itertools.paiwise`](https://docs.python.org/3/library/itertools.html#itertools.pairwise) is New in version 3.10.
# For previous python version, use [Itertool Recipes](https://docs.python.org/3/library/itertools.html#itertools-recipes) with `pip install more-itertools`

# <codecell>

#!pip install more-itertools 

# <codecell>

from youtubedata.preproc import get_data
from youtubedata.preproc import add_country_col
from youtubedata.preproc import merge_all_df
from youtubedata.preproc import get_data
from youtubedata.preproc import get_data

# <codecell>

import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cookbook_eda as eda

from pathlib import Path
from pprint import pprint
from more_itertools import pairwise
from sklearn.feature_extraction.text import CountVectorizer

# <markdowncell>

# ## Load data

# <markdowncell>

# ### Load

# <markdowncell>

# [Pandas I/O](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html)
# 
# `pd.read_csv`, `pd.read_json`

# <codecell>

df_list, country_id_list = get_data()

# <codecell>

df_list

# <codecell>

country_id_list

# <markdowncell>

# ### Merge 10 dataframes into one

# <codecell>

df_list_with_country = add_country_col(df_list, country_id_list)

# <codecell>

df_list_with_country

# <codecell>

all_df = merge_all_df(df_list_with_country)

# <codecell>

all_df.head(5)

# <markdowncell>

# ### Asserts

# <codecell>

# check all dataframes have the same number of columns
assert [len(df.columns) for df in df_list] == [16, 16, 16, 16, 16, 16, 16, 16, 16, 16]

# <codecell>

# check all dataframes have the same column names
assert [a == b for a, b in pairwise([set(df.columns) for df in df_list])] == [
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

# <codecell>

# check country creation
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

# <codecell>

# check number of rows in all_df vs rows in original dfs
assert set(all_df.country.value_counts()) == {df.country.shape[0] for df in df_list_with_country}

assert all_df.shape[0] == sum([df.shape[0] for df in df_list_with_country])
assert all_df.shape[1] == np.mean([df.shape[1] for df in df_list_with_country])

# <markdowncell>

# ## Explore dtypes

# <codecell>

all_df.dtypes.rename("data_types").sort_values().to_frame().T

# <codecell>

eda_df= eda.enriched_describe(all_df)
eda_df

# <codecell>

# remove duplicate outputs
lst = ["np int64", "np object"]
eda.subset(eda_df, lst)

# <markdowncell>

# ## Explore distributions

# <markdowncell>

# ### Snippets

# <markdowncell>

# ![Screenshot%202022-01-22%20at%2009.58.53.png](attachment:Screenshot%202022-01-22%20at%2009.58.53.png)

# <markdowncell>

# ![Screenshot%202022-01-22%20at%2010.00.42.png](attachment:Screenshot%202022-01-22%20at%2010.00.42.png)

# <markdowncell>

# Histogram on cat cols : 
# 
# ValueError: 
# Post Malone - Rockstar ft. T-Pain, Joey Bada$$ (Remix)
#                                             ^
# Expected end of text, found '$'  (at char 44), (line:1, col:45)

# <markdowncell>

# ### Code

# <codecell>

if all_df.shape[0] > 5_000:
    small_df = all_df.sample(5_000)

# <codecell>

def hist_num_cols(df):
    num_cols = {col for col in df.select_dtypes(include="number")}

    custom_params = {
        "axes.spines.left": False,
        "axes.spines.right": False,
        "axes.spines.bottom": True,
        "axes.spines.top": True,
        "axes.grid": False,
    }
    with sns.axes_style("whitegrid", rc=custom_params):

        fig, axs = plt.subplots(nrows=len(num_cols), ncols=1, figsize=(12, 8))

        for idx, col in enumerate(num_cols):
            sns.histplot(x=df[col], data=df, ax=axs[idx], kde=True)
            sns.rugplot(x=df[col], data=df, ax=axs[idx], c="black")
            axs[idx].set_ylabel(col, rotation=0, labelpad=50)
            
hist_num_cols(small_df)

# <codecell>

def box_num_cols(df):
    num_cols = {col for col in df.select_dtypes(include="number")}

    custom_params = {
        "axes.spines.left": False,
        "axes.spines.right": False,
        "axes.spines.bottom": True,
        "axes.spines.top": True,
        "axes.grid": False,
    }
    with sns.axes_style("whitegrid", rc=custom_params):

        fig, axs = plt.subplots(nrows=len(num_cols), ncols=1, figsize=(12, 8))

        for idx, col in enumerate(num_cols):
            sns.boxplot(x=df[col], data=df, ax=axs[idx], color="white")
            sns.stripplot(x=df[col], data=df, ax=axs[idx], alpha=0.6)
            axs[idx].set_ylabel(col, rotation=0, labelpad=50)
box_num_cols(small_df)

# <codecell>

def hist_cat_cols(df):
    num_cols = {col for col in df.select_dtypes(exclude="number")}

    custom_params = {
        "axes.spines.left": False,
        "axes.spines.right": False,
        "axes.spines.bottom": True,
        "axes.spines.top": True,
        "axes.grid": False,
    }
    with sns.axes_style("whitegrid", rc=custom_params):

        fig, axs = plt.subplots(nrows=len(num_cols), ncols=1, figsize=(12, 8))

        for idx, col in enumerate(num_cols):
            sns.histplot(x=df[col].fillna("NA"), data=df, ax=axs[idx])
            axs[idx].set_ylabel(col, rotation=0, labelpad=50)
            plt.tight_layout()

hist_cat_cols(small_df)

# <markdowncell>

# Input from Domain Expert:
#   - Missing values
#         - why are some values are missing? 
#         - can they have a meaning? 
#         eg NaN can be replaced by zero, observation should be imputed or discarted etc...
#         
#   - Any apparent issue ? 
#          - e.g. missing unique values in a category
#          - e.g. suspicious distributions, wrong units ...
#   - Candidate columns
#          - object to Categorical
#          - object to bool
#          - float to integer after NaN removal
#          - float precision 
#          - use Nullable type if it makes sense (pd.Int64, pd.Boolean)

# <markdowncell>

# _______________

# <markdowncell>

# # Data Cleaning

# <codecell>

def mem_report(all_df, optimized_df):
    original_size  = all_df.memory_usage(deep=True).sum()       // 1_000_000
    optimized_size = optimized_df.memory_usage(deep=True).sum() // 1_000_000

    print(f"{original_size} MB ==> {optimized_size} MB")
    print(f"{(1 - ((optimized_size / original_size))) * 100:.2f}% saved")

# <markdowncell>

# ## Drop duplicates & reset index

# <codecell>

optimized_df = all_df.drop_duplicates().reset_index(drop=True)

# <codecell>

mem_report(all_df, optimized_df)

# <markdowncell>

# ## Make `country`, `category_id` columns as categorical dtype

# <codecell>

optimized_df = (
    optimized_df.assign(category_id=lambda df_: pd.Categorical(df_.category_id))
                .assign(country=lambda df_: pd.Categorical(df_.country))
)
optimized_df.head().dtypes

# <codecell>

mem_report(all_df, optimized_df)

# <markdowncell>

# ## Turn into datetime `trending_date`, `publish_time`

# <codecell>

optimized_df = (
    optimized_df.assign(trending_date=lambda df_: pd.to_datetime(df_.trending_date, format="%y.%d.%m"))
                .assign(publish_time= lambda df_: pd.to_datetime(df_.publish_time))
)
optimized_df.dtypes

# <codecell>

mem_report(all_df, optimized_df)

# <markdowncell>

# ## Make `views`, `likes`, `dislikes`, `comment_count`,   as int32

# <markdowncell>

# ### Checks : uint16 and 32 , to_numeric

# <codecell>

cols = ["views", "likes", "dislikes", "comment_count"]

# <codecell>

# check : np.uint16 too small
for col in cols:
    print(col,
          f"uint16 min: {np.iinfo(np.uint16).min}, max:{np.iinfo(np.uint16).max}",
          f"{col} min: {min(all_df[col])}, max:{max(all_df[col])}",
          f"min fits in  np.uint16 : {np.iinfo(np.uint16).min <= min(all_df[col]) < np.iinfo(np.uint16).max}",
          f"max fits in  np.uint16 : {np.iinfo(np.uint16).min <= max(all_df[col]) < np.iinfo(np.uint16).max}",
         sep='\n')
    print("\n")

# <codecell>

# check : np.uint32 OK
for col in cols:
    print(col,
          f"uint32 min: {np.iinfo(np.uint32).min}, max:{np.iinfo(np.uint32).max}",
          f"{col} min: {min(all_df[col])}, max:{max(all_df[col])}",
          f"min fits in  np.uint32 : {np.iinfo(np.uint32).min <= min(all_df[col]) < np.iinfo(np.uint32).max}",
          f"max fits in  np.uint32 : {np.iinfo(np.uint32).min <= max(all_df[col]) < np.iinfo(np.uint32).max}",
         sep='\n')
    print("\n")

# <codecell>

# check if to_numeric would downcast as e want : need to use  downcast='unsigned'
for col in cols:
    print(col)
    print("original                        :", all_df[col].memory_usage(deep=True))
    print("to_numeric                      :", pd.to_numeric(all_df[col]).memory_usage(deep=True))
    print("to_numeric  downcast='unsigned' :", pd.to_numeric(all_df[col], downcast='unsigned').memory_usage(deep=True))
    print("int32                           :", all_df[col].astype(np.int32).memory_usage(deep=True))
    print("="*80)

# <markdowncell>

# ### Transformations

# <codecell>

optimized_df = (
    optimized_df.assign(views=         lambda df_: df_.views.astype(np.int32))
                .assign(likes=         lambda df_: df_.likes.astype(np.int32))
                .assign(dislikes=      lambda df_: df_.dislikes.astype(np.int32))
                .assign(comment_count= lambda df_: df_.comment_count.astype(np.int32))
)
optimized_df.dtypes

# <codecell>

mem_report(all_df, optimized_df)

# <markdowncell>

# ## Process `video_id`

# <markdowncell>

# ### Explore

# <codecell>

optimized_df.query("video_id == '#NAME?'").shape

# <codecell>

# 66 observations have the wrong video_id, not just the '#NAME?' value
# => specify video_id == '#NAME?' to change only the rows we want
(
    pd.concat(
        [
            optimized_df.thumbnail_link,
            optimized_df.video_id,
            optimized_df.thumbnail_link.str[23:]
                .str.replace("/default.jpg", "")
                .str.replace("/default_live.jpg", ""),
            optimized_df.thumbnail_link.str[23:]
                .str.replace("/default.jpg", "")
                .str.replace("/default_live.jpg", "")
            == optimized_df.video_id,
        ],
        axis="columns",
    )
    .rename(columns={0: "id_match"})
    .query("id_match==False and video_id!= '#NAME?'")
)

# <codecell>

optimized_df.video_id

# <markdowncell>

# ### Transformations

# <codecell>

optimized_df = optimized_df.assign(
    video_id=optimized_df.video_id.mask(
        (optimized_df.video_id == "#NAME?") | (optimized_df.video_id == "#VALUE!"),
        optimized_df.thumbnail_link.str.replace("/default.jpg", "")
            .str.replace("/default_live.jpg", "")
            .str[23:],
    )
)
optimized_df

# <markdowncell>

# ## Lowercase `title`, `channel_title`, `tags`, `description`

# <codecell>

optimized_df = optimized_df.assign(
    title=optimized_df.title.str.lower(),
    channel_title=optimized_df.channel_title.str.lower(),
    tags=optimized_df.tags.str.lower(),
    description=optimized_df.description.str.lower()
)

# <codecell>

optimized_df.sample(5)

# <markdowncell>

# ## Enrich with `category`

# <codecell>

def get_youtube_categories():
    
    source_url ="https://gist.githubusercontent.com/dgp/1b24bf2961521bd75d6c/raw/9abca29c436218972e6e28de7c03060845ed85d9/youtube%2520api%2520video%2520category%2520id%2520list"
    
    return (
    pd.read_csv(source_url, header=None, names=["category_id"])
        .category_id.str.split(" - ", expand=True)
        .rename(columns={0: "category_id", 1: "category"})
        .loc[:31]
        .assign(category_id= lambda df_: df_.category_id.astype("int64"))
        .assign(category_id= lambda df_: df_.category_id.astype("category"))
        .assign(category= lambda df_: df_.category.str.strip())
)

# <codecell>

categories_df = get_youtube_categories()
categories_df

# <codecell>

optimized_df = (
    optimized_df.merge(get_youtube_categories(), 
                       how='left', 
                       left_on='category_id', 
                       right_on='category_id')
)

optimized_df.sample(5)

# <markdowncell>

# ## Process `tags`

# <codecell>

del all_df

# <codecell>

""" #clean list of tags #not useful
tags_s = (
    optimized_df.tags.str.split("|")
                     .explode("tags")
                     .str.replace("\"", "")
                     .str.split()
                     .explode("tags")
)
tags_s"""

#del tags_s

# <codecell>

vec_list = [CountVectorizer().fit_transform(df.tags) for df in df_list]
vec_shapes = [vec.shape for vec in vec_list]

# <codecell>

vec_dict= {k:v for k, v in zip(df_countries, vec_list)}
vec_shapes_dict= {k:v for k, v in zip(df_countries, vec_shapes)}

# <codecell>

vec_dict

# <codecell>

vec_shapes_dict.items()

# <markdowncell>

# ## Next

# <codecell>

optimized_df.dtypes

# <codecell>

optimized_df

# <codecell>

optimized_df.loc[38914]

# <codecell>

all_df.drop_duplicates().shape

# <codecell>

# do we have the same shapes for each col vs original dataset ?
# Goal: find a good key for the index
for col in optimized_df.columns:
    print(col, optimized_df[col].shape == all_df.drop_duplicates().shape)

# <codecell>

mask=(optimized_df.trending_date.astype(str) + optimized_df.video_id.astype(str)).duplicated()
optimized_df[mask]

# <markdowncell>

# ## Data Cleaning Consolidation

# <codecell>

def clean_df(df):
    return (df.assign(category_id=  lambda df_: pd.Categorical(df_.category_id))
              .assign(country=      lambda df_: pd.Categorical(df_.country))
              .assign(trending_date=lambda df_: pd.to_datetime(df_.trending_date, format="%y.%d.%m"))
              .assign(publish_time= lambda df_: pd.to_datetime(df_.publish_time))
              .assign(views=        lambda df_: df_.views.astype(np.int32))
              .assign(likes=        lambda df_: df_.likes.astype(np.int32))
              .assign(dislikes=     lambda df_: df_.dislikes.astype(np.int32))
              .assign(comment_count=lambda df_: df_.comment_count.astype(np.int32))
              .assign(video_id=optimized_df.video_id.mask(
                        (optimized_df.video_id == "#NAME?") | (optimized_df.video_id == "#VALUE!"),
                         optimized_df.thumbnail_link.str.replace("/default.jpg", "")
                                                    .str.replace("/default_live.jpg", "")
                                                    .str[23:])
                     )
              .assign(title=optimized_df.title.str.lower(),
                      channel_title=optimized_df.channel_title.str.lower(),
                      tags=optimized_df.tags.str.lower(),
                      description=optimized_df.description.str.lower())
              .merge(get_youtube_categories(), 
                      how='left', 
                      left_on='category_id', 
                      right_on='category_id')
)
clean_df(all_df)

# <codecell>


