#!/usr/bin/env python3.11
# coding=utf-8

# author Veronika Nevarilova (xnevar00)

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import zipfile
import io
import matplotlib.dates as mdates

# muzete pridat libovolnou zakladni knihovnu ci knihovnu predstavenou na
# prednaskach

# dalsi knihovny pak na dotaz

# Ukol 1: nacteni dat ze ZIP souboru


def load_data(filename: str) -> pd.DataFrame:
    """
    brief Loads data from 'filename' into one DataFrame and adds column region
    based on the name of the file.

    param filename File with data to load.

    return Loaded data in one dataframe.
    """
    # headers of columns in csv files
    headers = [
        "p1",
        "p36",
        "p37",
        "p2a",
        "weekday(p2a)",
        "p2b",
        "p6",
        "p7",
        "p8",
        "p9",
        "p10",
        "p11",
        "p12",
        "p13a",
        "p13b",
        "p13c",
        "p14",
        "p15",
        "p16",
        "p17",
        "p18",
        "p19",
        "p20",
        "p21",
        "p22",
        "p23",
        "p24",
        "p27",
        "p28",
        "p34",
        "p35",
        "p39",
        "p44",
        "p45a",
        "p47",
        "p48a",
        "p49",
        "p50a",
        "p50b",
        "p51",
        "p52",
        "p53",
        "p55a",
        "p57",
        "p58",
        "a",
        "b",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "p5a",
    ]

    # abbreviation of regions based on its code
    regions = {
        "PHA": "00",
        "STC": "01",
        "JHC": "02",
        "PLK": "03",
        "ULK": "04",
        "HKK": "05",
        "JHM": "06",
        "MSK": "07",
        "OLK": "14",
        "ZLK": "15",
        "VYS": "16",
        "PAK": "17",
        "LBK": "18",
        "KVK": "19",
    }

    # load all data from 'filename' to one DataFrame
    loaded_data = pd.DataFrame()

    with zipfile.ZipFile(filename, "r") as main_zip:
        # going through inner zip files
        for file in main_zip.namelist():
            with main_zip.open(file) as inner_zip_file:
                with zipfile.ZipFile(
                    io.BytesIO(inner_zip_file.read())
                ) as inner_zip:
                    # going through csv files in the inner zip file
                    for region_name, region_number in regions.items():
                        with inner_zip.open(
                            f"{region_number}.csv", "r"
                        ) as csv_file:
                            df = pd.read_csv(
                                csv_file,
                                encoding="cp1250",
                                names=headers,
                                sep=";",
                                low_memory=False,
                            )
                            df["region"] = region_name
                            loaded_data = pd.concat(
                                [loaded_data, df], ignore_index=True
                            )

    return loaded_data


# Ukol 2: zpracovani dat


def parse_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    brief Adds a new column 'date' to the dataframe and converts columns to
    the right type.

    param df Dataframe to parse.
    param verbose If True, prints original and new size of the dataframe.

    return Parsed dataframe.
    """

    parsed_df = df.copy()
    parsed_df.drop_duplicates(subset="p1", keep="first", inplace=True)
    parsed_df["date"] = pd.to_datetime(parsed_df["p2a"])
    parsed_df.drop(columns=["p2a"], inplace=True)

    # columns to convert to category type
    category = ["p47", "h", "i", "k", "l", "p", "q", "t"]

    # columns to convert ','to '.'
    float = ["a", "b", "d", "e", "f", "g", "n", "o"]

    # columns to keep the way they are
    skip = ["region", "date"]

    for col in category:
        parsed_df[col] = parsed_df[col].astype("category")

    for col in float:
        parsed_df[col] = parsed_df[col].replace(",", ".", regex=True)

    for col in parsed_df.columns:
        if col not in category and col not in skip:
            parsed_df[col] = pd.to_numeric(parsed_df[col], errors="coerce")

    if verbose:
        orig_size = df.memory_usage(deep=True).sum() / 1e6
        new_size = parsed_df.memory_usage(deep=True).sum() / 1e6

        print(f"Original size: {orig_size:.1f} MB")
        print(f"New size: {new_size:.1f} MB")

    return parsed_df


# Ukol 3: počty nehod oidke stavu řidiče


def plot_state(
        df: pd.DataFrame, fig_location: str = None, show_figure: bool = False
):
    """
    brief Plots number of accidents based on the state of the driver.

    param df Dataframe to plot.
    param fig_location If set, saves the fig to the location.
    param show_figure If true, shows the figure.
    """
    states = {
        2: "Unaven, usnul, náhlá fyzická indispozice",
        3: "Pod vlivem léků, narkotik",
        4: "Pod vlivem alkoholu, obsah alkoholu v krvi do 0,99‰",
        5: "Pod vlivem alkoholu, obsah alkoholu v krvni 1‰ a více",
        8: "Řidič při jízdě zemřel",
        9: "Pokus o sebevraždu, sebevražda",
    }

    df_to_show = df.copy()

    # change state of driver to description
    df_to_show["p57"] = df_to_show["p57"].map(states)

    grouped_states = (
        df_to_show.groupby(["region", "p57"]).size().reset_index(name="sum")
    )

    # create figure
    fig, axes = plt.subplots(
        ncols=2, nrows=3, constrained_layout=True, figsize=(12, 10)
    )
    fig.suptitle("Počet nehod dle stavu řidiče při nedobrém stavu")

    for i, (state, desc) in enumerate(states.items()):
        ax = axes[i // 2, i % 2]
        ax.set_title(f"Stav řidiče: {desc}", fontsize=10)

        sns.barplot(
            x="region",
            y="sum",
            data=grouped_states[grouped_states["p57"] == desc],
            ax=ax,
            palette=sns.color_palette("dark:#5A9_r", as_cmap=True),
            hue="sum",
        )
        ax.get_legend().remove()

        # set axis appearance
        if i % 2 == 0:
            ax.set_ylabel("Počet nehod")
        else:
            ax.set(ylabel=None)
        if i // 2 == 2:
            ax.set_xlabel("Kraj")
        else:
            ax.set(xlabel=None)

    if fig_location:
        fig.savefig(fig_location)

    if show_figure:
        plt.show()
    else:
        plt.close(fig)


# Ukol4: alkohol v jednotlivých hodinách


def plot_alcohol(
        df: pd.DataFrame, fig_location: str = None, show_figure: bool = False
):
    """
    brief Plots number of accidents based on the hour of the day when
    the driver was under the influence of alcohol.

    param df Dataframe to plot.
    param fig_location If set, saves the fig to the location.
    param show_figure If true, shows the figure.
    """
    df_to_show = df.copy()
    df_to_show.dropna(subset=["p2b"], inplace=True)

    # convert time to hours
    df_to_show["p2b"] = df_to_show["p2b"] // 100

    # selected regions
    regions = ["ZLK", "STC", "JHM", "MSK"]

    # add column with information if the driver was under the influence
    df_to_show["Alkohol"] = df_to_show["p11"].apply(
        lambda x: "Ano" if x >= 3 else ("Ne" if x in [1, 2] else np.nan)
    )

    # filter invalid values
    df_to_show = df_to_show[
        (df_to_show["p2b"] >= 0) & (df_to_show["p2b"] <= 23)
    ]

    # filter selected regions
    df_to_show = df_to_show[df_to_show["region"].isin(regions)]
    grouped_data = (
        df_to_show.groupby(["region", "p2b", "Alkohol"])
        .size()
        .reset_index(name="pocet_nehod")
    )

    fig, axes = plt.subplots(
        ncols=2, nrows=2, constrained_layout=True, figsize=(12, 10)
    )

    for i, region in enumerate(regions):
        ax = axes[i // 2, i % 2]
        ax.set_facecolor("#dee2e6")
        ax.grid(color="white", linestyle="-", linewidth=1)

        sns.barplot(
            x="p2b",
            y="pocet_nehod",
            hue="Alkohol",
            data=grouped_data[grouped_data["region"] == region],
            ax=ax,
            zorder=2,
        )

        # set axis appearance
        ax.set_ylabel("Počet nehod")
        ax.set_xlabel("Hodina")
        ax.set_title(f"Kraj: {region}")
        ax.get_legend().remove()

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(1.1, 0.5),
        title="Alkohol",
        frameon=False,
    )

    if fig_location:
        fig.savefig(fig_location, bbox_inches="tight")

    if show_figure:
        plt.show()
    else:
        plt.close(fig)


# Ukol 5: Zavinění nehody v čase


def plot_fault(
        df: pd.DataFrame, fig_location: str = None, show_figure: bool = False
):
    """
    brief Plots number of accidents based on the cause of the accident.

    param df Dataframe to plot.
    param fig_location If set, saves the fig to the location.
    param show_figure If true, shows the figure.
    """

    # cause based on the code
    cause = {
        1: "Řidičem motorového vozidla",
        2: "Řidičem nemotorového vozidla",
        3: "Chodcem",
        4: "Lesní zvěří, domácím zvířectvem",
    }

    # selected regions
    regions = ["ZLK", "STC", "JHM", "MSK"]

    df_to_show = df.copy()

    # keep only selected regions and causes
    df_to_show = df_to_show[
        (df_to_show["region"].isin(regions)) & (df_to_show["p10"].isin(cause))
    ]

    # change cause to description
    df_to_show["p10"] = df_to_show["p10"].map(cause)

    df_to_show["date"] = pd.to_datetime(df_to_show["date"])

    pivot_data = df_to_show.pivot_table(
        index=["date", "region"], columns=["p10"], aggfunc="size", fill_value=0
    )

    monthly = pivot_data.groupby("region").resample("M", level=0).sum()

    monthly_stacked = monthly.stack().reset_index(name="count")
    monthly_stacked.columns = ["region", "date", "p10", "count"]

    fig, axes = plt.subplots(
        ncols=2, nrows=2, constrained_layout=True, figsize=(10, 8), sharey=True
    )

    for i, region in enumerate(regions):
        ax = axes[i // 2, i % 2]
        ax.set_facecolor("#dee2e6")
        ax.grid(color="white", linestyle="-", linewidth=1)

        region_data = monthly_stacked[monthly_stacked["region"] == region]
        sns.lineplot(x="date", y="count", hue="p10", data=region_data, ax=ax)

        # set axis appearance
        ax.set_title(f"Kraj: {region}")
        ax.set_xlabel(None)
        ax.set_ylabel("Počet nehod")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
        ax.set_xlim(pd.Timestamp("2016-01-01"), pd.Timestamp("2023-01-01"))
        ax.set_ylim(bottom=0, top=1200)
        ax.get_legend().remove()

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(1.3, 0.5),
        title="Zavinění",
        frameon=False,
    )

    if fig_location:
        fig.savefig(fig_location, bbox_inches="tight")

    if show_figure:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    # zde je ukazka pouziti, tuto cast muzete modifikovat podle libosti
    # skript nebude pri testovani pousten primo, ale budou volany konkreni
    # funkce.
    df = load_data("data.zip")
    df2 = parse_data(df, True)

    plot_state(df2, "01_state.png")
    plot_alcohol(df2, "02_alcohol.png")
    plot_fault(df2, "03_fault.png")


# Poznamka:
# pro to, abyste se vyhnuli castemu nacitani muzete vyuzit napr
# VS Code a oznaceni jako bunky (radek #%%% )
# Pak muzete data jednou nacist a dale ladit jednotlive funkce
# Pripadne si muzete vysledny dataframe ulozit nekam na disk (pro ladici
# ucely) a nacitat jej naparsovany z disku
