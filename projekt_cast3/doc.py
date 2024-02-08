# author Veronika Nevarilova (xnevar00)

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


def weather_graph(
    df: pd.DataFrame, fig_location: str = "fig", show_figure: bool = False
):
    """
    brief Generates graph of accidents in bad weather through years 2016-2022

    param df dataframe to process.
    """

    working_df = df.copy()

    weather = {
        2: "Mlha",
        3: "Slabý déšť, mrholení apod.",
        4: "Déšť",
        5: "Sněžení",
        6: "Tvoří se námraza, náledí",
        7: "Nárazový vítr (boční, vichřice apod.)",
        0: "Jiné ztížené"
    }

    # bad conditions
    working_df = working_df[working_df['p18'] != 1]
    working_df['p2a'] = pd.to_datetime(working_df['p2a'])

    # divided and grouped by quarters
    working_df['month_year'] = working_df['p2a'].dt.to_period('Q').\
        dt.strftime('%Y-Q%q')
    grouped_data = working_df.groupby(['month_year', 'p18']).size().\
        unstack(fill_value=0)

    # creating the graph
    plt.figure(figsize=(15, 6))
    for category in grouped_data.columns:
        plt.plot(
            grouped_data.index, grouped_data[category],
            label=f'{weather[category]}'
        )

    # visual settings
    plt.xlabel('Čtvrtletí')
    plt.ylabel('Počet nehod')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    if fig_location:
        plt.savefig(fig_location)

    if show_figure:
        plt.show()


def count_data(df: pd.DataFrame):
    """
    brief Calculates the precentage of accidents with different
          types of bad weather conditions

    param df dataframe to process.
    """
    df_with_weather = df.copy()

    # bad conditions
    df_with_weather = df_with_weather[df_with_weather['p18'] != 1]
    weather_count = len(df_with_weather)
    print("Poměr nehod při zhoršeném počasí ku celkovému počtu nehod:",
          f"{100*(weather_count / len(df)):.1f}")
    print("\n")

    df_categories = df.copy()

    # rain and drizzle
    df_rain = df_categories[
        (df_categories['p18'] == 3) | (df_categories['p18'] == 4)
    ]
    # snow and ice
    df_snow = df_categories[
        (df_categories['p18'] == 5) | (df_categories['p18'] == 6)
    ]
    # fog and wind
    df_air = df_categories[
        (df_categories['p18'] == 2) | (df_categories['p18'] == 7)
    ]
    # others
    df_others = df_categories[(df_categories['p18'] == 0)]

    print("\t\tPočet nehod\tPoměr")
    print("Déšť\t\t", len(df_rain), "\t\t",
          f"{(len(df_rain)/weather_count*100):.1f}")
    print("Sníh/Led\t", len(df_snow), "\t\t",
          f"{(len(df_snow)/weather_count*100):.1f}")
    print("Mlha/Vítr\t", len(df_air), "\t\t",
          f"{(len(df_air)/weather_count*100):.1f}")
    print("Jiné\t\t", len(df_others), "\t\t",
          f"{(len(df_others)/weather_count*100):.1f}")
    print("\n")


def statistics_winter_weather(df: pd.DataFrame):
    """
    brief Compares accidents in winter and other months
          in question of bad conditions

    param df dataframe to process.
    """

    working_df = df.copy()
    working_df['p2a'] = pd.to_datetime(working_df['p2a'])
    working_df['month'] = working_df['p2a'].dt.month

    # december -> march
    winter_months = [12, 1, 2, 3]

    # calculating percentage of accidents with bad conditions in winter
    df_winter = working_df.copy()
    df_winter = df_winter[df_winter['month'].isin(winter_months)]
    weather_winter = len(df_winter[df_winter['p18'] != 1])
    no_weather_winter = len(df_winter[df_winter['p18'] == 1])
    print(
        "Poměr nehod se ztíženými podmínkami ku celkovému počtu nehod v zimě:",
        f"{100*(weather_winter/(no_weather_winter+weather_winter)):.1f}%")

    # calculating percentage of accidents with bad conditions in other months
    df_others = working_df.copy()
    df_others = df_others[~df_others['month'].isin(winter_months)]
    weather_others = len(df_others[df_others['p18'] != 1])
    no_weather_others = len(df_others[df_others['p18'] == 1])
    print(
        "Poměr nehod se ztíženými podmínkami ku celkovému počtu nehod \
        v ostatních měsících:",
        f"{100*(weather_others/(no_weather_others+weather_others)):.1f}%"
    )
    print("\n")

    # divided to winter/non-winter season
    working_df['season'] = working_df['month'].\
        apply(lambda x: 'winter' if x in winter_months else 'non-winter')

    working_df['bad_weather'] = working_df['p18'] != 1

    ct = pd.crosstab(working_df['season'],
                     working_df['bad_weather'])

    # chi squared statistics to find out whether there is big difference
    # in accidents with bad weather in winter/in other months
    chi2, p, dof, expected = chi2_contingency(ct)
    print("Poměr nehod se ztíženými povětrnostními podmínkami v zimních \
          měsících ku ostatním měsícům:",
          f"{100*(ct.iloc[1, 1]/(ct.iloc[0, 1] + ct.iloc[1, 1])):.1f}%")

    if p < 0.05:
        print("Je výrazný rozdíl v poměru nehod se zhoršenými povětrnostními \
              podmínkami během zimních a jiných měsíců.")
    else:
        print("Není výrazný rozdíl v poměru nehod se zhoršenými \
              povětrnostními podmínkami během zimních a jiných měsíců.")


if __name__ == "__main__":
    # load data
    df = pd.read_pickle('./accidents.pkl.gz', compression='gzip')
    weather_graph(df, "fig")
    count_data(df)
    statistics_winter_weather(df)
