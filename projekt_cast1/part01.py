#!/usr/bin/env python3
"""
IZV cast1 projektu
Autor: Veronika Nevarilova (xnevar00)

Detailni zadani projektu je v samostatnem projektu e-learningu.
Nezapomente na to, ze python soubory maji dane formatovani.

Muzete pouzit libovolnou vestavenou knihovnu a knihovny
predstavene na prednasce
"""

from bs4 import BeautifulSoup
import requests
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.axes import Axes
from typing import List, Callable, Dict, Any


def integrate(
        f: Callable[[NDArray], NDArray], a: float, b: float, steps=1000
) -> float:
    """
    Calculate the definite integral of a function within a specified range.

    Args:
        f (NDArray): A function for which the integral is to be computed.
        a (float): The lower limit of integration.
        b (float): The upper limit of integration.
        steps (NDArray): The number of steps for the integration.

    Returns:
        float: The approximate value of the definite integral.
    """
    vals = np.linspace(a, b, num=steps)
    result_values = (vals[1:] - vals[:-1]) * f((vals[:-1] + vals[1:]) / 2)
    return np.sum(result_values)


def generate_graph(
    a: List[float], show_figure: bool = False, save_path: str | None = None
):
    """
    A function that generates a graph of a function over the interval <-3, 3>.

    Args:
        a (List[float]): A list of values used for function values computation.
        show_figure (bool): Whether the graph should be displayed.
        save_path (float): If set, a path where the image will be saved.
    """
    colours = {"colour0": "blue", "colour1": "orange", "colour2": "green"}

    x = np.linspace(-3, 3, 200)
    result = (np.array(a)[:, np.newaxis] ** 2) * (x**3) * np.sin(x)
    areas = [np.trapz(r, x) for r in result]

    plt.figure(figsize=(10, 5))
    plt.plot(x, result.T)
    plt.grid(False)
    plt.xlabel("x")
    plt.ylabel("$f_a(x)$")
    plt.xlim(-3, 5)
    plt.ylim(0, 40)
    plt.legend(
        [f"$y_{{{a_val}}}(x)$" for a_val in a],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
    )
    plt.xticks([-3, -2, -1, 0, 1, 2, 3])

    for i, area in enumerate(areas):
        plt.fill_between(
            x, 0, result[i], alpha=0.1, color=f'{colours["colour"+ str(i)]}'
        )
        # description of integral for each graph
        plt.text(
            3,
            result[i][-1],
            f"$\\int f_{{{a[i]}}}(x)dx = {area:.2f}$",
            fontsize=12,
            color="black",
            ha="left",
            va="center",
        )

    if show_figure is True:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)


def set_plt(ax: Axes, ylabel: str):
    """
    A function helping to set axes properly at one place

    Args:
        ax (Axes): An axis to be set.
        ylabel (str): Description of the y-axis.
    """
    fmt = ticker.FixedLocator([-0.8, -0.4, 0, 0.4, 0.8])

    ax.grid(False)
    ax.set_xlabel("t")
    ax.set_xlim(0, 100)
    ax.yaxis.set_major_locator(fmt)
    ax.set_ylim(-0.8, 0.8)
    ax.set_ylabel(ylabel)


def generate_sinus(show_figure: bool = False, save_path: str | None = None):
    """
    A function that generates three subplots (f1, f2, f1+f2).
    f1 = 0.5*cos(1/50*pi*t)
    f2 = 0.25*(sin(pi*t)+sin(3/2*pi*t))

    Args:
        show_figure (bool): Whether the graph should be displayed.
        save_path (float): If set, a path where the image will be saved.
    """
    fig, axes = plt.subplots(
        ncols=1, nrows=3, constrained_layout=True, figsize=(8, 10)
    )
    (ax1, ax2, ax3) = axes

    t = np.linspace(0, 100, 20000)

    # first subplot
    set_plt(ax1, "$f_1(t)$")
    result1 = 0.5 * np.cos(1 / 50 * np.pi * t)
    ax1.plot(t, result1, color="C0")

    # second subplot
    set_plt(ax2, "$f_2(t)$")
    result2 = 0.25 * (np.sin(np.pi * t) + np.sin(3 / 2 * np.pi * t))
    ax2.plot(t, result2, color="C0")

    # third subplot
    set_plt(ax3, "$f_1(t) + f_2(t)$")
    result3 = result1 + result2

    green_signal = result3.copy()
    red_signal = result3.copy()

    # values that shouldn't be displayed are set to NaN for each color
    green_signal[green_signal < result1] = np.nan
    red_signal[red_signal > result1] = np.nan

    ax3.plot(t, red_signal, color="r")
    ax3.plot(t, green_signal, color="g")

    # final result
    if show_figure is True:
        plt.show()
    if save_path is not None:
        fig.savefig(save_path)


def download_data() -> List[Dict[str, Any]]:
    """
    The function fetches data from web containing table of weather stations.
    It extracts data for each row of the table and creates a dict with keys:

    'position': Location of the station.
    'lat': Latitude.
    'long': Longitude.
    'height': Height above sea level.


    Returns:
        List[Dict]: A list of dicts, where each dict represents one station.
    """
    resp = requests.get("https://ehw.fit.vutbr.cz/izv/st_zemepis_cz.html")
    if resp.status_code != 200:
        return []

    resp.encoding = "utf-8"
    html_content = resp.text

    soup = BeautifulSoup(html_content, "html.parser")

    rows = soup.find_all(
        "tr", class_="nezvyraznit"
    )  # finding the table in the html doc

    records = []

    for row in rows:
        columns = row.find_all("td")
        if len(columns) == 7:
            city = columns[0].strong.text
            latitude = (
                columns[2].text.replace("°", "").replace(",", ".")
            )  # correct formatting
            longitude = columns[4].text.replace("°", "").replace(",", ".")
            altitude = columns[6].text.strip().replace(",", ".")
            records.append(
                {
                    "position": city,
                    "lat": float(latitude),
                    "long": float(longitude),
                    "height": float(altitude),
                }
            )

    # result list with data
    return records
