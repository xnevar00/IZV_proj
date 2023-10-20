#!/usr/bin/env python3
"""
IZV cast1 projektu
Autor: Veronika Nevarilova (xnevar00)

Detailni zadani projektu je v samostatnem projektu e-learningu.
Nezapomente na to, ze python soubory maji dane formatovani.

Muzete pouzit libovolnou vestavenou knihovnu a knihovny predstavene na prednasce
"""

from bs4 import BeautifulSoup
import requests
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import List, Callable, Dict, Any


def integrate(f: Callable[[NDArray], NDArray], a: float, b: float, steps=1000) -> float:
    values = np.linspace(a, b, num=steps)
    result_values = (values[1:] - values[:-1])*f((values[:-1] + values[1:])/2)
    return np.sum(result_values)


def generate_graph(a: List[float], show_figure: bool = False, save_path: str | None = None):
    colours = {'colour0': 'blue', 'colour1': 'orange', 'colour2': 'green'}

    x = np.linspace(-3, 3, 200)
    result = (np.array(a)[:, np.newaxis] ** 2) * (x ** 3) * np.sin(x)
    areas = [np.trapz(r, x) for r in result]

    plt.figure(figsize=(10, 5))
    plt.plot(x, result.T)
    plt.grid(False)
    plt.xlabel('x')
    plt.ylabel('$f_a(x)$')
    plt.xlim(-3,5)
    plt.ylim(0, 40)
    plt.legend([f'$y_{{{a_val}}}(x)$' for a_val in a], loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3)
    plt.xticks([-3, -2, -1, 0, 1, 2, 3])

    for i, area in enumerate(areas):
        plt.fill_between(x, 0, result[i], alpha=0.1, color=f'{colours["colour"+ str(i)]}')
        plt.text(3, result[i][-1], f'$\\int f_{{{a[i]}}}(x)dx = {area:.2f}$', fontsize=12, color='black', ha='left', va='center')

    if(show_figure == True):
        plt.show()
    if(save_path is not None):
        plt.savefig(save_path)


def generate_sinus(show_figure: bool = False, save_path: str | None = None):
    fig, axes = plt.subplots(ncols=1, nrows=3, constrained_layout=True, figsize=(8,10))
    (ax1,ax2, ax3) = axes

    fmt=ticker.FixedLocator([-0.8, -0.4, 0, 0.4, 0.8])
    t = np.linspace(0, 100, 20000)
    for ax in [ax1, ax2, ax3]:
        ax.grid(False)
        ax.set_xlabel("t")
        ax.set_xlim(0, 100)
        ax.yaxis.set_major_locator(fmt)
        ax.set_ylim(-0.8, 0.8)


    #firt subplot
    ax1.set_ylabel("$f_1(t)$")
    result1 = 0.5*np.cos(1/50 * np.pi * t)
    ax1.plot(t, result1, color='C0')

    #second subplot
    ax2.set_ylabel("$f_2(t)$")
    result2 = 0.25*(np.sin(np.pi * t) + np.sin(3/2 * np.pi * t))
    ax2.plot(t, result2, color='C0')

    #third subplot
    ax3.set_ylabel("$f_1(t) + f_2(t)$")
    result3 = result1 + result2

    green_signal = result3.copy()
    red_signal = result3.copy()
    green_signal[green_signal < result1] = np.nan
    red_signal[red_signal > result1] = np.nan

    ax3.plot(t, red_signal, color='r')
    ax3.plot(t, green_signal, color='g')

    #final result
    if(show_figure == True):
        plt.show()
    if(save_path is not None):
        fig.savefig(save_path)


def download_data() -> List[Dict[str, Any]]:
    resp = requests.get('https://ehw.fit.vutbr.cz/izv/st_zemepis_cz.html')
    if (resp.status_code != 200):
        return []

    html_content = resp.text
    html_content = resp.text.encode('latin-1').decode('utf-8')

    soup = BeautifulSoup(html_content, 'html.parser')

    rows = soup.find_all('tr', class_='nezvyraznit')

    records = []

    for row in rows:
        columns = row.find_all('td')
        if len(columns) == 7:
            city = columns[0].strong.text
            latitude = columns[2].text.replace("°", "").replace(",", ".")
            longitude = columns[4].text.replace("°", "").replace(",", ".")
            altitude = columns[6].text.strip().replace(",", ".")
            records.append({
                'position': city,
                'lat': float(latitude),
                'long': float(longitude),
                'height': float(altitude)
            })

    # result list with data
    return(records)
