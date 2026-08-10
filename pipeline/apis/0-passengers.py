#!/usr/bin/env python3
"""
Module to fetch available starships from SWAPI based on passenger capacity.
"""
import requests


def availableShips(passengerCount):
    """
    Returns a list of starships that can hold a given number of passengers.

    Args:
        passengerCount (int): Minimum required passenger capacity.

    Returns:
        list: List of starship names that hold at least passengerCount.
    """
    url = "https://swapi-api.hbtn.io/api/starships/"
    ships = []

    while url:
        res = requests.get(url)
        if res.status_code != 200:
            break
        data = res.json()
        results = data.get("results", [])

        for ship in results:
            passengers = ship.get("passengers", "").replace(",", "")
            try:
                if int(passengers) >= passengerCount:
                    ships.append(ship.get("name"))
            except ValueError:
                continue

        url = data.get("next")

    return ships
