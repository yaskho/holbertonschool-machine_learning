#!/usr/bin/env python3
"""
Module to fetch home planets of all sentient species from SWAPI.
"""
import requests


def sentientPlanets():
    """
    Returns a list of names of the home planets of all sentient species.

    Returns:
        list: List of planet names for all sentient species.
    """
    url = "https://swapi-api.hbtn.io/api/species/"
    planets = []

    while url:
        res = requests.get(url)
        if res.status_code != 200:
            break
        data = res.json()
        results = data.get("results", [])

        for species in results:
            classification = species.get("classification")
            designation = species.get("designation")

            if classification == "sentient" or designation == "sentient":
                homeworld = species.get("homeworld")
                if homeworld:
                    planet_res = requests.get(homeworld)
                    if planet_res.status_code == 200:
                        planet_name = planet_res.json().get("name")
                        if planet_name:
                            planets.append(planet_name)

        url = data.get("next")

    return planets
