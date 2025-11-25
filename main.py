
import os, sys, time, random, math, io, numpy as np

class Race():
    def __init__(self, name, lifespan, reproduction_rate, fighting_ability):
        self.name = name
        self.lifespan = lifespan  # in years
        self.reproduction_rate = reproduction_rate  # average number of offspring per year in thousands
        self.fighting_ability = fighting_ability  # scale from 1 to 10

class World():
    def __init__(self, races_positions:dict):
        self.races_positions = {} # for example: {"Elves": [(0, 0), (7, 8)], "Dwarves": [(1, 1), (12, 12)]}

    def display_map():
        for race, position in self.races_positions.items():
            print()


if __name__ == "__main__":
    elves = Race("Elves", 1500, 0.1, 7)
    dwarves = Race("Dwarves", 400, 1.0, 8)
    orcs = Race("Orcs", 40, 3.0, 9)
    humans = Race("Humans", 80, 10.0, 4)
    hobbits = Race("Hobbits", 100, 2.0, 2)

    
