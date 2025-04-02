'''
Ming Jin Yong
ACAD 222, Spring 2025
mingjiny@usc.edu
Lab 1
'''

special = input("Hi welcome to the Wait Restaurant, which special would you like to get? Our choices are COLOR, NUMBER, and LETTER. ").lower()

if special == "color":
    choice = input("What would you like from the COLOR special? We have: Blue Drink, Green Burger, Red Fries, Yellow Nuggets ")

elif special == "number":
    choice = input("What would you like from the NUMBER special? We have: 1 Drink, 2 Burger, 3 Fries, 4 Nuggets ")

elif special == "letter":
    choice = input("What would you like from the LETTER special? We have: A Drink, B Burger, C Fries, D Nuggets ")

print("One", choice.upper(), "from", special.upper(), "special coming right up!")