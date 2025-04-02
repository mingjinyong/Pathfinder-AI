'''
Ming Jin Ynng
ACAD 222, Spring 2025
mingjiny@usc.edu
Homework 1
'''

name = input("Hello, whose BMI shall I calculate? ")
print("Okay first I need " + name + "'s height. I'll take it in feet and inches.")
feet = int(input("Feet first... "))
inches = int(input("Enter inches... "))
print("Thanks. Now I need Sally's weight in pounds.")
pounds = int(input("Please enter " + name + "'s weight... "))

heightInches = (feet * 12) + inches
heightMeters = heightInches / 39.37
weightKilograms = pounds / 2.2

bmi = weightKilograms / (heightMeters*heightMeters)
print(name + "'s BMI " + "{:.1f}".format(bmi))

