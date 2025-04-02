'''
Ming Jin Yong
ACAD 222, Spring 2025
mingjiny@usc.edu
Lab 2
'''

name = input("What is your name (enter to quit)? ")
strLength = len(name)

vowels = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
vowCount = 0

i = 0

if strLength != 0:
    while strLength > i:
        if name[i] in vowels:
            vowCount += 1
        i += 1

    print(vowCount)