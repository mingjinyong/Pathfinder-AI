'''
Ming Jin Yong
ACAD 222, Spring 2025
mingjiny@usc.edu
Lab 3
'''

# input
sentence = input("What shall I censor: ")

# if user enters nothing
if sentence == "":
    print("Goodbye!")

# censor list
censor = ['fren', 'fral', 'drel', 'gron', 'glud', 'zarp', 'nark']

# loop through each word in the censor list
for word in censor:
    # keep replacing the word when it exists in the sentence
    while word in sentence.lower():
        # find the starting position of the censored word
        index = sentence.lower().find(word)
        # replace the word by slicing the string
        sentence = sentence[:index] + "BEEP" + sentence[index + len(word):]
# print the final censored sentence
print(sentence)