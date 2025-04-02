'''
Ming Jin Yong
ACAD 222, Spring 2025
mingjiny@usc.edu
Homework 2
'''

import random

def jumble_word(word):
    # Convert word to list of characters
    chars = list(word)
    jumbled = []
    
    # Keep removing random characters from chars and append to jumbled
    while chars:
        index = random.randint(0, len(chars) - 1)
        jumbled.append(chars.pop(index))
    
    return ''.join(jumbled)

def play_word_jumble():
    # List of words for the game
    words = ['platypus', 'elephant', 'giraffe', 'mouse', 'tiger', 'moose', 'horse', 'snake', 'rabbit', 'lizard']
    
    # Choose a random word
    word = random.choice(words)
    jumbled = jumble_word(word)
    
    print(f'The jumbled word is "{jumbled}"')
    
    # Keep track of guesses
    tries = 0
    while True:
        guess = input('Please enter your guess: ')
        tries += 1
        
        if guess.lower() == word:
            print('You got it!')
            print(f'It took you {tries} tries.')
            break
        else:
            print('Try again.')

def create_cipher(shift):
    # Create the original alphabet
    alphabet = list('abcdefghijklmnopqrstuvwxyz')
    # Create cipher using slicing
    cipher = alphabet[shift:] + alphabet[:shift]
    return alphabet, cipher

def encrypt_decrypt_message():
    message = input('Enter a message: ')
    shift = int(input('Enter a number to shift by (0-25): '))
    
    alphabet, cipher = create_cipher(shift)
    
    # Encrypt
    print('Encrypting message....')
    encrypted = ''
    for char in message.lower():
        if char.isalpha():
            index = alphabet.index(char)
            encrypted += cipher[index]
        else:
            encrypted += char
    print(f'Encrypted message: {encrypted}')
    
    # Decrypt
    print('Decrypting message....')
    decrypted = ''
    for char in encrypted:
        if char.isalpha():
            index = cipher.index(char)
            decrypted += alphabet[index]
        else:
            decrypted += char
    print(f'Decrypted message: {decrypted}')
    print(f'Original message: {message}')

def main():
    print("Part 1 - Word Jumble Game")
    play_word_jumble()
    print("Part 2 - Encrypt/Decrypt")
    encrypt_decrypt_message()

if __name__ == "__main__":
    main()

