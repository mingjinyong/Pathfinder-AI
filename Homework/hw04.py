'''
Ming Jin Yong
ACAD 222, Spring 2025
mingjiny@usc.edu
Homework 4
'''
import random

# Roll the dice
def roll():
    roll1 = random.randint(1, 8)
    roll2 = random.randint(1, 8)
    roll3 = random.randint(1, 8)
    return [roll1, roll2, roll3]

# Compute the bet result
def computeBetResult(rolls, bet, guessed):
    guessed = int(guessed)
    matches = rolls.count(guessed)
    print(f"You matched {matches} dice!")
    
    if matches == 0:
        return -bet  # Lost bet
    elif matches == 1:
        print(f"YOU WIN ${bet}")
        return bet  # Win 1x bet
    elif matches == 2:
        winnings = bet * 3
        print(f"YOU WIN ${winnings}")
        return winnings  # Win 3x bet
    else:  # matches == 3
        winnings = bet * 10
        print(f"YOU WIN ${winnings}")
        return winnings  # Win 10x bet

def main():
    money = 200
    
    print("STEP RIGHT UP AND PLAY SOME CHUCK-A-LUCK!")
    while money > 0:
        print(f"You have ${money}")
        
        # Get valid bet amount
        while True:
            bet = int(input("Enter bet amount: $"))
            if 0 < bet <= money:
                break
            print(f"Invalid bet. Please enter an amount from $1 to ${money}")
        
        # Get valid guess
        while True:
            guessed = input("What number do you want to bet on (1-8)? ")
            if guessed.isdigit() and 1 <= int(guessed) <= 8:
                break
            print("Invalid number. Must be between 1 and 8.")
            
        print(f"You bet ${bet} on {guessed}")
        print("You rolled:")
        rolls = roll()
        print(", ".join(str(x) for x in rolls))
        
        # Calculate and update money
        result = computeBetResult(rolls, bet, guessed)
        money += result
        
        if result < 0:
            print("You lost your bet :(")
        print(f"You now have ${money}")
        
        # Check if player wants to continue
        play_again = input("Would you like to play again (y/n)? ").lower()
        if play_again != 'y':
            break
            
    print(f"You ended the game with ${money}")

main()