'''
Ming Jin Yong
ACAD 222, Spring 2025
mingjiny@usc.edu
Lab 4
'''

import random

def get_factors(num):
    """Return a list of factors between 2 and 55 for the given number."""
    factors = []
    for i in range(2, 56):
        if num % i == 0:
            factors.append(i)
    return factors

def main():
    while True:
        # Get input from user
        user_input = input("Give me a number: ")
        
        # Check if user wants to quit
        if user_input.strip() == "":
            break
            
        try:
            n = int(user_input)
            
            # Initialize dictionary with factors 2-55 set to 0
            factor_counts = {i: 0 for i in range(2, 56)}
            
            # Generate n random numbers and count their factors
            for _ in range(n):
                num = random.randrange(40001)  # 0 to 40,000
                factors = get_factors(num)
                for factor in factors:
                    factor_counts[factor] += 1
            
            # Display results
            print(f"\nFor {n} random numbers, the factor frequencies are:")
            for factor in range(2, 56):
                count = factor_counts[factor]
                if count > 0:  # Only display factors with non-zero counts
                    stars = '*' * count
                    print(f"{factor} : {stars}")
            print()  # Add blank line between runs
            
        except ValueError:
            print("Please enter a valid number or press Enter to quit.\n")

if __name__ == "__main__":
    main()
