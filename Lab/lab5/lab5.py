'''
Ming Jin Yong
ACAD 222, Spring 2025
mingjiny@usc.edu
Lab 5
'''

def loadFile(filename, tallies):
    try:
        with open(filename, 'r', encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return False

    if not lines:
        return False

    # skip the header line
    for line in lines[1:]:
        line = line.strip()
        if line == "":
            continue
        # Split the line by comma.
        parts = line.split(',')
        if len(parts) < 2:
            continue
        subreddit = parts[0].strip()
        count = parts[1].strip()
        tallies[subreddit] = count
    return True

def main():
    tallies = {}
    while True:
        filename = input("Tallies file: ")
        if filename == "":
            print("Goodbye!")
            return

        if not loadFile(filename, tallies):
            print(f'Could not open the file "{filename}"')
        else:
            break

    print("Loading data...")
    
    while True:
        subreddit = input("Subreddit: ")
        if subreddit == "":
            print("Goodbye!")
            return
        if subreddit not in tallies:
            print(f'"{subreddit}" is an unknown subreddit')
        else:
            print(f'{subreddit} --> {tallies[subreddit]}')

main()
