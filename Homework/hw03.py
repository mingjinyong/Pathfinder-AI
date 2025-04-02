'''
Ming Jin Yong
ACAD 222, Spring 2025
mingjiny@usc.edu
Homework 3
'''

# P1: 
# Create months dictionary mapping month names to number of days
months = {
    "January": 31,
    "February": 28, 
    "March": 31,
    "April": 30,
    "May": 31,
    "June": 30,
    "July": 31,
    "August": 31,
    "September": 30,
    "October": 31,
    "November": 30,
    "December": 31
}

# P2: 
# Create calendar dictionary empty lists for each month
calendar = {}
for month, days in months.items():
    calendar[month] = [""] * days

# P3: 
# Get user input for dates and events
while True:
    date_input = input("Enter a date for a holiday (for example \"July 1\"): ").strip()
    
    if date_input == "":
        break
        
    # Split input into month and day
    parts = date_input.split()
    if len(parts) != 2:
        print("I don't see good input in there!")
        continue
        
    month, day = parts[0], parts[1]
    
    # Validate month
    if month not in months:
        print(f"I don't know about the month \"{month}\"")
        continue
        
    # Validate day
    try:
        day_num = int(day)
        if day_num < 1 or day_num > months[month]:
            print(f"That month only has {months[month]} days!")
            continue
    except ValueError:
        print("I don't see good input in there!")
        continue
        
    # Get event description
    event = input(f"What happens on {month}, {day}? ")
    
    # Store in calendar (subtract 1 from day for 0-based index)
    calendar[month][day_num-1] = event

# P4: 
# Display all dates with events
for month in months:
    for day_idx, event in enumerate(calendar[month]):
        if event:  # Only display non-empty events
            print(f"{month} {day_idx+1} : {event}")

print("Goodbye!")
