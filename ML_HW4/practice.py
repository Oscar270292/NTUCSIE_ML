import random


random_numbers = random.sample(range(0, 200), 120)
all_numbers = list(range(0, 200))
remaining_numbers = [num for num in all_numbers if num not in random_numbers]

print(random_numbers)
print(len(random_numbers))
print(remaining_numbers)
print(len(remaining_numbers))