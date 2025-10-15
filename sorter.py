import os
import numpy as np
import pandas as pd
import re


path = os.getcwd()
data_path = path + '/data'

my_df = pd.read_csv(path + '/mine.csv')
nhl_df = pd.read_csv(path + '/nhl.csv')

all_entries_list = nhl_df.values.flatten().tolist()

for ent in all_entries_list:
    print(ent)


# The regular expression pattern
# r'^\d+\.\s*(.+?),\s*[A-Z]{2,}\s*\('
# ^\d+\.\s* -> Match the start of the string, one or more digits, a literal period, and any whitespace.
# (.+?)        -> **Capture Group 1 (the name):** Match one or more characters (.+), non-greedily (?), until...
# ,\s*[A-Z]{2,}\s*\( -> ...a comma, optional whitespace, 2 or more capital letters (the team code), optional whitespace, and the literal opening parenthesis.
pattern = r'^\d+\.\s*(.+?),\s*[A-Z]{2,}\s*\('

names_list = []
for item in nhl_df:
    match = re.search(pattern, item)
    if match:
        # The name is in the first (and only) capture group
        names_list.append(match.group(1).strip())
    else:
        # Handle cases where the pattern isn't found
        names_list.append(None) 

print(names_list)
# Output: ['Andrei Kuzmenko', 'Connor McDavid', 'Elias Pettersson']