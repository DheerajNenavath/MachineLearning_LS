import pandas as pd
import numpy as np
import random

names = [f'Student{i}' for i in range(10)]
subjects = ['Math', 'Science', 'English']
df = pd.DataFrame({
    'Name': names,
    'Subject': [random.choice(subjects) for _ in range(10)],
    'Score': np.random.randint(50, 101, 10),
    'Grade': ''
})

# Assigning grades
def assign_grade(score):
    if score >= 90: return 'A'
    elif score >= 80: return 'B'
    elif score >= 70: return 'C'
    elif score >= 60: return 'D'
    else: return 'F'

df['Grade'] = df['Score'].apply(assign_grade)
print("Graded DataFrame:\n", df)

# Sorting by score
print("Sorted by Score:\n", df.sort_values(by='Score', ascending=False))

# Average score per subject
print("Average score per subject:\n", df.groupby('Subject')['Score'].mean())

# Function to filter A/B grades
def pandas_filter_pass(dataframe):
    return dataframe[dataframe['Grade'].isin(['A', 'B'])]

print("Filtered A/B students:\n", pandas_filter_pass(df))
