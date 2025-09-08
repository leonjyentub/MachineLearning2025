import numpy as np
import pandas as pd

np.random.seed(42)

# Generate study hours data
study_hours = np.random.normal(loc=5, scale=2, size=100)
study_hours = np.clip(study_hours, 0, 10)  # Clip values between 0 and 10 hours

# Generate exam scores based on study hours
exam_scores = 60 + 5 * study_hours + \
    np.random.normal(loc=0, scale=10, size=100)

# Determine pass/fail status (1 if score >= 70, 0 otherwise)
pass_fail = (exam_scores >= 70).astype(int)

# Create a DataFrame
data = pd.DataFrame({
    'study_hours': study_hours,
    'pass_fail': pass_fail
})

# Save to CSV
data.to_csv('student_exam_data.csv', index=False)

print(data.head())
print(f"\nShape of the dataset: {data.shape}")
print(f"\nNumber of students who passed: {data['pass_fail'].sum()}")
print(f"Number of students who failed: {len(data) - data['pass_fail'].sum()}")
