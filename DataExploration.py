import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('Bicycle_Thefts.csv')

# Basic information
print(data.head())
print(data.info())
print(data.describe())
print(f"Shape of data: {data.shape}")

# Column summaries
for column in data.columns:
    print(f"Column: {column}")
    print(f"Data Type: {data[column].dtype}")
    if data[column].dtype == 'object':
        print(f"Unique Values: {data[column].nunique()}")
        print(data[column].value_counts().head())
    elif np.issubdtype(data[column].dtype, np.number):
        print(f"Range: {data[column].min()} to {data[column].max()}")
    print("-" * 40)

# Missing data summary
missing_counts = data.isnull().sum()
missing_percentage = data.isnull().mean() * 100
missing_summary = pd.DataFrame({
    'Missing Count': missing_counts,
    'Missing Percentage': missing_percentage
})
print(missing_summary)

# Statistical summaries for numeric data
numeric_data = data.select_dtypes(include=[np.number])
numeric_summary = pd.DataFrame({
    'Mean': numeric_data.mean().round(2),
    'Median': numeric_data.median().round(2),
    'Mode': numeric_data.mode().iloc[0].round(2),
    'Standard Deviation': numeric_data.std().round(2)
})
print(numeric_summary)

# Correlation analysis
correlations = numeric_data.corr()
print(correlations)

# Heatmap visualization
plt.figure(figsize=(16, 12))
sns.heatmap(correlations, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Outlier detection
Q1 = numeric_data.quantile(0.25)
Q3 = numeric_data.quantile(0.75)
IQR = Q3 - Q1
outliers = ((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).sum()
print("Potential outliers per column:")
print(outliers)

# Bicycle Thefts over years
data['OCC_YEAR'] = pd.to_datetime(data['OCC_YEAR'], format='%Y')
thefts_per_year = data.groupby(data['OCC_YEAR'].dt.year).size()
plt.figure(figsize=(10, 6))
thefts_per_year.plot(kind='line', marker='o')
plt.title('Bicycle Thefts Over Years')
plt.xlabel('Year')
plt.ylabel('Number of Thefts')
plt.grid(True)
plt.show()

# Thefts by day of the week
data['OCC_DOW'] = pd.Categorical(data['OCC_DOW'],
                                 categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',
                                             'Sunday'],
                                 ordered=True)
thefts_by_day = data['OCC_DOW'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
plt.bar(thefts_by_day.index, thefts_by_day.values, color='skyblue')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Thefts')
plt.title('Thefts by Day of the Week')
plt.xticks(rotation=45)
plt.show()