import pandas as pd
import sqlalchemy as db
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
# Load the data into a pandas DataFrame
df = pd.read_csv('D:\code_panra\population.csv')

# Display the first few rows
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values or fill them with a suitable value
df = df.dropna()  # or df.fillna(method='ffill', inplace=True)

# Convert the 'Year' column to a datetime object (if necessary)
df['Year'] = pd.to_datetime(df['Year'], format='%Y')

# Ensure the 'Value' column is numeric (population values)
df['Value'] = df['Value'].astype(int)

# Display the cleaned data
print(df.info())

# Create an SQLAlchemy engine to connect to the database
engine = create_engine('sqlite:///world_population.db')

# Import the DataFrame into an SQL table
df.to_sql('population', con=engine, if_exists='replace', index=False)

# Create an SQLAlchemy engine to connect to the database
engine = create_engine('sqlite:///world_population.db')

# Create a connection
connection = engine.connect()

# Example SQL query: Calculate total population per year globally
query = text("""
SELECT Year, SUM(Value) as Total_Population
FROM population
GROUP BY Year
ORDER BY Year;
""")

# Execute the query and fetch the results
result = connection.execute(query).fetchall()

# Close the connection
connection.close()

# Print the results
for row in result:
    print(row)

# Calculate population growth rate per country
df['Growth_Rate'] = df.groupby('Country Code')['Value'].pct_change() * 100

# Display the first few rows with growth rates
print(df[['Country Name', 'Year', 'Value', 'Growth_Rate']].head())

# Population trend over the years for a specific country
plt.figure(figsize=(10, 6))
sns.lineplot(data=df[df['Country Name'] == 'India'], x='Year', y='Value')
plt.title('Population Trend Over Years for India')
plt.show()

# Population distribution by country (top 10 countries)
top_countries = df.groupby('Country Name')['Value'].max().nlargest(10).index
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[df['Country Name'].isin(top_countries)], x='Country Name', y='Value')
plt.title('Population Distribution by Top 10 Countries')
plt.show()

# Prepare data for prediction (e.g., for the United States)
usa_df = df[df['Country Name'] == 'United States']
X = usa_df['Year'].apply(lambda x: x.year).values.reshape(-1, 1)
y = usa_df['Value'].values

# Fit the model
model = LinearRegression()
model.fit(X, y)

# Predict future population for the United States
future_years = [[2025], [2030], [2035]]
predictions = model.predict(future_years)

print("Predicted populations for the United States:", predictions)
