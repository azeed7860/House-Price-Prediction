# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load your dataset (assuming you have a CSV file, adjust accordingly)
# Replace 'your_dataset.csv' with your actual dataset file
df = pd.read_csv('homes.csv')

print(df.head())

# Assuming 'X' contains your features and 'y' contains your target variable
x= df[['SqFt','BedRooms','Baths']]

y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40)

# Create and train your model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model to a file
# Save the model
with open("reg_model (1).pkl", "wb") as file:
    pickle.dump(model, file)
