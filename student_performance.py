import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# 1. Load the student dataset
data = pd.read_csv("student_data.csv")

print("Student Data:")
print(data)


# 2. Select input features and target
X = data[[
    "study_hours",
    "attendance",
    "previous_score",
    "assignments_completed"
]]

y = data["final_score"]


# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4. Create and train the Machine Learning model
model = LinearRegression()
model.fit(X_train, y_train)


# 5. Make predictions on test data
y_pred = model.predict(X_test)


# 6. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Final Score")
plt.ylabel("Predicted Final Score")
plt.title("Actual vs Predicted Student Scores")

plt.show()

# 7. Take input from a new student
print("\n--- Student Performance Prediction ---")

study_hours = float(input("Enter study hours per day: "))
attendance = float(input("Enter attendance percentage: "))
previous_score = float(input("Enter previous score: "))
assignments_completed = float(input("Enter assignment completion percentage: "))


# 8. Predict the student's final score
new_student = [[
    study_hours,
    attendance,
    previous_score,
    assignments_completed
]]

prediction = model.predict(new_student)[0]

# Keep prediction between 0 and 100
prediction = max(0, min(100, prediction))

print("\nPredicted Final Score:", round(prediction, 2))


# 9. Give a simple performance message
if prediction >= 75:
    print("Performance Prediction: Excellent")
elif prediction >= 60:
    print("Performance Prediction: Good")
elif prediction >= 50:
    print("Performance Prediction: Average")
else:
    print("Performance Prediction: Needs Improvement")