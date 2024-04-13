TITLE: CodTech IT Solutions Internship - Task Documentation: “TITANIC SURVIVAL PREDICTION” Using Python programming
INTERN INFORMATION: 
Name: Dibjyoti Hota
ID: ICOD6251

INTRODUCTION

The sinking of the RMS Titanic on its maiden voyage in April 1912 remains one of the most infamous maritime disasters in history. The tragic event resulted in the loss of over 1,500 lives, sparking global sorrow and raising questions about maritime safety and emergency preparedness. In the decades since, the Titanic disaster has captivated the public imagination, serving as the backdrop for countless books, films, and studies.
In this project, we embark on a journey to explore the Titanic dataset and employ machine learning techniques to predict the survival outcome of passengers. Our objective is to build a model that accurately determines whether a passenger aboard the Titanic survived or perished based on various demographic and socio-economic factors.
The dataset provides a treasure trove of information about individual passengers, including their age, gender, ticket class, fare, cabin, and ultimately, whether they survived the ordeal. By leveraging this rich dataset and employing predictive modeling techniques, we aim to uncover patterns and insights that shed light on the factors influencing survival rates aboard the ill-fated Titanic.
Through meticulous data pre-processing, exploratory data analysis (EDA), feature engineering, and model selection and training, we will meticulously craft a predictive model capable of discerning survival outcomes. Our approach involves employing popular machine learning algorithms, such as Random Forest, to harness the predictive power hidden within the data.
By the culmination of this project, we not only seek to build a robust predictive model but also to contribute to the broader understanding of the Titanic disaster. Our endeavor is not merely an exercise in data science but a tribute to the memory of those who perished aboard the Titanic and a testament to the enduring fascination with this tragic event.
Join us as we embark on this voyage through data and embark on a quest to unveil the hidden truths of the Titanic disaster, one prediction at a time.




Implementation
	Python Programming: Used python to build the titanic survival prediction
	Numpy/pandas: use of numpy and pandas modules to store data and to calculate mathematical problems matplot and sklearn is used.
	Responsive Design: Implement responsive design principles to ensure optimal viewing experience across desktop and mobile devices.
	User Interface Components: Utilize UI libraries for designing interactive and visually appealing components.


CODE EXPLAINATION

# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load the dataset
titanic_data = pd.read_csv("train.csv")

# Step 3: Data Preprocessing
# Check for missing values
print(titanic_data.isnull().sum())

# Handle missing values
titanic_data["Age"].fillna(titanic_data["Age"].median(), inplace=True)
titanic_data["Embarked"].fillna(titanic_data["Embarked"].mode()[0], inplace=True)

# Convert categorical variables into numerical
titanic_data = pd.get_dummies(titanic_data, columns=["Sex", "Embarked"])

# Drop irrelevant features
titanic_data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

# Step 4: Exploratory Data Analysis (EDA)
sns.countplot(x="Survived", data=titanic_data)
plt.show()

sns.countplot(x="Survived", hue="Sex_male", data=titanic_data)
plt.show()

sns.countplot(x="Survived", hue="Pclass", data=titanic_data)
plt.show()

# Step 5: Feature Engineering (No additional features in this example)

# Step 6: Model Selection and Training
X = titanic_data.drop("Survived", axis=1)
y = titanic_data["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Model Training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

# Step 8: Model Evaluation
y_pred = rf_classifier.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))











	OUTPUTS 

 

	
       







USAGE
Titanic survival prediction: Users can use this machine learning model to predict the survival rates
Observation: Users can know all sorts male and female population with this dataset
Easiness: Easiness in sorting the data and the survival rates.

CONCLUSION
In conclusion, the Titanic dataset to build a model that predicts whether a passenger on the Titanic survived or not. This is a classic beginner project with readily available data. . The dataset typically used for this project contains information about individual passengers, such as their age, gender, ticket class, fare, cabin, and whether or not they survived.







