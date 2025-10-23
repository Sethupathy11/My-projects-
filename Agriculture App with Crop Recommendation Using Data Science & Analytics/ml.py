import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('hii.csv')

print(data)

print(data.info())

print(data.isnull())
data_mean = data.fillna(data.mean(numeric_only=True))

label_map = {'rice': 1, 'maize': 2, 'chickpea': 3, 'kidneybeans': 4}
data_mean['label'] = data_mean['label'].map(label_map)
X = data_mean.drop("label", axis=1)
Y = data_mean["label"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(X_train)
print(Y_train)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, Y_train)
print(data_mean)


print("Encoded table:\n", data_mean)
Y_pred = model.predict(X_test)
print("Predictions:", Y_pred)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")



print(X_train.columns.tolist())

sample = {
    'N': [90],
    'P': [42],
    'K': [43],
    'temperature': [26.0],
    'humidity': [80],
    'ph': [6.5],
    'rainfall': [200]
}


sample_df = pd.DataFrame(sample)


prediction = model.predict(sample_df)
print("Prediction:", prediction[0])

if prediction[0]==1:
  print("rice")
elif  prediction[0]==2:
  print("maize")  
elif prediction[0]==3:
  print("chickpea")
elif prediction[0]==4:
  print("kidneybeans")
else :
  print("Nothing can be seed to this Climate")