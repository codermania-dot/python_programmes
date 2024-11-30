import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load data
music_data = pd.read_csv('music.csv')


X = music_data.drop(columns=['genre']) #input
Y = music_data['genre'] #output       

#training
model = DecisionTreeClassifier()
model.fit(X, Y)
#predictions
prediction_data = pd.DataFrame([[21, 1], [23, 0]], columns=X.columns)

predictions = model.predict(prediction_data)
print(predictions)
