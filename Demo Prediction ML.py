import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("music.csv")
X = data.drop(columns=["genre"])
y = data["genre"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

"""
X means inputset
y means outputset
"""

dtc = DecisionTreeClassifier()

dtc.fit(X_train, y_train) #train
predictions = dtc.predict(X_test)

print("predictions", predictions)

score = accuracy_score(y_test, predictions)
print(score)




"""

============= FINDING NORMAL PREDICTIONS =====================

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

dtc = DecisionTreeClassifier()
music_data = pd.read_csv("music.csv")
print(music_data)
X = music_data.drop(columns=["genre"])
y = music_data["genre"]

dtc.fit(X, y)
predictions = dtc.predict([[22,0],])

print("predictions", predictions)


===================================USING TRAIN SPLIT SELECTION======================================
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("music.csv")
X = data.drop(columns=["genre"])
y = data["genre"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

""
X means inputset
y means outputset
""

dtc = DecisionTreeClassifier()

dtc.fit(X_train, y_train) #train
predictions = dtc.predict(X_test)

print("predictions", predictions)

score = accuracy_score(y_test, predictions)
print(score)


"""