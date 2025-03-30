import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay

class points:
    def __init__(self, x, y, pType):
        self.x = x
        self.y = y
        self.pType = pType

data = pd.read_csv('dataset.csv')

xValues = data['sepal.length'].values
yValues = data['sepal.width'].values
pTypes = data['variety'].values

plt.figure(figsize=(12, 5))

X = data[['sepal.length', 'sepal.width']].values
y = data['variety'].values

clf = KNeighborsClassifier(n_neighbors=7)

clf.fit(X, y)

disp = DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    response_method="predict",
    plot_method="pcolormesh",
    xlabel='Sepal Length',
    ylabel='Sepal Width',
    shading="auto",
    alpha=0.5,
)

i = 0
for x in xValues:
    point = points(x, yValues[i], pTypes[i])
    colors = ' '
    match point.pType:
        case "Versicolor":
            colors = 'teal'
        case "Setosa":
            colors = 'purple'
        case "Virginica":
            colors = 'gold'
    plt.scatter(point.x, point.y, color=colors, s= 20)
    i += 1

plt.scatter([], [], color='teal', label="Versicolor")
plt.scatter([], [], color='purple', label="Setosa")
plt.scatter([], [], color='gold', label="Virginica")

plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.legend(title='Classes')

plt.show()
