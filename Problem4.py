from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Get the Iris dataset
iris = datasets.load_iris()

# Get the features and labels
X = iris.data
y = iris.target

# Splitting data into train and testing part 75:25 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# PCA(Dimension reduction to two) -> Scaling the data -> RandomForestClassifier
pipe = Pipeline([('pca', PCA(n_components = 2)), ('std', StandardScaler()), ('decision_tree', RandomForestClassifier())], verbose = True)

# fitting the data in the pipe
pipe.fit(X_train, y_train)

# Computing accuracy score
print(accuracy_score(y_test, pipe.predict(X_test)))

#To get the steps
print(pipe.named_steps)

print()

# To get the parameters
print(pipe.get_params())