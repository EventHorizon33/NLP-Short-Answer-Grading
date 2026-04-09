from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from preproc import main

# Run preprocessing to get the data
X_train, X_dev, X_test, y_train, y_dev, y_test, vectorizer, label_encoder = main(train_path=r"C:\Users\vaibh\Desktop\NLP Project\semeval-2013-train.csv", dev_path=r"C:\Users\vaibh\Desktop\NLP Project\semeval-2013-dev.csv", test_path=r"C:\Users\vaibh\Desktop\NLP Project\semeval-2013-test.csv")

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_dev)

print("Accuracy:", accuracy_score(y_dev, y_pred))
print(classification_report(y_dev, y_pred))