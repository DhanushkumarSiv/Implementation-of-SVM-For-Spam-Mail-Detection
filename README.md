# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

 1.Import libraries.

2.Read the CSV file and display data using head().

3.Split the dataset using train_test_split().

4.Calculate predictions and accuracy.

5.Print the outputs.

6.End the program.

## Program:
```

Program to implement the SVM For Spam Mail Detection..
Developed by: DHANUSHKUMAR SIVAKUMAR
RegisterNumber:  212224040067

```
```
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
```
```
data = pd.read_csv("spam.csv", encoding="Windows-1252")
data.head()
```
```
data.info()
```
```
data.isnull().sum()
```
```
# separating the features and labels
x = data["v2"].values  # text messages
y = data["v1"].values  # labels: spam or ham
```
```
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
```
```
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)
```

```
svc = SVC()
svc.fit(x_train, y_train)
```
```
y_pred = svc.predict(x_test)
y_pred
```
```
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100}%")
```

## Output:

Head Values

![Screenshot 2025-05-14 200854](https://github.com/user-attachments/assets/5d01ffde-2625-4877-b0ae-5023a5a9f3a6)

Dataframe Info

![Screenshot 2025-05-14 200904](https://github.com/user-attachments/assets/7eee7fb2-c1ab-448c-a866-335679801bfa)

Sum - Null Values

![Screenshot 2025-05-14 200911](https://github.com/user-attachments/assets/1ca00b61-cfaf-4e1f-8713-ad8333d3366c)

Training the model

![Screenshot 2025-05-14 200918](https://github.com/user-attachments/assets/1b8cb779-8562-43ed-aaf7-4b6ea51e2a29)

Predicting the test data

![Screenshot 2025-05-14 200931](https://github.com/user-attachments/assets/802c6a12-8836-4ef6-8c27-0d1e53f99030)

Accuracy

![Screenshot 2025-05-14 200937](https://github.com/user-attachments/assets/db1493f3-f12d-4f01-a764-df2bc6f11335)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
