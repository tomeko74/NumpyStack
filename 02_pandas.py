import pandas as pd

X = pd.read_csv("./linear_regression_class/data_2d.csv", header=None)

print(X.info())
print(X.head())
print(X.head(10))

# M = X.as_matrix()  # stare wywołanie, teraz powinno być jak niżej
M = X.values

# uwaga, w numpy X[0] -> zerowy wiersz, w pandas X[0] - kolumna o nazwie 0
print(X[0])
print(M[0])

# jak w pandas dostać wiersz?
print(X.iloc[0])

# lub
print(X.ix[0])

# można dostać np. 0 i 2 kolumnę:
print(X[[0, 2]])

# można dostać np. wiersze gdzie wartość 0 kolumny jest mniejsza niż 5
print(X[X[0] < 5])

# jak widać sam warunek zwraca kolumnę booleanów czy mniejszy czy nie
print(X[0] < 5)

# we need to skip the 3 footer rows
# skipfooter does not work with the default engine, 'c'
# so we need to explicitly set it to 'python'
df = pd.read_csv('./airline/international-airline-passengers.csv', engine='python', skipfooter=3)

# rename the columns because they are ridiculous
df.columns = ['month', 'passengers']
print(df.columns)
print(df['passengers'])
print(df.passengers)

# dodajemy nową kolumnę jedynek do danych
df['ones'] = 1

print(df.head())

from datetime import datetime
print(datetime.strptime("1949-05", "%Y-%m"))

df['dt'] = df.apply(lambda row: datetime.strptime(row['month'], "%Y-%m"), axis=1)


# teraz join
t1 = pd.read_csv('table1.csv')
print(t1)
t2 = pd.read_csv('table2.csv')
print(t2)

# właściwy join po polu 'user_id'
m = pd.merge(t1, t2, on='user_id')
print(m)

# można to też zrobić w inny sposób
print(t1.merge(t2, on='user_id'))
