import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# linear plot
x = np.linspace(0, 10, 101)  # zwraca tablicę 11 elementów od 0 do 10
print(x)

y = np.sin(x)

plt.plot(x, y)
plt.xlabel("Time")
plt.ylabel("Some function of time")
plt.title("My cool chart")

plt.show()


# scatter plot
A = pd.read_csv("./linear_regression_class/data_1d.csv", header=None).values

x = A[:, 0]  # pierwsza kolumna
y = A[:, 1]  # druga kolumna

# wykres punktowy z danych
plt.scatter(x, y)

# na oko dopasowujemy linię, żeby pokazać, ze można rysować różne rzeczy jednocześnie
x_line = np.linspace(0, 100, 101)
y_line = 2*x_line + 1

plt.plot(x_line, y_line)

plt.show()

# histogram z danych x
plt.hist(x)
plt.show()

# wygenerujmy losowo, żeby zobaczyć rozkład
R = np.random.random(100000)
plt.hist(R)
plt.show()

# teraz zamiast domyślnie podzielić histogram na 10, dzielimy na 50 próbek
plt.hist(R, bins=50)
plt.show()

# sprawdźmy czy dane są w rozkładzie normalnym i tak też różnią się od założonej kreski (ma być wykres r. normalnego)
y_actual = 2*x + 1
residuals = y - y_actual
plt.hist(residuals)
plt.show()

df = pd.read_csv("./mnist/train.csv")
print(df.shape)
M = df.values

im = M[0, 1:]  # pierwszy wiersz bez pierwszej kolumny (tam jest labelka co to faktycznie jest)
print(im)  # 784, ale chcemy 28x28, bo to obrazek
im = im.reshape(28,28)
print(im)

plt.imshow(im, cmap='gray')
plt.show()
print(M[0, 0]) # opis, tak to jedynka
