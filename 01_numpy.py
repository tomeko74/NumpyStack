import numpy as np

L = [1, 2, 3]
print(L)

A = np.array([1, 2, 3])
print(A)

for e in L:
    print(e)

for e in A:
    print(e)

L.append(4)
print(L)

# dla A nie ma łatwego append

# + w listach to konkatenacja
L = L + [5]
print(L)

# dla A nie ma łatwego dodania elementu do tablicy

# ale jak chcemy dodać do elementów listy to:
L2 = []
for e in L:
    L2.append(e + e)
print(L2)

# + w tablicach to wektoryzacja (suma po elementach)
# dla macierzy wielowymiarowej to będzie matematyczne dodawanie macierzy
print(A + A)

# mnożenie wektora przez skalar
print(2 * A)

# w listach mnożenie dokleja kolejne kopie
print(2 * L)

# jak chcemy podnieść do kwadratu elementy listy:
L2 = []
for e in L:
    L2.append(e ** 2)
print(L2)

# a dla tablicy mamy:
print(A ** 2)

# np. pierwiastek z elementów:
print(np.sqrt(A))

# np. logarytm z elementów:
print(np.log(A))

# np. funkcja eksponencjalna z elementów:
print(np.exp(A))


# teraz iloczyn skalarny (dot product)
a = np.array([1, 2])
b = np.array([2, 1])

dot = 0
for e, f in zip(a, b):
    dot += e*f

print(dot) # na piechotę

print(np.sum(a*b))  # wykorzystując mnożenie elementów i potem sumę
print(np.dot(a, b))  # gotowa funkcja np
print(a.dot(b))  # jest to funkcja tablicy, można też tak
print(b.dot(a))  # albo tak

print(np.linalg.norm(a))  # norma macierzowa Frobeniusa (pierwiastek z sumy kwadratów elementów)

cosangle = a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))
angle = np.arccos(cosangle)
print(angle)  # kąt w radianach

M = np.array([[1, 2], [3, 4]])
L = [[1, 2], [3, 4]]

print(L[0][0])
print(M[0][0])
print(M[0, 0])

M2 = np.matrix([[1, 2], [3, 4]])  # można tez mieć taką macierz, jednak zaleca się stosowanie array
print(M2)

A = np.array(M2)
print(A)
print(A.T)  # macierz transponowana (wiersze jako kolumny i odwrotnie)

# jak tworzyć tablice
print(np.array([1, 2, 3]))

Z = np.zeros(10)
print(Z)

Z = np.zeros((10, 10))
print(Z)

Z = np.ones((10, 10))
print(Z)

Z = np.random.random((10, 10))  # liczby losowe między 0 a 1
print(Z)

G = np.random.randn(10, 10)  # liczby losowe w rozkładzie Gaussa
print(G)

print(G.mean())  # mean - średnia elementów tablicy
print(G.var())  # variance - wariancja - miara zmienności, średnia kwadratów odchyleń od wartości oczekiwanej

# ważne - w numpy A.dot(B) to mnożenie macierzy, ale A * B to mnożenie kolejnych elementów przez siebie
# w dot muszą być wymiary x(,y) dot (y, z), w mnożeniu (x, y) * (x, y)
# mnożenie elementów oznaczane jako kropka w kółku lub x w kółku
M1 = np.array([[1, 2, 3], [4, 5, 6]])
M1a = np.array([[4, 5, 6], [1, 2, 3]])
M2 = np.array([[1, 2], [3, 4], [5, 6]])

print(f"Mnożenie:\n {M1 * M1a}")
print(f"Product:\n {M1.dot(M2)}")

# macierz odwrotna, czyli taka, że zachodzi A.dot(Ainv) = Ainv.dot(A) = I (jednostkowa), nie każda ma odwrotną
A = np.array([[1, 1], [1, 4]])
Ainv = np.linalg.inv(A)
print(Ainv)
print(A.dot(Ainv))
print(Ainv.dot(A))

# wyznacznik macierzy
print(np.linalg.det(A))

# diag = przekątna macierzy (diagonal element)
print(np.diag(A))  # wyciąga z istniejącej
print(np.diag([1, 2]))  # tworzy macierz na podstawie przekątnej

a = np.array([1, 2])
b = np.array([3, 4])
# outer product
print(np.outer(a, b))
# inner product
print(np.inner(a, b)) # to samo co a.dot(b)

# matrix trace (ślad macierzy, suma elementów diagonalnych, na przekątnej)
print(np.trace(A))
print(np.diag(A).sum())  # to samo

# eigenvalues i eigenwectors (wartości i wektory własne)
X = np.random.randn(100, 3)  # 100 próbek, 3 zmienne
cov = np.cov(X.T)  # macierz kowariancji, musimy transponować, żeby mieć 3x3 a nie 100x100

# dla wszystkich macierzy symetrycznych
print(np.linalg.eig(cov))  # najpierw tuple z trzema eigenvalues, potem trzy wektory eigenwectors

# dla macierzy symetrycznych i hermitowskich
print(np.linalg.eigh(cov))  # najpierw tuple z trzema eigenvalues, potem trzy wektory eigenwectors

# solving linear system - układ równań
# problem Ax = b
# A is matrix, x is column vector of values we are trying to solve for, b is vector of numbers

#           -1          -1
# solution A  Ax = x = A  b
A = np.array([[1, 2], [3, 4]])
b = np.array([1, 2])
x = np.linalg.inv(A).dot(b)
print(x)
print(A.dot(x)) # sprawdzam czy wyszło

# w numpy to samo robi
x = np.linalg.solve(A, b)
print(x)

# przykład ,wejściówka dla dziecka kosztuje 1.5$, dla dorosłego 4$, weszło 2200 ludzi i zebrano 5050$
# ile było dzieci a ile dorosłych
# x1 + x2 = 2200
# 1.5x1 + 4x2 = 5050
A = np.array([[1, 1], [1.5, 4]])
b = np.array([2200, 5050])
x = np.linalg.solve(A, b)
print(x)  # czyli 1500 dzieci i 700 dorosłych


