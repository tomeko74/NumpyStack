from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

print(norm.pdf(0))  # PDF - probability density function

print(norm.pdf(0, loc=5, scale=10))

r = np.random.randn(10000)

print(norm.pdf(r))

plt.scatter(r, norm.pdf(r))
plt.show()

plt.scatter(r, norm.logpdf(r))  # tańsze w obliczeniach
plt.show()

plt.scatter(r, norm.cdf(r))  # dystrybuanta
plt.show()

plt.scatter(r, norm.logcdf(r))
plt.show()

plt.hist(r, bins=100)
plt.show()


r = np.random.randn(10000, 2)
plt.scatter(r[:, 0], r[:, 1])
plt.axis('equal')
plt.show()

# z variance = 5 i mean = 10 (wariancja i wartość oczekiwana lub mediana)
r[:, 1] = 5*r[:, 1] + 10
plt.scatter(r[:, 0], r[:, 1])
plt.axis('equal')
plt.show()


# kiedy wymiary nie są od siebie niezależne, zdefiniujmy macierz kowariancji
# oznacza, że mamy wariancję 1 w wymiarze pierwszym, 3 w wymiarze drugim i 0.8 kowariancję między nimi
cov = np.array([[1, 0.8], [0.8, 3]])

from scipy.stats import multivariate_normal as mvn

mu = np.array([0, 2])
r = mvn.rvs(mean=mu, cov=cov, size=1000)  # Draw random samples from a multivariate normal distribution

plt.scatter(r[:, 0], r[:, 1])
plt.axis('equal')
plt.show()

# sprawdźmy czy to dobry wynik:
r = np.random.multivariate_normal(mean=mu, cov=cov, size=1000)
plt.scatter(r[:, 0], r[:, 1])
plt.axis('equal')
plt.show()


# teraz trochę przydatnych funkcji w scipy:
# scipy.io.loadmat() - ładuje pliki matlaba .mat
# scipy.wavfile.read() - zwraca częstotliwość próbkowania w plikach wav
# scipy.io.wavfile.read() - odczyt plików wave
# scipy.io.wavfile.write() - zapis plików wave
# scipy.signal.convolve() - konwolucja dwóch wielowymiarowych macierzy
# scipy.signal.convolve2d() - lepsze dla obrazów jako 2d

# transformata Fouriera - z zależności czasowych na częstotliwość
x = np.linspace(0, 100, 10001)
y = np.sin(x) + np.sin(3*x) + np.sin(5*x)  # mamy przebieg wieloczęstotliwościowy, nałożenie kilku częstotliwości

plt.plot(y)
plt.show()

# i robimy Fouriera
Y = np.fft.fft(y)

plt.plot(np.abs(Y))  # trzeba powiększyć lewą cześć wykresu i wtedy widać piki przy 16, 48 i 80
plt.show()

# to teraz możemy policzyć na piechotę z pików jakie były współczynniki częstotliwości:
print(f"Składowa z piku 16: {2*np.pi*16/100}")
print(f"Składowa z piku 48: {2*np.pi*48/100}")
print(f"Składowa z piku 80: {2*np.pi*80/100}")
