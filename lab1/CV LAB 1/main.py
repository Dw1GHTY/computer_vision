from math import floor
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Učitavanje crno-bele slike
img = cv2.imread('input.png', 0)

plt.imshow(img, cmap='gray')
plt.title('Originalna slika')
plt.show()

# Primena 2D Furijeove transformacije nad slikom
f = np.fft.fft2(img)

# Shift the zero-frequency component to the center of the spectrum
fshift = np.fft.fftshift(f)

# Izračunavanje magnitude(amplitude) spektra
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# Prikaz magnitude
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnituda spektra pre uklanjanja šuma')
plt.savefig('fft_mag.png')
plt.show()


rows, cols = img.shape
crow, ccol = rows//2, cols//2
r = 5  # poluprečnik kruga koji će zaklopiti centar
mask = np.ones((rows, cols), np.uint8)
cv2.circle(mask, (ccol, crow), r, 0, -1)

# Setovanje vrednosti FFT magnitude na 0 unutar kruznog regiona (oko centra kruga)
magnitude_spectrum = magnitude_spectrum * mask

# Pronalaženje lokacija piksela sa maksimalnim vrednostima FFT magnitude, van centra kruga
max_loc = np.where(magnitude_spectrum >= floor(np.max(magnitude_spectrum)))
#print(magnitude_spectrum)
print("MAKS" , np.max(magnitude_spectrum))

for i in range(len(max_loc[0])):
    row = max_loc[0][i]
    col = max_loc[1][i]
    fshift[row, col] = 0
    print("Piksel na lokaciji (",row,",",col,") izazivao šum")


# Izračunavanje magnitude(amplitude) spektra
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# Ponovni prikaz magnitude spektra
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnituda spektra nakon uklanjanja šuma')
plt.savefig('fft_mag_filtered.png')
plt.show()

# Shift the zero-frequency component back to the top-left corner
f_ishift = np.fft.ifftshift(fshift)

# Primena inverzne Furijeove transformacije
img_filtered = np.fft.ifft2(f_ishift).real

# Prikaz konačnog rezultata obrade
plt.imshow(img_filtered, cmap='gray')
plt.title('Konačna slika nakon uklanjanja šuma')
plt.savefig('output.png')
plt.show()