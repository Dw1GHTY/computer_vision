import cv2 
import numpy as np
import matplotlib.pyplot as plt


input_img = cv2.imread('slika2.jpg')
grayscale_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(grayscale_img, 10, 255, cv2.THRESH_BINARY)

#Kreira se marker koji je kopija threshold-a, cija kompletna unutrasnjost se setuje na 0, tako da samo ivice imaju vrednost 255
marker = thresh.copy()
for i in range(1, marker.shape[0]-1):
    for j in range(1, marker.shape[1]-1):
        marker[i,j] = 0


tmp=np.zeros_like(marker) #Setovanje tmp na neku vrednost razlicitu od mask kako bi se inicijalno uslo u petlju

#Petlja koja se izvrsava sve dok postoji neka razlika izmedju starog i novog markera 
#"Novi" marker u svakoj iteraciji dobija se, najpre, primenom dilacije, nakon cega se nad thresholdom i markerom primenjuje min fja
#Min funkcija za cilj ima da ukloni one regione markera (nad kojim je primenjena dilacija) koji prevazilaze granice thresholda  
while np.any(marker - tmp) != 0:
    tmp=marker.copy()
    marker=cv2.dilate(marker, kernel=np.ones((3,3),np.uint8))
    marker=cv2.min(thresh, marker)
    # plt.imshow(marker, cmap='gray')
    # plt.show()

#Nakon izvrsene petlje, marker sada sadrzi sliku iskljucivo sa elementima koji dodiruju ivicu, koje je potrebno eliminisati
#Marker se invertuje u cilju dobijanja konacne maske koja se and-uje sa originalnom slikom, kako bi ostali samo objekti koji ne dodiruju ivice
# plt.imshow(marker, cmap='gray')
# plt.show()
mask=cv2.bitwise_not(marker)
output_img=cv2.bitwise_and(input_img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))


# plt.imshow(input_img, cmap='gray')
# plt.show()
plt.imshow(output_img, cmap='gray')
plt.show()
