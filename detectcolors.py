import cv2

#Entrada: a imagem recebida deve estar no sistema BGR (BlueGreenRed)
#Saida:  a imagem de saida é imagem original com o contorno dos objetos encontrados desenhados,
#        uma mascara tal que as partes que estão dentro do intervalo de cor são brancas e os contornos
#        dos objetos


def detectRed(img):
    img = img.copy() #.copy() é necessário para não alterar
                     # o próprio argumento que foi passado no main
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    minRed1 = (0, 150, 100)
    maxRed1 = (8, 255, 255)
    minRed2 = (170, 150, 100)
    maxRed2 = (179, 255, 255)
    cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    rangeRed1 = cv2.inRange(imgHsv, minRed1, maxRed1)
    rangeRed2 = cv2.inRange(imgHsv, minRed2, maxRed2)
    maskRed = cv2.bitwise_or(rangeRed1, rangeRed2)

    contournsRed, hierarchy = cv2.findContours(maskRed, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contournsRed)>0:
        cv2.drawContours(img, contournsRed, -1, (0,255,0), 2)

    return img, maskRed, contournsRed


def detectBlue(img):
    img = img.copy()
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    minBlue = (90, 120, 135)
    maxBlue = (135, 255, 255)

    maskBlue = cv2.inRange(imgHsv, minBlue, maxBlue)

    contournsBlue, hierarchy = cv2.findContours(maskBlue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contournsBlue) > 0:
        cv2.drawContours(img, contournsBlue, -1, (0, 255, 0), 2)

    return img, maskBlue, contournsBlue

