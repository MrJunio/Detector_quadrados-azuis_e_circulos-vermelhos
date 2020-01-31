import cv2
import numpy as np

#recebe imagem e mascará com os possíveis candidatos a serem um quadrado
#retorna imagem com os contornos desenhados, mascara dos contornos e contornos
def detectSquares(img, mask):
    maskSquares = np.zeros((mask.shape[0], mask.shape[1]), dtype='uint8') #Mascara preta
    contourns, hierarchi = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = [] #Vetor de polígonos

    for contourn in contourns:
        convex = cv2.convexHull(contourn) #Transforma o contorno em uma superfícia convexa
        if cv2.contourArea(convex) > 500:   #Se a área da superfície for maior que 100
            perimeter = cv2.arcLength(convex, True) #calcula o comprimento do arco
                                                    #parâmetros: contorno, True(indica que a curva é fechada)
                                                    #retorno: comprimento do arco

            poly = cv2.approxPolyDP(convex, 0.04*perimeter, True) #Aproxima a superfície convexa a um polígono
                                                                  #argumentos: contorno, distância máxima do contorno ao contorno aproximado,
                                                                  #True(indica que a curva é fechada)

            x,y,w,h = cv2.boundingRect(poly) #Pegando os quatro cantos de uma retângulo em volta do polígono

            # Se o polygono tiver 4 lados E Se a largura divergir da altura em no máximo 30% para mais ou para menos
            if len(poly) == 4 and 0.85<=float(w/h)<=1.15:
                cv2.drawContours(img, [poly], -1, (0,255,0), 2)    #Desenhando contorno do polígono na imagem
                cv2.drawContours(maskSquares, [poly], -1, 255, -1) #Desenhando o polígono na Mascara
                polys.append(poly)

    return img, maskSquares, polys

#recebe imagem e mascará com os possíveis candidatos a serem um círculo
#retorna imagem com os contornos desenhados, mascara dos contornos e contornos
def detectCircles(img, mask):
    maskCircles = np.zeros((mask.shape[0], mask.shape[1]), dtype='uint8') #Mascara preta
    contourns, hierarchi = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = [] #Vetor de polígonos

    for contourn in contourns:
        convex = cv2.convexHull(contourn) #Transforma o contorno em uma superfícia convexa
        if cv2.contourArea(convex) > 500: #Se a área da superfície for maior que 100
            perimeter = cv2.arcLength(convex, True) #calcula o comprimento do arco
                                                    #parâmetros: contorno, True(indica que a curva é fechada)
                                                    #retorno: comprimento do arco
            poly = cv2.approxPolyDP(convex, 0.01*perimeter, True) #Aproxima a superfície convexa a um polígono
                                                                  #argumentos: contorno, distância máxima do contorno ao contorno aproximado,
                                                                  #True(indica que a curva é fechada)
            if len(poly)>8: # Se o polygono tiver no mínimo 6 lados
                cv2.drawContours(img, [poly], -1, (0,255,0), 2)    #Desenhando contorno do polígono na imagem
                cv2.drawContours(maskCircles, [poly], -1, 255, -1) #Desenhando o polígono na mascara
                polys.append(poly)

    return img, maskCircles, polys



