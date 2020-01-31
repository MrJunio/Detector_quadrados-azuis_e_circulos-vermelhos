import cv2
from detectcolors import *
from detectforms import *

camera = cv2.VideoCapture(0)

while(True):
    sucess, frame = camera.read()
    if not sucess:
        print("A captura falhou.")
        break

    frameBlur = cv2.GaussianBlur(frame, (3,3), 0)
    frameRed, maskRed, contournsRed = detectRed(frameBlur.copy())
    frameBlue, maskBlue, contournsBlue = detectBlue(frameBlur.copy())

    imgBlueSquares, maskBlueSquares, contournsBlueSquares = detectSquares(frame.copy(), maskBlue)
    imgRedCircles, maskRedCircles, contournsRedCircles = detectCircles(frame.copy(), maskRed)

    #O fatiamento ([::2, ::2]) nas imagens abaixo serve para reduzir a quantidade de linhas
    #e colunas ao meio para visualizar melhor os vídeos

    cv2.imshow('Blue Squares', imgBlueSquares[::,::-1])
    #cv2.imshow('frameBlue', frameBlue[::2, ::2][::,::-1])
    #cv2.imshow('maskBlue', maskBlue[::2,::2][::,::-1])
    #cv2.imshow('maskBlueSquares', maskBlueSquares[::,::-1])

    cv2.imshow('Red Circles', imgRedCircles[::,::-1])
    #cv2.imshow('frameRed', frameRed[::2,::2][::,::-1])
    #cv2.imshow('maskRed', maskRed[::2,::2][::,::-1])
    #cv2.imshow('maskRedCircles', maskRedCircles[::,::-1])


    if (cv2.waitKey(1) & 0xFF) == ord('s'):
        break






