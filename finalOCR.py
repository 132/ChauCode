#!/usr/bin/env python3
import os
import numpy as np
import cv2
################################################
def cutImg2(img):
    # print(img)
    # img_gray = cv2.imread(img,0)
    # img = cv2.imread(img,flags=cv2.IMREAD_COLOR)

    # 3 matrices for the image
    # _blue = img[:,:,0]
    # _green = img[:,:,1]
    # _red = img[:,:,2]
    #
    # # take 3 matrices distance for those color [0,1]
    # distanceMatrixBlue = statisticIM(_blue)
    # distanceMatrixGreen = statisticIM(_green)
    # distanceMatrixRed = statisticIM(_red)

    # distance each pixel
    ret, bw_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

    H, W = bw_img.shape
    print(bw_img)
    #tempImg = img_gray
    # sum each columns
    sumCol = [sum(bw_img.transpose()[i]) for i in range(W)]
    sumRow = [sum(bw_img[i]) for i in range(H)]
    # print(sumCol)
    # print(sumRow)

    # scale W the range startpoint and endpoint in
    for i in range(W):
        if sumCol[i] > H:
            startW = i
            break
    for i in range(W-1,-1,-1):
        if sumCol[i] > H:
            endW = i
            break
    for i in range(H):
        if sumRow[i] > W:
            startH = i
            break
    for i in range(H-1,-1,-1):
        if sumRow[i] > W:
            endH = i
            break

    # print (startH, endH)
    # print (startW, endW)

    # capture the position for each charater
    # H_start = [startH]
    # H_end= []
    W_start= []
    W_end= []

    # print(sumCol[startW:endW])

    count = 0
    tempFlag = [0] * len(sumCol)

    tempW = sumCol.copy()

    tempFlag_Check = [0] * len(sumCol)
    tempFlag_Minimum = [0] * len(sumCol)
    for i in range(len(sumCol)):
        # if sumCol[i] > 34:
        #     for j in range(H):
        #         if tempImg[j][i] == 127:
        #             tempFlag_Check[i] = 1
        if sumCol[i] < (255*H):
            tempFlag_Minimum[i] = 1

    #print('check1: ', tempFlag_Check)
    #print('check2: ', tempFlag_Minimum)

    for i in range(len(tempFlag_Minimum)):
        if i == 0:
            continue
        elif i==len(tempFlag_Minimum) - 1:
            break
        else:
            if tempFlag_Minimum[i] == 1 and tempFlag_Minimum[i-1] == 0:
                W_start.append(i)
            if tempFlag_Minimum[i] == 1 and tempFlag_Minimum[i+1] == 0:
                W_end.append(i)

    # for i in range(1,10):
    #     #print(tempW[startW:endW])
    #     min_ = startW
    #     value_min = tempW[min_]
    #     for w in range(startW + 1, endW):
    #         if tempFlag[w] == 1:
    #             continue
    #         elif value_min > tempW[w] and tempFlag_Check[w] == 0:
    #         #elif value_min > tempW[w]:
    #             min_ = w
    #             value_min = tempW[w]
    #
    #     # print ("minimum: ", min_, value_min)
    #     tempFlag[min_] = 1
    #     tempW[min_] += 100
    #     W_start.append(min_)

    #W_start.append(endW)
    W_start.sort()
    W_end.sort()
    print(W_start)
    print(W_end)

    if len(W_start) < 6 or len(W_end) < 6:
        lack_region = 6 - len(W_start)
        print('lack regions: ', lack_region)
        W_start,W_end = speccialCase(W_start,W_end, sumCol, lack_region)

    print (W_start,W_end)

    tempImg_ = []
    #test_ = []
    oriSetImg = []
    for i in range(len(W_start)):

        tempImg_.append(bw_img[startH:endH, W_start[i]:W_end[i]])
        # matrix gray scale
        #test_.append(img_gray[startH:endH,  W_start[i]:W_end[i]])
        #ori img
        #oriSetImg.append(img[startH:endH, W_start[i]:W_start[i+1]-1])
        oriSetImg.append(img[startH:endH,  W_start[i]:W_end[i]])
    #
    # print("----------------------------------------")
    # # print(len(tempImg_))
    #print(tempImg_)

    # normalize image to be come 0 or 1
    for k in range(len(tempImg_)):
        tempImg_[k] = tempImg_[k]/ 255

    return [tempImg_, oriSetImg]

def speccialCase(W_start, W_end, sumCol, lack_region):
    if lack_region == 0:
        return [W_start, W_end]
    else:
        space_region = []
        for i in range(len(W_start)):
            space_region.append(W_end[i] - W_start[i])
        max_Index_space_region = []
        max_valueSpace= 0
        max_indexSpace = 0
        for i in range(len(W_start)):
            if space_region[i] > max_valueSpace and i not in max_Index_space_region:
                max_valueSpace = space_region[i]
                max_indexSpace = i
        max_Index_space_region.append(max_indexSpace)
        print(space_region)
        print (max_Index_space_region)
        lack_region = lack_region - 1

        max_value = sumCol[W_start[max_indexSpace]]
        max_index = max_indexSpace
        for i in range(W_start[max_indexSpace],W_end[max_indexSpace]):
            if sumCol[i] > max_value:
                max_value = sumCol[i]
                max_index = i
        print(max_index, max_value)
        W_start.append(max_index)
        W_end.append(max_index)
        W_start.sort()
        W_end.sort()

        return speccialCase(W_start, W_end, sumCol, lack_region)

# B2 Cat chu thanh Box
def cutImg(img):
    # convert to gray-scale image
    ret, bw_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

    # take the height and width
    h, w = bw_img.shape
    flag = []

    # start from 0
    for i in range(w):
        if sum(bw_img.transpose()[i]) < (255*h - 255):      # take a list of minimum columns
            flag.append(i)

    # start and stop lists are to separate characters. If there are 6 characters = 6 elements on each list
    startF = []
    stopF = []
    # tempImg and oriImg are about images for characters - grayscale and normal
    tempImg = []
    oriImg = []

    # way to separate characters
    startF.append(flag[0])
    for j in range(len(flag)):
        if j == len(flag) - 1:
            stopF.append(flag[j] + 1)
            tempImg.append(bw_img[:, startF[-1]:stopF[-1]])
            oriImg.append(img[:, startF[-1]:stopF[-1]])
            break
        if flag[j+1] > flag[j]+1:
            stopF.append(flag[j] + 1)
            tempImg.append(bw_img[:, startF[-1]:stopF[-1]])
            oriImg.append(img[:, startF[-1]:stopF[-1]])
            startF.append(flag[j+1])

    # normalize image to be come 0 or 1
    for k in range(len(tempImg)):
        tempImg[k] = tempImg[k]/ 255

    return [tempImg, oriImg]


# B3 so sanh chu
# list_characters is the list images (already separated) which need to be recognized
# avg_list is the roundTruth for the comparison
def compareCharacter(list_characters, avg_list):
    # dictionary for characters
    dict = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # an empty string to return at the end
    text = ''

    # for each character image is compared to the round truth to select the min difference
    for matrix_char in range(len(list_characters)):
        # list of difference for a character image with all character in round truth
        list_diff = []
        h1, w1 = list_characters[matrix_char].shape

        # scan the whole round truth
        for matrix_stand in range(len(avg_list)):
            h2, w2 = avg_list[matrix_stand].shape
            # compare the width and height before computing the absolute value between 2 imgs
            if h1 == h2 and w1 == w2:
                # compute the different for 2 charater images
                diff = np.abs(list_characters[matrix_char] - avg_list[matrix_stand])
                # store the value into the list_diff
                list_diff.append(diff.mean())
            else:
                list_diff.append(1000)

        # after getting all different between the charater image with all characters from round truth
        # selecting the minimum index
        inIndex = list_diff.index(min(list_diff))

        # take the character from the dictionary
        text += dict[inIndex]

    # return the final result
    return text

## 36 matrices characters
def roundTruth(folderName):
    dict = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    A = []
    B = []
    C = []
    D = []
    E = []
    F = []
    G= []
    H= []
    I= []
    #w__, h__ = 8, 5;
    #Matrix = [[0 for x in range(w__)] for y in range(h__)]
    J= []
    K= []
    L= []
    M= []
    N= []
    O= []
    P= []
    Q= []
    R= []
    S= []
    T= []
    U= []
    V= []
    W= []
    X= []
    Y= []
    Z= []
    a_0 = []
    a_1 = []
    a_2 = []
    a_3 = []
    a_4 = []
    a_5 = []
    a_6 = []
    a_7 = []
    a_8 = []
    a_9 = []

    for root, dirs, files in os.walk(folderName):
        for filename in files:
            # cut img_name
            img = cv2.imread(folderName+'/'+filename,0)
            [tempImg, oriImg] = cutImg(img)

            tempList = list(filename[:-4])

            for index_char in range(len(tempList)):
                if len(tempImg) == 6:
                    if tempList[index_char] == 'A':
                        A.append(tempImg[index_char])
                    elif tempList[index_char] == 'B':
                        B.append(tempImg[index_char])
                    elif tempList[index_char] == 'C':
                        C.append(tempImg[index_char])
                    elif tempList[index_char] == 'D':
                        D.append(tempImg[index_char])
                    elif tempList[index_char] == 'E':
                        E.append(tempImg[index_char])
                    elif tempList[index_char] == 'F':
                        F.append(tempImg[index_char])
                    elif tempList[index_char] == 'G':
                        G.append(tempImg[index_char])
                    elif tempList[index_char] == 'H':
                        H.append(tempImg[index_char])
                    elif tempList[index_char] == 'I':
                        I.append(tempImg[index_char])
                    elif tempList[index_char] == 'J':
                        J.append(tempImg[index_char])
                    elif tempList[index_char] == 'K':
                        K.append(tempImg[index_char])
                    elif tempList[index_char] == 'L':
                        L.append(tempImg[index_char])
                    elif tempList[index_char] == 'M':
                        M.append(tempImg[index_char])
                    elif tempList[index_char] == 'N':
                        N.append(tempImg[index_char])
                    elif tempList[index_char] == 'O':
                        O.append(tempImg[index_char])
                    elif tempList[index_char] == 'P':
                        P.append(tempImg[index_char])
                    elif tempList[index_char] == 'Q':
                        Q.append(tempImg[index_char])
                    elif tempList[index_char] == 'R':
                        R.append(tempImg[index_char])
                    elif tempList[index_char] == 'S':
                        S.append(tempImg[index_char])
                    elif tempList[index_char] == 'T':
                        T.append(tempImg[index_char])
                    elif tempList[index_char] == 'U':
                        U.append(tempImg[index_char])
                    elif tempList[index_char] == 'V':
                        V.append(tempImg[index_char])
                    elif tempList[index_char] == 'W':
                        W.append(tempImg[index_char])
                    elif tempList[index_char] == 'X':
                        X.append(tempImg[index_char])
                    elif tempList[index_char] == 'Y':
                        Y.append(tempImg[index_char])
                    elif tempList[index_char] == 'Z':
                        Z.append(tempImg[index_char])
                    elif tempList[index_char] == '0':
                        a_0.append(tempImg[index_char])
                    elif tempList[index_char] == '1':
                        a_1.append(tempImg[index_char])
                    elif tempList[index_char] == '2':
                        a_2.append(tempImg[index_char])
                    elif tempList[index_char] == '3':
                        a_3.append(tempImg[index_char])
                    elif tempList[index_char] == '4':
                        a_4.append(tempImg[index_char])
                    elif tempList[index_char] == '5':
                        a_5.append(tempImg[index_char])
                    elif tempList[index_char] == '6':
                        a_6.append(tempImg[index_char])
                    elif tempList[index_char] == '7':
                        a_7.append(tempImg[index_char])
                    elif tempList[index_char] == '8':
                        a_8.append(tempImg[index_char])
                    elif tempList[index_char] == '9':
                        a_9.append(tempImg[index_char])
                else:
                    print("************************** lengh is not 6 **************************************************")
                    print(filename)

    #return [A, B]
    finalList = [A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
    a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9]
    avg_list = []

    # adding a matrix to J since it does not have this character in the training set
    if len(J) == 0:
        J.append(np.matrix('1 2; 3 4'))

    for indexList in range(0,len(finalList)):
        if len(finalList[indexList]) > 0:

            avg_sublist = finalList[indexList][0]
            for indexSublit in range(1,len(finalList[indexList])):
                h1, w1 = finalList[indexList][indexSublit].shape
                h2, w2 = avg_sublist.shape

                if h1 == h2 and w1 == w2:
                    avg_sublist = np.abs(avg_sublist + finalList[indexList][indexSublit])
                else:
                    print (finalList[indexList][indexSublit])
                    print("******************** the lengh of matrix is not fit ********************************************************")

            avg_sublist = avg_sublist / len(finalList[indexList])
            avg_list.append(avg_sublist)
    return avg_list

# B4 predict the new images based on a roundTruth
def predict_text(fileName, avg_list):
    img = cv2.imread(fileName,0)
    [tempImg, oriImg] = cutImg (img)
    #print (tempImg)
    text = compareCharacter(tempImg, avg_list)
    return text

#folderName = 'CAPCHA'
def runFolder(folderName,avg_list):
    countTrue = 0
    countFalse = 0

    for root, dirs, files in os.walk(folderName):
        for filename in files:
            # take the name from the file image
            fileNameCom = filename[:-4]
            # prdict the image
            text_ = predict_text(folderName+'/'+filename,avg_list)
            # compare between the result and the filename
            if text_==fileNameCom:
                countTrue+=1
            else:
                countFalse+=1
                print(text_)
                print (fileNameCom)
    print(countTrue)
    print(countFalse)
