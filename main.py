# import needed libraries
import cv2
import numpy as np

# read the input image
img = cv2.imread('alphanumeric.png')

# convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# define the template images for each alphanumeric character
templates = {
    # Upper case letters
    'A': cv2.imread('Letters/A.png', 0),
    'B': cv2.imread('Letters/B.png', 0),
    'C': cv2.imread('Letters/C.png', 0),
    'D': cv2.imread('Letters/D.png', 0),
    'E': cv2.imread('Letters/E.png', 0),
    'F': cv2.imread('Letters/F.png', 0),
    'G': cv2.imread('Letters/G.png', 0),
    'H': cv2.imread('Letters/H.png', 0),
    'I': cv2.imread('Letters/I.png', 0),
    'J': cv2.imread('Letters/J.png', 0),
    'K': cv2.imread('Letters/K.png', 0),
    'L': cv2.imread('Letters/L.png', 0),
    'M': cv2.imread('Letters/M.png', 0),
    'N': cv2.imread('Letters/N.png', 0),
    'O': cv2.imread('Letters/O.png', 0),
    'P': cv2.imread('Letters/P.png', 0),
    'Q': cv2.imread('Letters/Q.png', 0),
    'R': cv2.imread('Letters/R.png', 0),
    'T': cv2.imread('Letters/T.png', 0),
    'U': cv2.imread('Letters/U.png', 0),
    'V': cv2.imread('Letters/V.png', 0),
    'W': cv2.imread('Letters/W.png', 0),
    'X': cv2.imread('Letters/X.png', 0),
    'Y': cv2.imread('Letters/Y.png', 0),
    'Z': cv2.imread('Letters/Z.png', 0),
    # Lower case letters
    'a': cv2.imread('lowercase/a.png', 0),
    'b': cv2.imread('lowercase/b.png', 0),
    'c': cv2.imread('lowercase/c.png', 0),
    'd': cv2.imread('lowercase/d.png', 0),
    'e': cv2.imread('lowercase/e.png', 0),
    'f': cv2.imread('lowercase/f.png', 0),
    'g': cv2.imread('lowercase/g.png', 0),
    'h': cv2.imread('lowercase/h.png', 0),
    'i': cv2.imread('lowercase/i.png', 0),
    'j': cv2.imread('lowercase/j.png', 0),
    'k': cv2.imread('lowercase/k.png', 0),
    'l': cv2.imread('lowercase/l.png', 0),
    'm': cv2.imread('lowercase/m.png', 0),
    'n': cv2.imread('lowercase/n.png', 0),
    'o': cv2.imread('lowercase/o.png', 0),
    'p': cv2.imread('lowercase/p.png', 0),
    'q': cv2.imread('lowercase/q.png', 0),
    'r': cv2.imread('lowercase/r.png', 0),
    's': cv2.imread('lowercase/s.png', 0),
    't': cv2.imread('lowercase/t.png', 0),
    'u': cv2.imread('lowercase/u.png', 0),
    'v': cv2.imread('lowercase/v.png', 0),
    'w': cv2.imread('lowercase/w.png', 0),
    'x': cv2.imread('lowercase/x.png', 0),
    'y': cv2.imread('lowercase/y.png', 0),
    'z': cv2.imread('lowercase/z.png', 0),
    # numbers
    '0': cv2.imread('numbers/0.png', 0),
    '1': cv2.imread('numbers/1.png', 0),
    '2': cv2.imread('numbers/2.png', 0),
    '3': cv2.imread('numbers/3.png', 0),
    '4': cv2.imread('numbers/4.png', 0),
    '5': cv2.imread('numbers/5.png', 0),
    '6': cv2.imread('numbers/6.png', 0),
    '7': cv2.imread('numbers/7.png', 0),
    '8': cv2.imread('numbers/8.png', 0),
    '9': cv2.imread('numbers/9.png', 0),
}

# get the user input string
user_input = input('Enter the characters to detect: ')

# define the threshold value for template matching
threshold = 0.9

# loop over all the characters in the input string and perform template matching
for char in user_input:
    if char in templates:
        template = templates[char]
        # get the width and height of the template
        w, h = template.shape[::-1]
        # perform template matching using normalized cross-correlation
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        # find the locations where the matching template exceeds the threshold
        locations = np.where(result >= threshold)
        # loop over all the locations and draw a rectangle around the matched characters
        for pt in zip(*locations[::-1]):
            cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

# show the output image
cv2.imshow('Output Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()