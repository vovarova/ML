from tkinter import *

import PIL
from PIL import Image, ImageDraw

from ToMnist import *
from hw3 import *

width = 200
height = 200
white = (255, 255, 255)
green = (0, 128, 0)
number = 0

w = loadModel()


def save():
    x = toMnist(image).reshape(1, 784)
    guessedNumber.set(str(getPredictedNumber(x, w)[0]))
    print(x)


def paint(event):
    # python_green = "#476042"
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_oval(x1, y1, x2, y2, fill="black", width=8)
    draw.line([x1, y1, x2, y2], fill="black", width=8)


def createImage():
    return PIL.Image.new("L", (width, height), color="white")


def clear():
    cv.delete("all")
    global image, draw
    image = createImage()
    draw = ImageDraw.Draw(image)
    pass


root = Tk()

# Tkinter create a canvas to draw on
cv = Canvas(root, width=width, height=height, bg='white')
cv.pack()

# PIL create an empty image and draw object to draw on
# memory only, not visible
image = createImage()
draw = ImageDraw.Draw(image)

# do the Tkinter canvas drawings (visible)
# cv.create_line([0, center, width, center], fill='green')

cv.pack(expand=YES, fill=BOTH)
cv.bind("<B1-Motion>", paint)

# do the PIL image/draw (in memory) drawings
# draw.line([0, center, width, center], green)

# PIL image can be saved as .png .jpg .gif or .bmp file (among others)
# filename = "my_drawing.png"
# image1.save(filename)
button = Button(text="save", command=save)
clear = Button(text="clear", command=clear)
guessedNumber = StringVar()
Label(root, textvariable=guessedNumber).pack()
button.pack()
clear.pack()
root.mainloop()
