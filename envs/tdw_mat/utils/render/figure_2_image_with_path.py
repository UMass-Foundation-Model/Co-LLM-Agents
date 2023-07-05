import numpy as np
from PIL import Image, ImageDraw

image = Image.open("figure_2/a/img_0000.jpg")
print(image.size)
screen_positions = np.load("figure_2/screen_positions.npy")
print(screen_positions.shape, np.max(screen_positions), np.min(screen_positions))
color_0 = (255, 217, 102)
color_1 = (215, 168, 246)
colors = [color_0, color_1]
line_width = 8
draw = ImageDraw.Draw(image)
frame = 2000
for j, color in enumerate(colors):
    for i in range(1, frame):
        line = [(screen_positions[j, i - 1][0], 1080- screen_positions[j, i - 1][1]), (screen_positions[j, i][0], 1080 -screen_positions[j, i][1])]
        draw.line(line, fill=color, width=line_width)
image.save("figure_2.png")