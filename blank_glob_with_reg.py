import glob
import sys
from LiLF import lib_img

images = glob.glob(sys.argv[1])
print(f'Blanking images: ', images)
print(f'using region: ', sys.argv[2])
for image in images:
    lib_img.blank_image_reg(image, sys.argv[2], blankval=0.)