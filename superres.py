"""

# Upscale Examples:

upscale_models = ["R-ESRGAN General WDN 4x V3", "R-ESRGAN 4x+", "R-ESRGAN 4x+ Anime6B", "R-ESRGAN 2x+", "ESRGAN 4x", "Swin2SR 4x"]
from super_resolution.upscale_methods import upscale_one, upscale_frames, upscale_folder
from PIL import Image

1. upscale_one('./input.png', './output.png', "R-ESRGAN General WDN 4x V3", return_image=False)
2. image = Image.open('./input.jpg')
   image = upscale_one(image, model_name="R-ESRGAN 2x+")
   image.save('./output.jpg')
3. upscale_frames('./input.gif', './output.gif', "R-ESRGAN 4x+ Anime6B")
4. upscale_frames('./input.mp4', './output.mp4', "ESRGAN 4x")
5. upscale_folder('./input_folder', './output_folder', "Swin2SR 4x")




# Beautify Examples:

from super_resolution.portraitsr import PortraitSR
from PIL import Image
b = PortraitSR()

1.  image = Image.open('./blurred_portrait.jpeg')
    image = b.beautify_image(image, uncertainty=1.0)
    image.save('./highres_portrait.jpeg')
2.  b.beautify_gif('./input.gif', './output.gif')
3.  b.beautify_video('./input.mov', './output.mov')
4.  b.beautify_folder('./input_folder', './output_folder')

"""

from super_resolution.portraitsr import PortraitSR
from PIL import Image
b = PortraitSR()

image = Image.open('./blurred_face.jpg')
image = b.beautify_image(image, uncertainty=0.0)
image.save('./highres_face.jpg')