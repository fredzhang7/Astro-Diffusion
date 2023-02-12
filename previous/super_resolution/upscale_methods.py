from PIL import Image

models = {
    'r-esrgan general 4x v3': None,
    'r-esrgan general wdn 4x v3': None,
    'r-esrgan 4x+': None,
    'r-esrgan 4x+ anime6b': None,
    'r-esrgan 2x+': None,
    'esrgan 4x': None,
    'swin2sr 4x': None
}

def upscale_one(input, out_path='', model_name='', return_image=True):
    """Upscale image using the given model_name.\n\n
    Args:
        input (str or PIL.Image): Path to input image or PIL.Image object.
        out_path (str): Path to output image.
        model_name (str): One of "R-ESRGAN General 4x V3", "R-ESRGAN General WDN 4x V3", "R-ESRGAN 4x+", "R-ESRGAN 4x+ Anime6B", "R-ESRGAN 2x+", "ESRGAN 4x", "Swin2SR 4x"
    """
    model_name = model_name.lower()
    if 'r-esrgan' in model_name and models[model_name] is None:
        from super_resolution.automatic111 import UpscalerRealESRGAN
        models[model_name] = UpscalerRealESRGAN('./')
    elif 'esrgan' in model_name and models[model_name] is None:
        from super_resolution.automatic111 import UpscalerESRGAN
        models[model_name] = UpscalerESRGAN('./')
    elif 'swin2sr' in model_name and models[model_name] is None:
        from super_resolution.automatic111 import UpscalerSwin2SR
        models[model_name] = UpscalerSwin2SR('./')

    if not isinstance(input, Image.Image):
        input = Image.open(input)
    
    try:
        image = models[model_name].do_upscale(input, model_name)
    except:
        raise ValueError(f'Invalid model name: {model_name}' + '\nValid model names: ' + ', '.join(models.keys()))

    if return_image:
        return image
    image.save(out_path)


def upscale_frames(in_path, out_path, model_name):
    """
    Upscale gif or video using the given model_name.\n\n
    Args:
        in_path (str): Path to input gif or video.
        out_path (str): Path to output gif or video.
        model_name (str): One of "R-ESRGAN General 4x V3", "R-ESRGAN General WDN 4x V3", "R-ESRGAN 4x+", "R-ESRGAN 4x+ Anime6B", "R-ESRGAN 2x+", "ESRGAN 4x", "Swin2SR 4x"
    """
    if 'r-esrgan' in model_name.lower():
        from super_resolution.automatic111 import UpscalerRealESRGAN
        method = UpscalerRealESRGAN('./')
    elif 'esrgan' in model_name.lower():
        from super_resolution.automatic111 import UpscalerESRGAN
        method = UpscalerESRGAN('./')
    elif 'swin2sr' in model_name.lower():
        from super_resolution.automatic111 import UpscalerSwin2SR
        method = UpscalerSwin2SR('./')
    else:
        raise ValueError('Invalid model name.')

    import imageio, numpy as np

    try:
        reader = imageio.get_reader(in_path)
    except:
        import subprocess
        subprocess.call(['pip', 'install', 'imageio[ffmpeg]'])
        return upscale_frames(in_path, out_path, model_name)
        
    try:
        writer = imageio.get_writer(out_path, fps=reader.get_meta_data()['fps'])
    except KeyError:
        # estimate the fps
        import math, time
        start = time.time()
        for i, frame in enumerate(reader):
            pass
        end = time.time()

        import psutil
        cpu_speed = psutil.cpu_freq().max
        num_cores = psutil.cpu_count()
        num_python_processes = len([p for p in psutil.process_iter() if p.name() == 'python.exe'])
        time_to_iterate =  (end - start) * (cpu_speed / (num_cores * (num_python_processes / 4)))
        fps = math.ceil(i / time_to_iterate)
        writer = imageio.get_writer(out_path, fps=fps)

    for frame in reader:
        if len(frame.shape) == 2:
            frame = np.stack((frame,)*3, axis=-1)
        elif frame.shape[2] == 4:
            frame = frame[:, :, :3]
        frame = Image.fromarray(frame)
        frame = method.do_upscale(frame, model_name)
        writer.append_data(np.array(frame))

    writer.close()
    reader.close()

def upscale_folder(in_folder, out_folder, model_name):
    """
    Upscale all images, gifs, and videos in the given folder with the given model.\n\n
    Args:
        in_folder (str): Path to input folder.
        out_folder (str): Path to output folder.
        model_name (str): One of "R-ESRGAN General 4x V3", "R-ESRGAN General WDN 4x V3", "R-ESRGAN 4x+", "R-ESRGAN 4x+ Anime6B", "R-ESRGAN 2x+", "ESRGAN 4x", "Swin2SR 4x"
    """
    import os
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    for file in os.listdir(in_folder):
        name, ext = os.path.splitext(file)
        if ext in ['.jpg', '.jpeg', '.png']:
            upscale_one(os.path.join(in_folder, file), os.path.join(out_folder, file), model_name, return_image=False)
        elif ext in ['.gif', '.mp4', '.avi', '.mov']:
            upscale_frames(os.path.join(in_folder, file), os.path.join(out_folder, file), model_name)
        else:
            print(f'Invalid file type: {file}')