import subprocess, time

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def setup_environment(setup_environment: bool, print_subprocess: bool):
    if (not setup_environment): return
    print(bcolors.HEADER + "Setting up environment... This may take a few minutes..." + bcolors.ENDC)
    start_time = time.time()
    all_process = [
        [
            *"pip install".split(), 'numpy', 'requests', 'IPython', 'Pillow'
        ],
        [
            *"pip install".split(), 'torch==1.12.1+cu113',
            'torchvision==0.13.1+cu113', '--extra-index-url',
            'https://download.pytorch.org/whl/cu113'
        ],
        [
            *"pip install".split(), 'omegaconf==2.2.3', 'einops==0.4.1',
            'pytorch-lightning==1.7.4', 'torchmetrics==0.9.3',
            'torchtext==0.13.1', 'transformers==4.21.2', 'kornia==0.6.7'
        ],
        "git clone https://github.com/deforum/stable-diffusion".split(),
        [
            *"pip install -e".split(),
            "git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers",
        ],
        [
            *"pip install -e".split(),
            "git+https://github.com/openai/CLIP.git@main#egg=clip",
        ],
        [
            *"pip install".split(),
            *"accelerate ftfy jsonmerge matplotlib resize-right timm torchdiffeq"
            .split(),
        ],
        [
            *"pip install".split(),
            *'opencv-python opencv-contrib-python scikit-image scipy'.split()
        ],
        "git clone https://github.com/shariqfarooq123/AdaBins.git".split(),
        "git clone https://github.com/isl-org/MiDaS.git".split(),
        "git clone https://github.com/MSFTserver/pytorch3d-lite.git".split(),
    ]
    for process in all_process:
        running = subprocess.run(process,
                                 stdout=subprocess.PIPE).stdout.decode('utf-8')
        if print_subprocess:
            print(running)

    print(
        subprocess.run(
            [*'git clone https://github.com/deforum/k-diffusion/'.split()],
            stdout=subprocess.PIPE).stdout.decode('utf-8'))
    with open('k-diffusion/k_diffusion/__init__.py', 'w') as f:
        f.write('')

    end_time = time.time()
    print(f"Environment set up in {end_time-start_time:.0f} seconds")