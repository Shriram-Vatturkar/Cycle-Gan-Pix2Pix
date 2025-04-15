import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch
from torch.nn.parallel import DataParallel
from pytorch_fid import fid_score
from tqdm import tqdm  # for progress bar

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options

    # ========== Fixed Testing Setup ==========
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    # ========== Multi-GPU Setup ==========
    if torch.cuda.device_count() > 1:
        opt.batch_size = torch.cuda.device_count() * 4
        opt.serial_batches = False

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)

    if torch.cuda.device_count() > 1:
        model.netG = DataParallel(model.netG)

    # initialize wandb
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # ========== Setup HTML Webpage ==========
    web_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.epoch}')
    if opt.load_iter > 0:
        web_dir = f'{web_dir}_iter{opt.load_iter}'
    print('Creating web directory:', web_dir)
    webpage = html.HTML(web_dir, f'Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.epoch}')

    # ========== Setup Directories for FID ==========
    real_images_path = os.path.join(web_dir, 'real_images')
    fake_images_path = os.path.join(web_dir, 'fake_images')
    os.makedirs(real_images_path, exist_ok=True)
    os.makedirs(fake_images_path, exist_ok=True)

    # ========== Set model to evaluation mode if required ==========
    if opt.eval:
        model.eval()

    for i, data in enumerate(tqdm(dataset, desc="Processing images")):
        if i >= opt.num_test:
            break

        model.set_input(data)
        with torch.no_grad():
            model.test()

        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()

        if i % 5 == 0:
            print(f'Processing ({i:04d})-th image: {img_path}')

        save_images(
            webpage, visuals, img_path,
            aspect_ratio=opt.aspect_ratio,
            width=opt.display_winsize,
            use_wandb=opt.use_wandb,
            real_path=real_images_path,
            fake_path=fake_images_path
        )

    webpage.save()  # Save HTML page

    # ========== FID Calculation ==========
    fid_value = fid_score.calculate_fid_given_paths(
        [real_images_path, fake_images_path],
        batch_size=opt.batch_size,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        dims=2048
    )
    print(f'\nFID Score: {fid_value:.2f}')
