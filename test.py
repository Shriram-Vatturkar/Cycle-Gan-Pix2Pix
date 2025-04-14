"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
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
