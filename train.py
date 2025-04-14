import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from tqdm import tqdm
import torch
import numpy as np

if _name_ == '_main_':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create dataset
    dataset_size = len(dataset)    
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      
    model.setup(opt)               
    visualizer = Visualizer(opt)   
    total_iters = 0                

    # Early stopping setup
    best_loss = float('inf')
    patience = opt.patience if hasattr(opt, 'patience') else 10
    patience_counter = 0
    
    # Enable automatic mixed precision
    scaler = torch.cuda.amp.GradScaler() if opt.amp else None

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    
        epoch_start_time = time.time()
        epoch_iter = 0
        epoch_losses = []
        
        # Progress bar
        pbar = tqdm(dataset, desc=f'Epoch {epoch}')
        
        for i, data in enumerate(pbar):
            iter_start_time = time.time()
            
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            
            # Training with AMP support
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    model.set_input(data)
                    model.optimize_parameters(scaler)
            else:
                model.set_input(data)
                model.optimize_parameters()

            # ... rest of the visualization and logging code ...
            
            # Track losses for early stopping
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                current_loss = sum(losses.values())
                epoch_losses.append(current_loss)
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{current_loss:.3f}'})

        # Early stopping check
        avg_epoch_loss = np.mean(epoch_losses)
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
            # Save best model
            model.save_networks('best')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch} epochs')
            break

        print('End of epoch %d / %d \t Time Taken: %d sec' % 
              (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
