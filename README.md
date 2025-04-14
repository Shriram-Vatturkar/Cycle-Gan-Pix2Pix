## How to Run the Project:

1.  **Clone the repository and install requirements:**
    ```bash
    git clone [https://github.com/Shriram-Vatturkar/Cycle-Gan-Pix2Pix](https://github.com/Shriram-Vatturkar/Cycle-Gan-Pix2Pix)
    cd Cycle-Gan-Pix2Pix
    pip install -r requirements.txt
    ```

2.  **Train the Cycle-Gan Model:**
    ```bash
    python train.py --dataroot ./datasets/data --name datasetname --model cycle_gan --batch_size 1 --gpu_ids 0
    ```
    * Replace `--dataroot ./datasets/data` with the actual path to your training data.
    * Replace `--name datasetname` with a name for your training experiment.
    * Adjust `--batch_size` and `--gpu_ids` according to your system configuration.

3.  **Monitor Training:**
    * Start the Visdom server:
        ```bash
        python -m visdom.server
        ```
    * Open the Visdom interface in your web browser at: `http://localhost:8097`

4.  **Test the model:**
    ```bash
    python test.py --dataroot ./datasets/data/testA --name datasetname --model test --no_dropout --num_test 50
    ```
    * Replace `--dataroot ./datasets/data/testA` with the path to your test dataset (e.g., test set for domain A).
    * Ensure `--name` matches the name you used during training.
    * Adjust `--num_test` to the number of test images you want to process.

5.  **Apply the model to your own images:**
    ```bash
    python test.py --dataroot ./datasets/your_images --name datasetname --model test --no_dropout --num_test 1
    ```
    * Replace `--dataroot ./datasets/your_images` with the path to the directory containing your own images.
    * Ensure `--name` matches the name you used during training.
    * Adjust `--num_test` if you have multiple images in your directory.
