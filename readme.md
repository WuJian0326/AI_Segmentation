This repository contains code for training and testing AI models for segmentation tasks. It provides functionality to modify different models and train them using various parameters.

## Train

The `train.py` script can be used to train the models. It accepts several parameters, as listed below:
--lr, --learning_rate learning rate (default: 1e-3)
-b, --batch_size batch size (default: 1)
-e, --epoch number of epochs (default: 160)
-worker, --num_worker number of workers (default: 16)
-class, --num_class number of classes (default: 4)
-c, --in_channels number of input channels (default: 1)
-size, --image_size image size (default: 224)
-flow, --train_flow training flow (default: 10)


You can modify these parameters to suit your specific training requirements.

## Test

The `test.py` script is currently under testing and can be used to predict the results of the trained models.

## Dataset

The repository supports various data formats, including h5, png, jpg, and numpy arrays. Currently, the following datasets are available:

- ACDC: Heart MRI dataset
- Microglia: Microglia dataset
- Brain Tumor: Brain tumor dataset

Please note that the datasets may have different file formats and structures. Make sure to preprocess the data accordingly before using it for training or testing.

## Usage

1. Clone the repository:

   ```shell
   git clone https://github.com/your-username/AI_segmentation.git
2. Modify the parameters in train.py to configure the training process.
3. Run the training script:
    ```shell
    python train.py
4. Use test.py to predict the results once the models are trained and ready.
    ```shell
    python test.py --checkpoint=<Path> --out=<outPath>

Feel free to explore and modify the code according to your needs. If you encounter any issues or have any suggestions, please feel free to open an issue or submit a pull request.

Happy segmentation!

This Readme made by Chatgpt