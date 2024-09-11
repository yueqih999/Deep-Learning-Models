# Training and Testing the LeNet5 Model
This project implements the LeNet5 model to classify the MNIST dataset with various regularization techniques: Dropout, Weight Decay (L2 Regularization), and Batch Normalization. The following instructions will guide you on how to train the model under each setting and how to test it using the saved weights.

## Training the Model
To train the original model:
python main.py --technique none

To train the model with Dropout at the hidden layers:
python main.py --technique dropout

To train the model with L2 Regularization:
python main.py --technique l2 --weight_decay 1e-4

To train the model with Batch Normalization:
python main.py --technique batch_norm

## Testing the Model with Saved Weights
python test.py --model_path models/the_name_of_model.pth