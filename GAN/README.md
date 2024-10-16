# Train DCGAN, WGAN, WGAN-GP
## Architecture 1

```
train.py --model dcgan --architecture 1 --learning_rate 0.001 --num_epochs 20
train.py --model wgan --architecture 1 --learning_rate 0.00005
train.py --model wgan-gp --architecture 1 --learning_rate 0.0005
```

## Architecture 2 (D without BatchNorm)

```
train.py --model dcgan --architecture 2 --learning_rate 0.0005 --num_epochs 20
train.py --model wgan --architecture 2 --learning_rate 0.0001
train.py --model wgan-gp --architecture 2 --learning_rate 0.001
```

# Generate Images from Trained Models

```
generate.py
```