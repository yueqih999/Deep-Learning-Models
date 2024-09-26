# Training and Testing the LSTM and GRU Model

## LSTM without dropout

```python LSTM/train.py --batch_size 20 --num_epochs 20 --learning_rate 0.1 --dropout 0.0 --num_layers 2 --decay_epochs 8 --decay_co 1.15 ```

## LSTM with dropout

```python LSTM/train.py --batch_size 20 --num_epochs 25 --learning_rate 0.3 --dropout 0.4 --num_layers 2 --decay_epochs 10 --decay_co 1.15```

## GRU without dropout

```python GRU/train.py --dropout 0.0```

## GRU with dropout

```python GRU/train.py```

## Test with model under model folder
