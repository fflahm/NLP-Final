# NLP-Final

## Report
See *report.pdf*

## Install Dependencies

```shell
pip install -r requirements.txt
```

## Train model from scratch

```shell
python train.py --train
```

## Train model from a checkpoint

```shell
python train.py --train --model $CHECKPOINT_PATH
```

## Evaluate model

```shell
python train.py --model $CHECKPOINT_PATH
```

## View results
For translation results of our SOTA model, see *inference.ipynb*.
For raw evaluation data, see *results.json*
