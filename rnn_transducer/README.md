# Sequence To Sequence Modeling

## Model
* RNN-Transducer: main model code borrowed from https://github.com/HawkAaron/E2E-ASR
* A bit of improvement: 
  1. add the projection between LSTM layers (pytorch 1.8 or later)
  2. refactor tensor op before the joint layer
  3. add dynamic quantization for RNN-T model
  4. add more general seq2seq scaffold beyond ASR 

## Usage

* Train
```
python train.py --lr 4e-4 --bi --dropout 0.5 --out exp/rnnt_bi_lr4e-4_dp0.5 
```

* Evaluate
```
python eval.py [trained_model_path] --bi
```

* Quantize
```
python quantize.py [trained_model_path] --bi --quantized_model [quantized_model_path]
```

