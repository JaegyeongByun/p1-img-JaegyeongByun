# P Stage 2 - Image Classification

## Getting Started    
### Dependencies
- torch==1.6.0
- torchvision==0.7.0                                                              

### Install Requirements
- `pip install -r requirements.txt`

## Codes
### Train
- `$ python train.py`
- `train()` for single model
- `train_multi()` for multi head model

### Evaluate
- `$ python evaluation.py`
- `evaluation()` for three independent model(age / gender / mask)
- `evaluation_age_gender()` for two independent model(age_gender / mask)
- `evaluation_all()` for single model
- `evaluation_multi()` for multi head model
- `evaluation_tta()` for single mode and using TTA
- `evaluation_tta_with_multi()` for multi head model and using TTA
- `evaluation_ensemble()` for ensemble two single model
- `evaluation_ensemble_tta()` for ensemble two single model using TTA
- `evaluation_ensemble_multi_tta()` for ensemble one single model and two multi head model(one single model and one multi head model use TTA and other multi head model don't use TTA)

### Inference
- `python inference.py`
- `inference()` for three independent model(age / gender / mask)
- `inference_all()` for single model
- `inference_tta()` for single model and using TTA
- `inference_multi()` for multi head model
- `inference_tta_with_multi()` for multi head model using TTA
- `inference_ensemble()` for ensemble two single model
- `inference ensemble_tta()` for ensemble two single model and using TTA(two single models are using TTA)
- `inference ensemble_with_multi_and_tta()` for ensemble one single model and two multi head model(one single model and one multi head model use TTA and other multi head model don't use TTA)

### Model
- `AgeGenderModel()` : only predicting age and gender
- `CustomResNext()` : multi head model
