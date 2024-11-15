# simclr_optimization
This repository contains the code for feature Extraction Optimization in SimCLR through Model Compression Techniques and Training Enhancements

---

The repository have the following structure:
- `classes`: contains the files with `SimCLR` and its variants; each file has exactly one class 
- `dataset`: contains the code for preparing and downloading `CIFAR-10`; creates the folder `data` inside and store data there
- `linear_evaluation`: contains the class for linear evaluation of `SimCLR` implementation and stores the weight for trained logistic models for each experiment
- `log`: contains the log files for each experiment
- `metrics`: contains the file with class Metrics, that compute and stores metrics; also contain stored metrics for each experiment
- `models`: contains the .zip files (under 100 MB) of weights for each experiment model
- `plots`: contains the files for exploring metrics and stores the plots
- `main.py`: the entry point; this file prepares dataset, train the model, evaluate it and save all the data
---
# How to run the code
Follow this structure of command in your terminal:

```python
python main.py [model_name] [batch_size] [epochs]
```

Provide all the parameters or none of them. If you will not pass the parameters, then they will be set to default values: ```model_name = "baseline", batch_size = 256, epochs = 800```

Otherwise, the next example you may use:

```python
python main.py baseline 256 501
```

### Model names:
- baseline
