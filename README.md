# Plant Sickness Detection

This repository contains a dataset and research on plant sickness detection using machine learning.

The original owner of the dataset and research is [Pratik Kayal](https://github.com/pratikkayal/PlantDoc-Dataset), and this repository serves as a reference to their work.

## Dataset

The dataset consists of images of healthy and sick plants, along with labels indicating the type of sickness.

To use it in the script **train.py** the dataset strutcture must be like this:

```
dataset
├── train
│   ├── healthy
│   │   ├── cherry
│   │   │   ├── healthy_cherry1.jpg
│   │   │   ├── healthy_cherry2.jpg
│   │   │   └── ...
│   │   ├── peach
│   │   │   ├── healthy_peach1.jpg
│   │   │   ├── healthy_peach2.jpg
│   │   │   └── ...
│   │   └── ...
│   └── sick
│       ├── cherry
│       │   ├── sick_cherry1.jpg
│       │   ├── sick_cherry2.jpg
│       │   └── ...
│       ├── peach
│       │   ├── sick_peach1.jpg
│       │   ├── sick_peach2.jpg
│       │   └── ...
│       └── ...
└── test
    ├── healthy
    │   ├── cherry
    │   │   ├── healthy_cherry1.jpg
    │   │   ├── healthy_cherry2.jpg
    │   │   └── ...
    │   ├── peach
    │   │   ├── healthy_peach1.jpg
    │   │   ├── healthy_peach2.jpg
    │   │   └── ...
    │   └── ...
    └── sick
        ├── cherry
            ├── sick_cherry1.jpg
            └── ...
```
## Research

The research involves building a machine learning model to accurately detect plant sickness. 


The code for training the model can be found in **train.py** and to test it on the **val.py**.

**PS:** The path for the *dataset* (for training) is hardcoded and must be changed accordingly.

To train:
```
python3 train.py
```

To test it on a image:
```
python3 val.py -i image.jpg
```

## Usage

To use this repository, simply clone or download it to your local machine. You can then use the dataset to train your own machine learning model for plant sickness detection. Please reference [Pratik Kayal's repository](https://github.com/pratikkayal/PlantDoc-Dataset) if you use their dataset.

## Examples
<div align="center" display="flex">
Healthy plant:

![Example output image](assets/planta_prediction.jpg)

Sick plant:

![Sick plant](assets/sept_prediction.jpg)


</div>
