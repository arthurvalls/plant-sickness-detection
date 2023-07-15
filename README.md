# Detecção de Doenças em Plantas

Este repositório contém um script que realiza a detecção de doenças em plantas usando aprendizado de máquina.

O proprietário original do conjunto de dados e da pesquisa é [Pratik Kayal](https://github.com/pratikkayal/PlantDoc-Dataset), e este repositório serve como referência para o trabalho dele.

## Conjunto de Dados

O conjunto de dados consiste em imagens de plantas saudáveis e doentes, juntamente com rótulos indicando o tipo de doença.

Para usá-lo no script **train.py** a estrutura do conjunto de dados deve ser assim:
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

## Pesquisa

A pesquisa envolve a construção de um modelo de aprendizado de máquina para detectar com precisão doenças em plantas.

O código para treinar o modelo pode ser encontrado em **train.py** e para testá-lo em **val.py**.

**PS:** O caminho para o *dataset* (para treinamento) está hardcorded e deve ser alterado adequadamente.

Para treinar:
```
python3 train.py
```

Para testá-lo em uma imagem:
```
python3 val.py
```

Ele abrirá uma interface e você poderá fazer upload da imagem.

## Uso

Para usar este repositório, basta cloná-lo ou baixá-lo para sua máquina local.

1. Crie e ative um ambiente virtual:
```
python3 -m venv env
source env/bin/activate
```

2. Instale os pacotes necessários:
```
pip3 install -r requirements.txt
```

3. Se você estiver usando Linux e não tiver o Tkinter instalado para a GUI:
```
sudo apt-get install python3-tk
```

Você pode então usar o conjunto de dados para treinar seu próprio modelo de aprendizado de máquina para detecção de doenças em plantas. Por favor, faça referência ao [repositório de Pratik Kayal](https://github.com/pratikkayal/PlantDoc-Dataset) se você usar o conjunto de dados dele.

## Exemplos

<div align="center" display="flex">
Planta saudável:

![Exemplo de imagem de saída](assets/planta_prediction.jpg)

Planta doente:

![Planta doente](assets/sept_prediction.jpg)

</div>
