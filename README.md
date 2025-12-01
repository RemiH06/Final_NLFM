![Made with Python](https://forthebadge.com/images/badges/made-with-python.svg)
![Build with Love](http://ForTheBadge.com/images/badges/built-with-love.svg)

```ascii
███████╗██╗███╗   ██╗ █████╗ ██╗         ███╗   ██╗██╗     ███████╗███╗   ███╗
██╔════╝██║████╗  ██║██╔══██╗██║         ████╗  ██║██║     ██╔════╝████╗ ████║
█████╗  ██║██╔██╗ ██║███████║██║         ██╔██╗ ██║██║     █████╗  ██╔████╔██║
██╔══╝  ██║██║╚██╗██║██╔══██║██║         ██║╚██╗██║██║     ██╔══╝  ██║╚██╔╝██║
██║     ██║██║ ╚████║██║  ██║███████╗    ██║ ╚████║███████╗██║     ██║ ╚═╝ ██║
╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝    ╚═╝  ╚═══╝╚══════╝╚═╝     ╚═╝     ╚═╝
                                                                                
       by Hex (@RemiH06)          version 0.0
```

### General Description

### Project Structure
│
├── data/
│   ├── raw/              # Datos crudos de Banxico
│   └── processed/        # Datos procesados
│
├── src/
│   ├── data_acquisition.py    # Funciones para API Banxico
│   ├── data_processing.py     # Limpieza y transformación
│   ├── feature_engineering.py # Ventanas, escalamiento
│   ├── modeling.py            # Arquitectura y entrenamiento
│   └── visualization.py       # Gráficas y análisis
│
├── notebooks/
│   └── main_analysis.ipynb    # El mero mero notebook
│
├── requirements.txt
└── README.md