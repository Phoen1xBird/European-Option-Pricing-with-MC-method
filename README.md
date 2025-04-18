# 📈 European Option Pricing with Monte Carlo Method
Этот проект демонстрирует оценку стоимости европейских опционов с использованием метода Монте-Карло. Реализация представлена в Jupyter Notebook с подробными пояснениями и визуализациями

## 🧠 Описание
Метод Монте-Карло используется для численного приближения цены опциона, моделируя множество возможных траекторий цены базового актив. Для оценки стоимости опциона рассчитывается средняя дисконтированная выплат.

## 📘 Теоретические материлы

Файл `Option_Pricing_theory.pdf` содержит теоретическую основу оценки опционов, включая формулы и объяснения, используемые в реализции.
## 📂 Структура проект


```bash
European-Option-Pricing-with-MC-method/
├── demonstration.ipynb         # Основной ноутбук с реализацией метода
├── Option_Pricing_theory.pdf   # Теоретические материалы по оценке опционов
├── README.md                   # Документация проекта
└── src/                        # Исходные файлы
    ├── black_sholes.py         # Реализация модели Блэка-Шоулза
    └── monte_carlo.py          # Реализация метода Монте-Карло
```


## 🚀 Как запустить

1. **Клонируйте репозиторий:**
   ```bash
   git clone https://github.com/Phoen1xBird/European-Option-Pricing-with-MC-method.git
   cd European-Option-Pricing-with-MC-method
   ``


2. **Установите необходимые зависимости:**
Убедитесь, что установлены следующие библиотеки: `numpy`, `matplotlib`, `scipy`, `jupyter`.*

3. **Запустите Jupyter Notebook:**
   ```bash
   jupyter notebook
   ``

   Откройте файл `demonstration.ipynb` в интерфейсе Jupyter для ознакомления с реализацией.


## 📊 Пример выода

В ноутбуке `demonstration.ipynb` представлены графики и результаты симуляций, иллюстрирующие процесс оценки опционов методом Монте-Карло.
