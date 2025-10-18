# ML Project 4

Проект по анализу рынка подержанных авто: .ipynb файл с запуском утилит имплементированных аналогов sklearn моделей и метрик.

## Структура
- `datasets/data/` — исходные CSV.
- `requirments.txt` — зависимости (Python 3.10+).
- `src/main.ipynb` — основной файл.
- `src/algorithms/` — `s21LogisticRegression`, `s21KNN`, `s21NaiveBayes`.
- `src/utils/` — разбиение данных, генерация признаков, расчёт Gini/ROC-AUC, настройка пайплайна.
- `src/saved/tune_result.pkl` — лучший конфиг.

## Установка зависимостей
Windows (PowerShell):
```powershell
cd path\to\ML_project4
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "${PWD}\src"
jupyter notebook src/main.ipynb
```

macOS / Linux (bash/zsh):
```bash
cd /path/to/ML_project4
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirments.txt
export PYTHONPATH="$(pwd)/src"
jupyter notebook src/main.ipynb
```

Далее ожидается последовательное выполнение ячеек — там загрузка данных, генерация признаков, обучение и подбор гиперпараметров.