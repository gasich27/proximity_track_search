# Proximity Track Search

Сейчас первый этап ML-проекта по поиску похожей музыки.

Сейчас проект собирает треки из Telegram-канала и извлекает для каждого трека
признаки четырьмя способами: `Librosa`, `OpenL3`, `YAMNet` и `CLAP`.

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Librosa](https://img.shields.io/badge/-Librosa-8A2BE2?style=flat-square)
![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Telegram](https://img.shields.io/badge/-Telegram-26A5E4?style=flat-square&logo=telegram&logoColor=white)

---
---

## Зачем нужен проект

На первом этапе нужно собрать около 300 треков и вручную оценить каждый по трём
шкалам от 0 до 100: `style`, `energy` и `darkness`.

После разметки можно будет сравнить четыре способа представления аудио и выбрать
embedding, который лучше всего отражает сходство между треками.

---
---

## Как работает

```text
Telegram-канал
↓
Загрузка одного трека
↓
Три 10-секундных фрагмента
↓
Librosa / OpenL3 / YAMNet / CLAP
↓
Сохранение embeddings и строки для ручной разметки
↓
Удаление временного аудиофайла
```

Трек попадает в датасет только в том случае, если успешно отработали все четыре
метода. После каждого трека результаты сразу сохраняются, поэтому сбор можно
остановить и продолжить позже.

---
---

## Результат этапа

```text
data/
├── labels.csv
├── librosa_embeddings.csv
├── openl3_embeddings.csv
├── yamnet_embeddings.csv
└── clap_embeddings.csv
```

`labels.csv` предназначен для ручной разметки. Один `track_id` соответствует
одному и тому же треку во всех файлах.

---
---

## Что будет дальше

После ручной аккуратной разметки embeddings будут сравнены между собой и для будущего бота
будет выбран лучший метод из представленных.

Планируемый бот получит трек от пользователя, найдёт через KNN десять наиболее
похожих треков и добавит исходный трек в Telegram-канал. KNN, рекомендации и сам
бот на текущем этапе ещё не реализованы.

---
---

## Запуск

```powershell
pip install -r requirements.txt
Copy-Item .env.example .env
python main.py
```

Перед запуском нужно заполнить Telegram-параметры в `.env`.

---
---
---

![alt text](<Frame 766.png>)