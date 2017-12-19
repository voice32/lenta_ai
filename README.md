# Нейронные сети, генерирующая фейковые заголовки новостей методами Char-RNN и Word Embeddings

## Содержание
* `preprocessing.ipynb`: ноутбук с кодом предварительной обработки данных и составления корпуса для обучения
* `char_rnn.ipynb`: код нейронной сети на основе подхода Char-RNN
* `word_embeddings.ipynb`: нейронная сеть, реализующая Word Embeddings
* `./data`: предобработанные файлы с данными
* `./models`: предобученная модель

## Зависимости
* python 3.5
* tensorflow 1.4
* keras 2.0
* numpy
* ijson
* pymorphy2

## Установка
Клонируете репозиторий, устанавливаете зависимости.

## Предварительная обработка текста
Понадобится [оригинальный датасет](https://drive.google.com/open?id=1NlFuOjOt0oQ9Mx70Z7ZvfOsB3-1fCALp), подготовленный [Ильдар Габдрахманов ildarchegg](https://habrahabr.ru/post/343838/), который нужно распаковать в `./data/` и переименовать файл `lenta.json` в `lenta_full.json`.

## Обучение
Запускаете `char_rnn.ipynb` или `word_embeddings.ipynb`.

## Запуск предобученной модели
Необходимо распаковать содержимое `./models/model.zip` в `./models/`, затем  запускаем из корня `python lenta_ai.py`. Параметры генерации можно менять внутри скрипта.

## Результаты
Примеры генерируемых заголовков можно посмотреть здесь: [https://lenta-ai.herokuapp.com/](https://lenta-ai.herokuapp.com/)
