import os
import pickle


def load_data(path):
    """
    Загрузка датасета
    """
    input_file = os.path.join(path)
    with open(input_file, "r", encoding='utf-8') as f:
        data = f.read()

    return data


def preprocess_and_save_data(dataset_path, token_lookup, create_lookup_tables, text):
    """
    Предварительная обработка и сохранение
    """
    # text = open('./data/headers_sample.txt', encoding='utf-8')
    
    # Ignore notice, since we don't use it for analysing the data
    # text = text[81:]

    token_dict = token_lookup()
    for key, token in token_dict.items():
        text = text.replace(key, ' {} '.format(token))

    text = text.lower()
    text = text.split()

    vocab_to_int, int_to_vocab = create_lookup_tables(text)
    int_text = [vocab_to_int[word] for word in text]
    pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('preprocess.p', 'wb'))


def load_preprocess():
    """
    Загрузка предобработанных данных
    """
    return pickle.load(open('preprocess.p', mode='rb'))


def save_params(params):
    """
    Сохранение параметров в файл
    """
    pickle.dump(params, open('params.p', 'wb'))


def load_params():
    """
    Загрузка параметров из файла
    """
    return pickle.load(open('params.p', mode='rb'))
