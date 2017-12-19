# coding: utf-8

import helper # Вспомогательные функции для сохранения и загрузки модели и параметров
import numpy as np
import random
import time
import tensorflow as tf

data_dir = './data/headers_full.txt'
text = helper.load_data(data_dir)

# Файл модели
load_dir = './models/word_emb'

tokens = {
        ".": "||PERIOD||",
        ",": "||COMMA||",
        '"': "||QUOT_MARK||",
        ";": "||SEMICOL||",
        "!": "||EXCL_MARK||",
        "?": "||QUEST_MARK||",
        "(": "||L_PARENTH||",
        ")": "||R_PARENTH||",
        "--": "||DASH||",
        "\n": "||RETURN||"
    }

for key, token in tokens.items():
    text = text.replace(key, ' {} '.format(token.lower()))

lines = text.split(' ||period||  ')
    
first_words = list(set([line.split(" ")[0] for line in lines]))  

_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
seq_length, _ = helper.load_params()

# Длина генерируемой последовательности
gen_length = 10
phrases = 10

def get_tensors(loaded_graph):

    inputs = loaded_graph.get_tensor_by_name('input:0')
    initial_state = loaded_graph.get_tensor_by_name('initial_state:0')
    final_state = loaded_graph.get_tensor_by_name('final_state:0')
    probs = loaded_graph.get_tensor_by_name('probs:0')
    
    return inputs, initial_state, final_state, probs

def pick_word(probabilities, int_to_vocab):

    return int_to_vocab[np.argmax(probabilities)]

headlines = []
loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:

    # Загружаем модель
    loader = tf.train.import_meta_graph(load_dir +'.meta')
    loader.restore(sess, load_dir)

    # Получаем тензоры из модели
    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})
    
    while len(headlines) < phrases:
        
        print("Генерация загоровка: %d из %d" % (len(headlines)+1, phrases), end="\r")
        
        # Инициализируем переменную, где будем хранить сгенерированную последовательность
        headline = ''
        prime_word = first_words[random.randint(0, len(first_words))]
        gen_sentences = [prime_word]
        pred_word = ''
        
        # Генерация последовательности
        while pred_word.find('||period||') < 0:
            dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
            dyn_seq_length = len(dyn_input[0])

            # Получаем вероятности
            probabilities, prev_state = sess.run(
                [probs, final_state],
                {input_text: dyn_input, initial_state: prev_state})

            # Получаем следующее слово
            pred_word = pick_word(probabilities[0][dyn_seq_length-1], int_to_vocab)
            gen_sentences.append(pred_word)

        # Удаляем токены пунктуации, заменяя их на соответствующие символы
        headline = ' '.join(gen_sentences)
        for key, token in token_dict.items():
            ending = ' ' if key in ['\n', '(', '"'] else ''
            headline = headline.replace(' ' + token.lower(), key)
        headline = headline.replace('\n ', '\n')
        headline = headline.replace('( ', '(')
        headline = headline.replace('. ', '.\n')

        if headline.replace('.', '') not in lines and len(headline.split(" ")) > 2:
            print(" "*100, end="\r")
            print(headline)
            headlines.append(headline)
    
    with open('./headlines.txt', 'w') as file:
        for hl in headlines:
            file.write(hl + '\n')
    print("\nГотово, заголовки сохранены в headlines.txt")
   