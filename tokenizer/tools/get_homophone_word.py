import codecs
from os import path
from time import time

from pypinyin import lazy_pinyin

from preprocessor import DataReader

# get sentences
data = DataReader.read_init_data()
sentences = DataReader.concatenate(data, "left", "right")

# file url
package_dir_name = path.dirname(path.realpath(__file__))
homophone_url = package_dir_name + "/../dict/homophone.txt"
homophone_replace_url = package_dir_name + "/../dict/homophone_replace.txt"


# tool functions
def __to_unicode(_sentence):
    if isinstance(_sentence, str):
        return _sentence.decode("utf8")
    return _sentence


def __to_str(_sentence):
    if isinstance(_sentence, unicode):
        return _sentence.encode("utf8")
    return _sentence


def __word_to_pinyin(_word_unicode):
    return u"".join(lazy_pinyin(_word_unicode))


def __get_all_homophone_words(_sentence, _word):
    char_list = list()
    target_length = len(_word)
    word_list = list()
    for char in _sentence:
        if lazy_pinyin(char)[0] == lazy_pinyin(_word[len(char_list)])[0]:
            char_list.append(char)
        else:
            char_list = list()
        if len(char_list) == target_length:
            homophone_word = u"".join(char_list)
            word_list.append(homophone_word)
            char_list = list()
    return word_list


# find
begin = time()
homophone_word_dict = dict()
homophone_replace_dict = dict()
with codecs.open(homophone_url, "r", encoding="utf8") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        homophone_word_dict[__word_to_pinyin(line)] = line
with codecs.open(homophone_replace_url, "w", encoding="utf8") as f:
    for sentence in sentences:
        sentence = __to_unicode(sentence)
        sentence_pinyin = __word_to_pinyin(sentence)
        for key_pinyin in homophone_word_dict.keys():
            if sentence_pinyin.find(key_pinyin) != -1:
                word_true = homophone_word_dict[key_pinyin]
                homophone_word_list = __get_all_homophone_words(sentence, word_true)
                for word in homophone_word_list:
                    homophone_replace_dict[word] = word_true
    for key, value in homophone_replace_dict.items():
        f.write(key + "@" + value + "\n")
end = time()
print("cost time : {:.2f}".format(end - begin))
