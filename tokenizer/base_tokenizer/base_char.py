# coding=utf-8
import codecs
from os import path


class CharBased:
    # init url
    __base_dir = path.dirname(path.realpath(__file__))
    __keyword_url = __base_dir + "/../dict/keyword_char_based.txt"
    __replace_urls = [__base_dir + "/../dict/replace.txt",
                      __base_dir + "/../dict/homophone_replace.txt"]
    # components
    __keyword_dict = None
    __num_to_keyword = None
    __replace_dict = None

    # global var
    __count = 0

    def __init__(self):
        if CharBased.__keyword_dict is None:
            CharBased.__init_keyword()
        if CharBased.__replace_dict is None:
            CharBased.__init_replace()
        pass

    @staticmethod
    def __init_keyword():
        CharBased.__keyword_dict = dict()
        CharBased.__num_to_keyword = dict()
        with codecs.open(CharBased.__keyword_url, "r", encoding="utf8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                CharBased.__keyword_dict[line] = "&" + str(CharBased.__count)
                CharBased.__num_to_keyword["&" + str(CharBased.__count)] = line
                CharBased.__count += 1

    @staticmethod
    def __init_replace():
        CharBased.__replace_dict = dict()
        for url in CharBased.__replace_urls:
            with codecs.open(url, "r", encoding="utf8") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    items = line.split("@")
                    CharBased.__replace_dict[items[0]] = items[1]

    @staticmethod
    def __preprocess(sentence):
        for key, value in CharBased.__replace_dict.items():
            sentence = sentence.replace(key, value)
        for key, value in CharBased.__keyword_dict.items():
            sentence = sentence.replace(key, value)
        return sentence

    @staticmethod
    def cut(sentence):
        sentence = CharBased.__preprocess(sentence)
        char_list = list()
        for char in sentence:
            if ('a' <= char <= 'z') \
                    or ('A' <= char <= 'Z') \
                    or ('0' <= char <= '9') \
                    or (char == '*') \
                    or (char == '.') \
                    or (char == '&'):
                char_list.append(char)
            else:
                if char_list:
                    char_sequence = "".join(char_list)
                    yield CharBased.__num_to_keyword.get(char_sequence, char_sequence)
                    char_list = list()
                yield char
        if char_list:
            char_sequence = "".join(char_list)
            yield CharBased.__num_to_keyword.get(char_sequence, char_sequence)


if __name__ == "__main__":
    c = CharBased()
    print("/".join(c.cut(u"你好啊，2340，呵呵，get,借呗和之付宝，怎么样,天哪，余额宝")))
