from .base_tokenizer.base_jieba import Jieba
from .base_tokenizer.base_char import CharBased


class Tokenizer:
    __tokenizer = Jieba()
    __charSpliter = CharBased()

    def __init__(self):
        pass

    @staticmethod
    def cut(sentence):
        return Tokenizer.__tokenizer.cut(sentence)

    @staticmethod
    def char_split(sentence):
        return Tokenizer.__charSpliter.cut(sentence)
