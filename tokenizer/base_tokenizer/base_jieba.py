# coding=utf-8
import codecs
import logging
import re
from os import path

import jieba


class Jieba:
    # init url
    __base_dir = path.dirname(path.realpath(__file__))
    __keyword_url = __base_dir + "/../dict/keyword.txt"
    __replace_url = __base_dir + "/../dict/replace.txt"
    __word_split_list = ["/../dict/4+ch_word_split.txt",
                         "/../dict/3ch_word_split.txt"]
    __homophone_url = __base_dir + "/../dict/homophone_replace.txt"
    # components
    __replace = None
    __word_spliter = None
    __homophone_transformer = None

    def __init__(self):
        jieba.load_userdict(Jieba.__keyword_url)
        if Jieba.__replace is None:
            Jieba.__init_replace()
        if Jieba.__word_spliter is None:
            Jieba.__init_word_split()
        if Jieba.__homophone_transformer is None:
            Jieba.__init_homophone_transformer()

    @staticmethod
    def __init_replace():
        Jieba.__replace = dict()
        with codecs.open(Jieba.__replace_url, "r", encoding="utf8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                items = line.split("@")
                assert len(items) == 2
                Jieba.__replace[items[0]] = items[1]

    @staticmethod
    def __init_word_split():
        Jieba.__word_spliter = dict()
        for url in Jieba.__word_split_list:
            with codecs.open(Jieba.__base_dir + url, "r", "utf8") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    items = line.split("@")
                    assert len(items) == 2
                    Jieba.__word_spliter[items[0]] = items[1].split(" ")

    @staticmethod
    def __init_homophone_transformer():
        Jieba.__homophone_transformer = dict()
        with codecs.open(Jieba.__homophone_url, "r", encoding="utf8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                items = line.split("@")
                assert len(items) == 2
                Jieba.__homophone_transformer[items[0]] = items[1]

    @staticmethod
    def __replace_before_cut(sentence):
        for old, new in Jieba.__replace.items():
            sentence = sentence.replace(old, new)
        return sentence

    @staticmethod
    def __replace_homophone_before_cut(sentence):
        for old, new in Jieba.__homophone_transformer.items():
            sentence = sentence.replace(old, new)
        return sentence

    @staticmethod
    def __preprocess(sentence):
        obj1 = re.compile(r"\*\*+")
        obj2 = re.compile(r"\.\.+")
        sentence = obj1.sub("num", sentence)
        sentence = obj2.sub("fig", sentence)
        sentence = Jieba.__replace_before_cut(sentence)
        sentence = Jieba.__replace_homophone_before_cut(sentence)
        return sentence

    @staticmethod
    def __std_cut(sentence):
        sentence = Jieba.__preprocess(sentence)
        return jieba.cut(sentence, cut_all=False)

    @staticmethod
    def cut(sentence):
        word_iter = Jieba.__std_cut(sentence)
        for word in word_iter:
            if word == u"\ufeff":
                continue
            if word in Jieba.__word_spliter.keys():
                for new_word in Jieba.__word_spliter[word]:
                    yield new_word
            else:
                yield word


if __name__ == "__main__":
    jieba.setLogLevel(logging.INFO)
    t = Jieba()
    print("/".join(t.cut(u"你需要有不错的个人信用，问一下你的账号密码")))
    print("/".join(t.cut(u"买个花被怎么样，你那个借被咋样了")))
    print("/".join(t.cut(u"***,hello..., **, ..h.h..h*")))
