# coding=utf-8
from os import path
from pypinyin import lazy_pinyin  # lazy模式是不考虑音调的简单模式，适合排序
import codecs


class DictSorter:
    encoding_list = ["utf8", "gbk"]

    def __init__(self):
        pass

    @staticmethod
    def dict_sort(url, sort_kind="pinyin", encoding="utf8"):
        if encoding not in DictSorter.encoding_list:
            raise Exception("encoding must in " + str(DictSorter.encoding_list))
        if sort_kind == "pinyin":
            with codecs.open(url, "r", encoding=encoding) as f:
                lines = f.readlines()
                lines = map(lambda x: x.strip(), lines)
                text_pinyin_pairs = map(lambda x: (x, lazy_pinyin(x)), lines)
                text_pinyin_pairs = sorted(text_pinyin_pairs, key=lambda x: x[1])
            with codecs.open(url, "w", encoding=encoding) as f:
                lines_sorted = map(lambda x: x[0], text_pinyin_pairs)
                lines_sorted = map(lambda x: x + "\n", lines_sorted)
                f.writelines(lines_sorted)
        else:
            raise Exception("undefined sort kind : " + sort_kind)


if __name__ == "__main__":
    package_root_name = path.dirname(path.realpath(__file__)) + "/../"
    sort_file_list = ["../data/analysis/3ch_word.txt",
                      "../data/analysis/4+ch_word.txt",
                      "/dict/keyword.txt",
                      "/dict/3ch_word_split.txt",
                      "/dict/4+ch_word_split.txt",
                      "/dict/homophone.txt",
                      "/dict/replace.txt",
                      "/dict/homophone_replace.txt",
                      "/dict/keyword_char_based.txt"]
    for sort_file in sort_file_list:
        DictSorter.dict_sort(package_root_name + sort_file)
