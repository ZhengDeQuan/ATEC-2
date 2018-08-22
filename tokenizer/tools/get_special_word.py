import codecs
from os import path

from preprocessor import data_reader
from tokenizer import Tokenizer

# logging
base_dir = path.dirname(path.realpath(__file__)) + "/.."

# read data
data = data_reader.DataReader.read_init_data()


# get special word
def get_special_word(file_url, func_filter):
    word_set = set()
    with codecs.open(file_url, "w", encoding="utf8") as f:
        for i, left, right, label in zip(data["ordinal"], data["left"], data["right"], data["label"]):
            seg_left = Tokenizer.cut(left)
            for word in seg_left:
                if func_filter(word):
                    word_set.add(word)
            seg_right = Tokenizer.cut(right)
            for word in seg_right:
                if func_filter(word):
                    word_set.add(word)
        for word in word_set:
            f.write(word + "\n")


# main
def main():
    get_special_word(base_dir + "/../data/analysis/3ch_word.txt", lambda x: len(x) == 3)
    get_special_word(base_dir + "/../data/analysis/4+ch_word.txt", lambda x: len(x) >= 4)


if __name__ == "__main__":
    main()
