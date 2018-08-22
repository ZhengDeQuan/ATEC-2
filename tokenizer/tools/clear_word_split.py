from os import path
import codecs


def dict_clear(url, char="@"):
    with codecs.open(url, "r", encoding="utf8") as f:
        lines = f.readlines()
        lines_after = filter(lambda x: x.find(char) != -1, lines)
    with codecs.open(url, "w", encoding="utf8") as f:
        f.writelines(lines_after)


package_dir_name = path.dirname(path.realpath(__file__)) + "/../"
sort_file_list = ["dict/4+ch_word_split.txt",
                  "dict/3ch_word_split.txt"]
for sort_file in sort_file_list:
    dict_clear(package_dir_name + sort_file, char="@")
