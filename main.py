import json
import re
import string
from sys import argv
import os

class Token:
    def __init__(self, Token, serial_num):
        self.Token = Token
        self.punc = Token[0] in string.punctuation
        self.num = Token[0].isdigit()
        self.cap = Token[0].isupper()
        self.len = len(Token)
        self.serial_num = serial_num

class Sentence:
    def __init__(self, text, serial_num):
        self.text = text
        self.tokens = []
        self.nu_of_punc = 0
        self.nu_of_num = 0
        self.serial_num = serial_num

class Passage:
    def __init__(self, name, text):
        self.name = name
        self.text = name + ". " + text
        self.sentences = []
        self.nu_of_tokens = 0
class Corpus:
    def __init__(self, corpus_dir):
        self.passages = []
        self.corpus_dir = corpus_dir
        self.text = ""
        self.special_expressions = ['U.S.A', 'U.S.', 'Dr. ', 'B.Sc.', 'B.A.', 'Ph.D', 'Mr.', 'Ms.', 'Mrs.', 'etc.',
                                    'P.M.',
                                    'A.M.', 'M.D.', 'i.e.', 'mt.', 'no.', 'Sr.', 'vol.', 'vs.']
        for file in os.listdir(corpus_dir):
            with open(os.path.join(corpus_dir, file), 'r') as f:
                str = f.read()
                # add passage name to the beginning of the text
                str = "== " + file[:-4] + " ==" + str
                self.text += str
        self.clean_text()
        self.divide_passages()

    def divide_passages(self):
        divided_text = re.split(r' =====|===== | ====|====| ===|=== |== | ==', self.text)
        for i in range(len(divided_text)):
            self.passages+= Passage(passage.split("==")[1], passage)

    def clean_text(self):
        # remove emojis
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"   u"\U0001F300-\U0001F5FF"    u"\U0001F680-\U0001F6FF"  u"\U0001F1E0-\U0001F1FF"
                                   u"\U00002500-\U00002BEF"          u"\U00002702-\U000027B0"  u"\U00002702-\U000027B0"       u"\U000024C2-\U0001F251"
                                   u"\U0001f926-\U0001f937"       u"\U00010000-\U0010ffff"  u"\u2640-\u2642"  u"\u2600-\u2B55"
                                   u"\u200d" u"\u23cf"      u"\u23e9"    u"\u231a"  u"\ufe0f"  # dingbats
                                   u"\u3030"    "]+", re.UNICODE)

        self.text = re.sub(emoji_pattern, '', self.text)

        # remove http, uml and url
        self.text = re.sub(r'www\S+', '', self.text)
        self.text = re.sub(r'http\S+', '', self.text)
        html_pattern = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});|#([a-z0-9A-Z]+;)')
        self.text = re.sub(html_pattern, '', self.text)

        # change expressions such as Henry D. Ford to Henry D Ford
        callback = lambda pat: pat.group(1)[0]
        self.text = re.sub(r'[A-Z][\.]', callback, self.text)

        #change instances of c. 3566 to c.3566
        callback = lambda pat: pat.group(1)[2].delete()
        self.text = re.sub(r'c\.\s[\d+]', callback, self.text)

        # We'll hide abbreviations for now
        for j in range(len(self.special_expressions)):
                replacement_str = '#' + str(j)
                self.text.replace(self.special_expressions[j], replacement_str)

    # Restore abbreviated expressions
    def restore_abbreviations(self):
        for j in range(len(self.special_expressions)):
                replacement_str = '#' + str(j)
                self.text.replace(replacement_str, self.special_expressions[j])

    # senteces will be divided based on chars contained in {.!?:;}
    def divide_sentences(self):
        for passage in self.passages:
            sentences = re.split(r'(?<![\d+\.+\d+][A-Z][a-z])(?<!\w\.\w.)(?<=\.|\?|\!)\s', passage.text)
            for i in range(len(sentences)):
                passage.sentences += Sentence(sentences[i], i)

    def divide_tokens(self):
        for passage in self.passages:
            for sentence in passage.sentences:
                divided_text = split = re.split("[\s.,!?:;'\"-]+",sentence.text)
                for i in range(len(divided_text)):
                    sentence.tokens += Token(divided_text[i], i)

    def write_json(self):
        with open('corpus.json', 'w') as f:
            json.dump(self, f, default=lambda o: o.__dict__, indent=4)



if __name__ == "__main__":

    corpus_dir = argv[1]    # The directory in which the Wiki files are, full pathname
    output_file = argv[2]   # The text file that the corpus is written onto, full pathname

    Corpus = Corpus(corpus_dir);




