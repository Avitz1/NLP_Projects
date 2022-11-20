import json
import re
import string
import sys
from sys import argv
import os

class Token:
    def __init__(self, token, serial_num):
        self.token = token
        self.punc = len(token) > 0 and token[0] in string.punctuation
        self.num = len(token) > 0 and token[0].isdigit()
        self.cap = len(token) > 0 and token[0].isupper()
        self.len = len(token)
        self.serial_num = serial_num

class Sentence:
    def __init__(self, text, serial_num):
        self.text = text
        self.tokens = []
        self.num_of_punc = 0
        self.num_of_num = 0
        self.serial_num = serial_num
        self.special_expressions = ['transl.', 'lit.', 'U.S.A.', 'U.S.', 'Dr. ', 'B.Sc.', 'B.A.', 'Ph.D', 'Mr.', 'Ms.', 'Mrs.', 'etc.',
                                    'P.M.',
                                    'A.M.', 'M.D.', 'i.e.', 'mt.', 'no.', 'Sr.', 'vol.', 'vs.']
        self.restore_abbreviations()
        self.divide_tokens()
        for token in self.tokens:
            if token.punc:
                self.num_of_punc += 1
            if token.num:
                self.num_of_num += 1
        # Restore abbreviated expressions
    def restore_abbreviations(self):
        for j in range(len(self.special_expressions)):
            replacement_str = '#' + str(j)
            self.text = self.text.replace(replacement_str, self.special_expressions[j])

    # senteces will be divided based on any special characters
    def divide_tokens(self):
        divided_text = re.split("([\s\W])", self.text)
        for i in range(len(divided_text)):
            if divided_text[i] != '' and divided_text[i] != ' ':
                self.tokens += [Token(divided_text[i].strip(), i)]

    def print_sentence(self):
        for token in self.tokens:
            print(token.token, end=' ')


class Passage:
    def __init__(self, name, text):
        self.name = name
        self.text = name + ". " + text
        self.sentences = []
        self.num_of_tokens = 0
        self.divide_sentences()

    def divide_sentences(self):
        divided_text = re.split(r'(?<![\d+\.+\d+][A-Z][a-z])(?<!\w\.\w.)(?<=\.|\?|\!)\s', self.text)
        for i in range(len(divided_text)):
            if divided_text[i] != '' and divided_text[i] != ' ':
                self.sentences += [Sentence(divided_text[i], i)]

    def print_passage(self):
        for sentence in self.sentences:
            sentence.print_sentence()
            print("\n", end="")
class Corpus:
    def __init__(self, corpus_dir):
        self.passages = []
        self.corpus_dir = corpus_dir
        self.text = ""
        self.num_of_sentences = 0
        self.special_expressions = ['transl.', 'lit.', 'U.S.A.', 'U.S.', 'Dr. ', 'B.Sc.', 'B.A.', 'Ph.D', 'Mr.', 'Ms.', 'Mrs.', 'etc.',
                                    'P.M.',
                                    'A.M.', 'M.D.', 'i.e.', 'mt.', 'no.', 'Sr.', 'vol.', 'vs.'] # Abbreviated expressions

        for file in os.listdir(corpus_dir):
            with open(os.path.join(corpus_dir, file), 'r', encoding='utf8') as f:
                str = f.read()
                # add passage name to the beginning of the text
                str = "== " + file[:-4] + " ==" + str
                self.text += str
        self.clean_text()
        self.divide_passages()

        for passage in self.passages:
            self.num_of_sentences += len(passage.sentences)


    def divide_passages(self):
        divided_text = re.split(r' =====|===== | ===== | ====|==== | ==== | ===|=== | === | ==|== | == ', self.text)
        i = 0;
        while i < (len(divided_text)):
            divided_text[i] = divided_text[i].replace('\n', '')
            if len(divided_text[i]) == 0 or divided_text[i] == '' or divided_text[i] == ' ':
                i += 1
                continue
            if i < len(divided_text)-1 and len(divided_text[i+1]) > 50:
                self.passages += [Passage(divided_text[i], divided_text[i+1].replace('\n', ''))]
                i += 2
            else:
                self.passages += [Passage(divided_text[i], '')]
                i += 1

    def clean_text(self):
        # remove all \n
        # remove emojis
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"   u"\U0001F300-\U0001F5FF"    u"\U0001F680-\U0001F6FF"  u"\U0001F1E0-\U0001F1FF"    u"\U00002500-\U00002BEF"    
                                   u"\U00002702-\U000027B0"  u"\U00002702-\U000027B0"       u"\U000024C2-\U0001F251"  u"\U0001f926-\U0001f937"       u"\U00010000-\U0010ffff" 
                                   u"\u2640-\u2642"  u"\u2600-\u2B55" u"\u200d" u"\u23cf"      u"\u23e9"    u"\u231a"  u"\ufe0f"  # dingbats 
                                   u"\u3030"    "]+", re.UNICODE)
        self.text = re.sub(emoji_pattern, '', self.text)

        # remove http, uml and url
        self.text = re.sub(r'www\S+', '', self.text)
        self.text = re.sub(r'http\S+', '', self.text)
        html_pattern = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});|#([a-z0-9A-Z]+;)')
        self.text = re.sub(html_pattern, '', self.text)

        # We'll hide abbreviations for now
        for j in range(len(self.special_expressions)):
            replacement_str = '#' + str(j)
            self.text = self.text.replace(self.special_expressions[j], replacement_str)

        # change expressions such as Henry D. Ford to Henry D Ford
        callback = lambda pat: pat.group(0)[0]
        self.text = re.sub(r'[A-Z][\.]', callback, self.text)

        #change instances of c. 3566 to c.3566
        callback = lambda pat: pat.group(0)[0:2] + pat.group(0)[3]
        self.text = re.sub(r'c\.\s[\d+]', callback, self.text)



    def print_corpus(self):
        for passage in self.passages:
            passage.print_passage()
            print("\n\n", end="")

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
            return obj.__dict__

    def writeJson(self, obj):
        json_file = json.dumps(self, cls=CustomEncoder, indent=4)
        json
        with open('output.json', 'w', encoding="utf-8") as f:
            f.write(json_file)
            f.close()

if __name__ == "__main__":

    corpus_dir = argv[1]    # The directory in which the Wiki files are, full pathname
    output_file = argv[2]   # The text file that the corpus is written onto, full pathname
    # output_file = "output.txt"
    # corpus_dir = "corpus_dir"

    corpus = Corpus(corpus_dir);


    with open(output_file, 'w', encoding='utf8') as f:
        sys.stdout = f
        corpus.print_corpus()

    # # corpus.write_json()


