import csv, os,unicodedata
from nltk.tokenize import TweetTokenizer
tokenizer = TweetTokenizer(strip_handles=True,reduce_len=True)
def printAndSaveTweetTextFromCsv(file):
    newfile = open(os.path.splitext(file)[0] + "_cleanLOWERCASE.txt",'w',encoding='ascii')
    with open(file, encoding='latin-1') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            normal = [x.lower() for x in row['text'].split()]
            normal_stringed = ' '.join(map(str, normal))
            cleaned = unicodedata.normalize('NFKD',normal_stringed).encode('ascii','ignore')
            tokenized = tokenizer.tokenize(cleaned)
            if len(tokenized)>0:
                cut = tokenized[2:]
                full = tokenized
                #print( cut if tokenized[0] == "RT" else full )
                newfile.write(' '.join(cut)) if tokenized[0] == "rt" else newfile.write(' '.join(full))
                newfile.write('\n')

def testNLTK(file):
    with open(file, encoding='latin-1') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            normal = [x.lower() for x in row['text'].split()]
            normal_stringed = ' '.join(map(str, normal))
            cleaned = unicodedata.normalize('NFKD', normal_stringed).encode('ascii', 'ignore')

            print('normal',normal)
            print('cleaned',cleaned)
            tokenized = tokenizer.tokenize(cleaned)
            print('tokenized', tokenized)


if __name__ == "__main__":
    printAndSaveTweetTextFromCsv("real.csv")
    printAndSaveTweetTextFromCsv("fake.csv")
    testNLTK("real.csv")