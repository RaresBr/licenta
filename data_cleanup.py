import csv, os,unicodedata
from nltk.tokenize import TweetTokenizer
tokenizer = TweetTokenizer(strip_handles=True,reduce_len=True)
def printTweetTextFromCsv(file):
    newfile = open(os.path.splitext(file)[0] + "_clean.txt",'w',encoding='ascii')
    with open(file, encoding='latin-1') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            cleaned = unicodedata.normalize('NFKD',row["text"]).encode('ascii','ignore')
            tokenized = tokenizer.tokenize(cleaned)
            if len(tokenized)>0:
                cut = tokenized[2:]
                full = tokenized
                print( cut if tokenized[0] == "RT" else full )
                newfile.write(' '.join(cut)) if tokenized[0] == "RT" else newfile.write(' '.join(full))
                newfile.write('\n')

if __name__ == "__main__":
    printTweetTextFromCsv("real.csv")
    printTweetTextFromCsv("fake.csv")