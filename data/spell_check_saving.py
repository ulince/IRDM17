import pandas as pd
from spell_check import spell_check as tcheck
import requests
import re
import time
from random import randint
from nltk.stem.porter import *
import csv
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from nltk.stem.snowball import SnowballStemmer
import pickle

stemmer = SnowballStemmer("english")

def str_stem(s):
    strNum = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':0}
    if isinstance(s, str):
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) #Split words with a.A
        s = s.lower()
        s = s.replace("  "," ")
        s = s.replace(",","") #could be number / segment later
        s = s.replace("$"," ")
        s = s.replace("?"," ")
        s = s.replace("-"," ")
        s = s.replace("//","/")
        s = s.replace("..",".")
        s = s.replace(" / "," ")
        s = s.replace(" \\ "," ")
        s = s.replace("."," . ")
        s = re.sub(r"(^\.|/)", r"", s)
        s = re.sub(r"(\.|/)$", r"", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = s.replace(" x "," xbi ")
        s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
        s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
        s = s.replace("*"," xbi ")
        s = s.replace(" by "," xbi ")
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
        s = s.replace("°"," degrees ")
        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
        s = s.replace(" v "," volts ")
        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
        s = s.replace("  "," ")
        s = s.replace(" . "," ")
        s = (" ").join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])
        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        s = s.lower()
        s = s.replace("toliet","toilet")
        s = s.replace("airconditioner","air conditioner")
        s = s.replace("vinal","vinyl")
        s = s.replace("vynal","vinyl")
        s = s.replace("skill","skil")
        s = s.replace("snowbl","snow bl")
        s = s.replace("plexigla","plexi gla")
        s = s.replace("rustoleum","rust-oleum")
        s = s.replace("whirpool","whirlpool")
        s = s.replace("whirlpoolga", "whirlpool ga")
        s = s.replace("whirlpoolstainless","whirlpool stainless")
        return s
    else:
        return "null"


START_SPELL_CHECK = "<span class=\"spell\">Showing results for</span>"
END_SPELL_CHECK = "<br><span class=\"spell_orig\">Search instead for"
HTML_Codes = (
    ("'", '&#39;'),
    ('"', '&quot;'),
    ('>', '&gt;'),
    ('<', '&lt;'),
    ('&', '&amp;'),
)

def spell_checking(s):
    q = '+'.join(s.split())
    time.sleep(randint(0,2)) #relax and don't let google be angry
    r = requests.get("https://www.google.co.uk/search?q="+q)
    content = r.text
    start=content.find(START_SPELL_CHECK)
    if (start > -1):
        start = start + len(START_SPELL_CHECK)
        end=content.find(END_SPELL_CHECK)
        search= content[start:end]
        search = re.sub(r'<[^>]+>', '', search)
        for code in HTML_Codes:
            search = search.replace(code[1], code[0])
        search = search[1:]
    else:
        search = s
    return search


df_train = pd.read_csv('Data_Kaggle/train.csv', encoding="ISO-8859-1")
product_description = pd.read_csv('Data_Kaggle/product_descriptions.csv', encoding="ISO-8859-1")
attributes = pd.read_csv('Data_Kaggle/attributes.csv', encoding="ISO-8859-1")

# for i in range(len(df_train.search_term)):
#      df_train.loc[i, 'search_term'] = str_stem(spell_checking(df_train.search_term[i]))
#      if i%100 == 0:
#          print("Train",i)
#          df_train.to_csv('Data_Kaggle/train_spelled.csv')
#          pickle.dump(i, open("iteration_query" + ".p", "wb"))
#
# for i in range(len(product_description.product_description)):
#      #product_description.loc[i, 'product_description'] = str_stem(spell_checking(product_description.product_description[i]))
#      product_description.loc[i, 'product_description'] = str_stem(product_description.product_description[i])
#      if i%100 == 0:
#          print("Decription",100*i/len(product_description.product_description))
#          product_description.to_csv('Data_Kaggle/product_description_spelled.csv')
#          pickle.dump(i, open("iteration_description" + ".p", "wb"))

for i in range(len(attributes.value)):
    attributes.loc[i, 'value'] = str_stem((attributes.value[i]))
    if i%100 == 0:
        print("Attributes",round(100*i/len(attributes.value),2),"%")

attributes.to_csv('Data_Kaggle/attributes_spelled.csv')


print(df_train.search_term)






