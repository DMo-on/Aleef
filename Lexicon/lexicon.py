import nltk
import re
import pickle
from nltk.stem.arlstem import ARLSTem
from collections import defaultdict
import re
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk import ConditionalFreqDist,ConditionalProbDist, MLEProbDist, ELEProbDist
import pickle 
import numpy as np
from numpy import unravel_index
from nltk.stem.arlstem import ARLSTem
from nltk.stem.isri import ISRIStemmer
import itertools
from model import Corpus, Model
from nltk import ConditionalFreqDist

# =============================================================================
#               Enrich Lexicon with WikiFane tokens
# =============================================================================
def enr_lex():
    EN_dict = defaultdict(str) # Dict of Arabic Name Entities
    all_EN_dict = defaultdict(set) # Dict of Arabic NE + Other tags
    old_lexicon = []
    isri = ISRIStemmer()
    arabic_stmr = ARLSTem()
    dest_ne_lexicon = open("NE_Lexicon_2019.txt",encoding = "Windows-1256",mode = "w")
    # ------------- ---Ar_Name Lexicon Unique tag -----------------------
    f = open("ArNameLexicon.txt", encoding = "Windows-1256", mode = "r")
    dest = open("ArNameLexicon_2019.txt",encoding = "Windows-1256",mode = "w")
    lines = f.read().split("\n")
    for line in lines:
        if len(line) > 0:
            token,NE_tag = line.split()
    
            if token.startswith("و"): # because we don't want initial waw removd
                waaw = re.findall("(^و+)", token)[0]
                len_w = len(waaw)
                token = waaw + arabic_stmr.norm(token[len_w:])
            else:
                token = arabic_stmr.norm(token)
    
            old_lexicon.append(token)
            
            EN_dict[token]=NE_tag
    print(len(EN_dict.keys()))
    f.close()
    c = Corpus('WikiFane')
    words,tags = c.reader_fane()
    all_paires = list(zip(words,tags))
    p=0
    v=0
    w=0
    pr=0
    l=0
    o=0
    tagset = ['VEHICLE','WEAPON','PRODUCT','PERSON','LOC','ORG']
    paires = [(w,t) for w,t in all_paires if t in tagset]
    f = open("lexicon_dict.pkl","rb")
    all_dict = pickle.load(f)
    all_EN_dict = defaultdict(set,all_dict)
    print(type(all_EN_dict))
    f.close()
    cfd  = ConditionalFreqDist(paires)
    clues = set(['OCLUE','PCLUE','LCLUE'])
    for f in cfd:
        if f not in old_lexicon and len(f)>2 and len(all_EN_dict[f].intersection(clues))==0:
            if cfd[f].max() == 'PERSON':
                p+=1
            elif cfd[f].max() == 'VEHICLE':
                v+=1
            elif cfd[f].max() == 'WEAPON':
                w+=1
            elif cfd[f].max() == 'PRODUCT':
                pr+=1
            elif cfd[f].max() == 'LOC':
                l+=1
            elif cfd[f].max() == 'ORG':
                o+=1
            EN_dict[f]=cfd[f].max()
    print("****************")
    print(p,v,w,pr,l,o)
    print(all_EN_dict['ها'])
    for i in EN_dict:
        dest.write(i+"\t"+EN_dict[i])
        dest.write("\n")
    dest.close()
    for w,t in all_paires:
        if len(w)>1 and t not in ['NPREFIX','DEF','CONJ','PREP','START','END'] and len(all_EN_dict[w].intersection(clues))==0:
            all_EN_dict[w].add(t)
    print(all_EN_dict['ها'])
    for j in all_EN_dict:
        dest_ne_lexicon.write(j+"\t"+"\t".join(all_EN_dict[j]))
        dest_ne_lexicon.write("\n")
    dest_ne_lexicon.close()
    arne_pkl = open('arne_lexicon.pkl', 'wb')
    pickle.dump(EN_dict,arne_pkl)
    arne_pkl.close()
    ne_pkl = open('ne_lexicon.pkl', 'wb')
    pickle.dump(all_EN_dict,ne_pkl)
    ne_pkl.close()
# =============================================================================
# =============================================================================
#                   Merging ArNameLexicon + NE_lexicon
# =============================================================================
# =============================================================================

def merge_lexicons():
    # ------------- this guy only has tag unique -----------------------
    f = open("ArNameLexicon.txt", encoding = "Windows-1256", mode = "r")
    
    lines = f.read().split("\n")
    
    EN_dict = dict()
    for line in lines:
        if len(line) > 0:
            token,NE_tag = line.split()
    
            if token.startswith("و"): # because we don't want initial waw removd
                waaw = re.findall("(^و+)", token)[0]
                len_w = len(waaw)
                token = waaw + arabic_stmr.norm(token[len_w:])
            else:
                token = arabic_stmr.norm(token)
    
            temp = set()
            temp.add(NE_tag)
            
            EN_dict[token] = temp # a vect of a single tag (need later)
        
    f.close()
    
    # -----------------  can have more than a single tag ----------------
    f = open("NELexicon.txt", encoding = "Windows-1256", mode = "r")
    
    lines = f.read().split("\n")
    
    for line in lines:
        tokens = line.split()
        if len(tokens) > 1:
            if tokens[1] != 'OTHER':
                if tokens[0].startswith("و"): # because we don't want initial waw removd
                    waaw = re.findall("(^و+)", tokens[0])[0]
                    len_w = len(waaw)
                    tokens[0] = waaw + arabic_stmr.norm(tokens[0][len_w:])
                else:
                    tokens[0] = arabic_stmr.norm(tokens[0])
        
                if tokens[0] not in EN_dict : # if it's not there we add it
                    temp = set(tokens[1:])
                    EN_dict[tokens[0]] = temp
                else : # otherwise we add the tags it can take
                    EN_dict[tokens[0]] = EN_dict[tokens[0]].union(set(tokens[1:]))
        
    f.close()
    # ______________ save it to a picle file ____________
    out = open(r"lexicon_dict.pkl","wb")
    pickle.dump(EN_dict, out)
    out.close()

if __name__== '__main__':
    enr_lex()