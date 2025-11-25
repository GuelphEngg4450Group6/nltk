# Natural Language Toolkit: Carnegie Mellon Pronouncing Dictionary Corpus Reader
#
# Copyright (C) 2001-2025 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
The Carnegie Mellon Pronouncing Dictionary [cmudict.0.6]
ftp://ftp.cs.cmu.edu/project/speech/dict/
Copyright 1998 Carnegie Mellon University

File Format: Each line consists of an uppercased word, a counter
(for alternative pronunciations), and a transcription.  Vowels are
marked for stress (1=primary, 2=secondary, 0=no stress).  E.g.:
NATURAL 1 N AE1 CH ER0 AH0 L

The dictionary contains 127069 entries.  Of these, 119400 words are assigned
a unique pronunciation, 6830 words have two pronunciations, and 839 words have
three or more pronunciations.  Many of these are fast-speech variants.

Phonemes: There are 39 phonemes, as shown below:

Phoneme Example Translation    Phoneme Example Translation
------- ------- -----------    ------- ------- -----------
AA      odd     AA D           AE      at      AE T
AH      hut     HH AH T        AO      ought   AO T
AW      cow     K AW           AY      hide    HH AY D
B       be      B IY           CH      cheese  CH IY Z
D       dee     D IY           DH      thee    DH IY
EH      Ed      EH D           ER      hurt    HH ER T
EY      ate     EY T           F       fee     F IY
G       green   G R IY N       HH      he      HH IY
IH      it      IH T           IY      eat     IY T
JH      gee     JH IY          K       key     K IY
L       lee     L IY           M       me      M IY
N       knee    N IY           NG      ping    P IH NG
OW      oat     OW T           OY      toy     T OY
P       pee     P IY           R       read    R IY D
S       sea     S IY           SH      she     SH IY
T       tea     T IY           TH      theta   TH EY T AH
UH      hood    HH UH D        UW      two     T UW
V       vee     V IY           W       we      W IY
Y       yield   Y IY L D       Z       zee     Z IY
ZH      seizure S IY ZH ER
"""

from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.util import Index
from nltk import tokenize


class CMUDictCorpusReader(CorpusReader):
    def entries(self, transcription_format: str = "ARPA"):
        """
        :return: the cmudict lexicon as a list of entries
            containing (word, transcriptions) tuples.
        """
        if transcription_format == "IPA":
            return concat(
                [
                    StreamBackedCorpusView(fileid, read_cmudict_block_IPA, encoding=enc)
                    for fileid, enc in self.abspaths(None, True)
                ]
            )
        else:
            return concat(
                [
                    StreamBackedCorpusView(fileid, read_cmudict_block, encoding=enc)
                    for fileid, enc in self.abspaths(None, True)
                ]
            )

    def words(self, transcription_format: str = "ARPA"):
        """
        :return: a list of all words defined in the cmudict lexicon.
        """
        return [word.lower() for (word, _) in self.entries(transcription_format)]

    def dict(self, transcription_format: str = "ARPA"):
        """
        :return: the cmudict lexicon as a dictionary, whose keys are
            lowercase words and whose values are lists of pronunciations.
        """
        return dict(Index(self.entries(transcription_format)))
    
    def phones_to_word(self, phones: list, transcription_format: str = "ARPA"):
        """
        :return: a list of all words with matching pronunciations (exact match required).
        """
        d = self.dict(transcription_format).items() #get dictionary items
        l = list()
        for word, pronunc in d: #iterate through dictionary
            if phones in pronunc:   #check each IPA transcription for match
                l.append(word)      #add words corresponding to sound matches to output list
        return l

    def tok_by_phone(self, words: str, transcription_format: str = "ARPA"):
        """
        :return: a set of all the sounds used in the passed-in sentence/word.
        """
        invalidWords = list()
        phones = list()
        toks = tokenize.word_tokenize(words)
        for w in toks:
            try:
                t = self.dict(transcription_format)[w.lower()][0] #just take first pronunciation for each word
                for p in t:
                    print(p)
                    phones.append(p)  
            except KeyError as e:
                invalidWords.append(w)
        return [list(set(phones)), invalidWords]    #emove duplicates with set conversion
        


def read_cmudict_block(stream): #default, ARPA transcription mode
    entries = []
    while len(entries) < 100:  # Read 100 at a time.
        line = stream.readline()
        if line == "":
            return entries  # end of file.
        pieces = line.split()
        entries.append((pieces[0].lower(), pieces[2:]))
        # print(entries);
    return entries

def read_cmudict_block_IPA(stream): #if user wants IPA conversion, use this block reader
    entries = []
    while len(entries) < 100:  # Read 100 at a time.
        line = stream.readline()
        if line == "":
            return entries  # end of file.
        pieces = line.split() 
        # print(pieces[0].lower())
        # print("\n")
        # print(pieces[2:])
        # print("\n")
        temp_phones = arpa_to_ipa(pieces[2:])
        for i in range(len(temp_phones)):
            if temp_phones[i] == "ˈə":  #Often considered allophones in English, ə and ʌ are somewhat interchangeable.
                temp_phones[i] = "ˈʌ"   #Typically, however, ə is used in unstressed syllables while ʌ is used in stressed syllables.
        entries.append((pieces[0].lower(), temp_phones))
    return entries

def arpa_to_ipa(arpa_phonemes):
    st_map = {
        "0": "",  #no stress
        "1": "ˈ", #primary stress
        "2": "ˌ"  #secondary stress
    }
    ph_map = {  #phoneme mappings
        "AA": "ɑ",
        "AE": "æ",
        "AH": "ə",
        "AO": "ɔ",
        "AW": "aʊ",
        "AY": "aɪ",
        "B": "b",
        "CH": "tʃ",
        "D": "d",
        "DH": "ð",
        "EH": "ɛ",
        "ER": "ɚ",
        "EY": "eɪ",
        "F": "f",
        "G": "ɡ",
        "HH": "h",
        "IH": "ɪ",
        "IY": "i",
        "JH": "dʒ",
        "K": "k",
        "L": "l",
        "M": "m",
        "N": "n",
        "NG": "ŋ",
        "OW": "oʊ",
        "OY": "ɔɪ",
        "P": "p",
        "R": "r",
        "S": "s",
        "SH": "ʃ",
        "T": "t",
        "TH": "θ",
        "UH": "ʊ",
        "UW": "u",
        "V": "v",
        "W": "w",
        "Y": "j",
        "Z": "z",
        "ZH": "ʒ"
    }

    # print(type(arpa_phonemes));
    if type(arpa_phonemes) is str:
        ph = str("")  #string for current phoneme
        temp_ph = str("") #temporary holder for phoneme without stress marker
        if '0' in arpa_phonemes:
            temp_ph = arpa_phonemes[:-1]  #removes unstressed vowel marker
            ph = ""
        elif '1' in arpa_phonemes:
            temp_ph = arpa_phonemes[:-1]  #removes primary stress marker,
            ph = "\ˈ";                  #adds IPA primary stress marker at beginning of phoneme
        elif '2' in arpa_phonemes:
            temp_ph = arpa_phonemes[:-1]  #removes secondary stress marker
            ph = "\ˌ";                  #adds IPA secondary stress marker at beginning of phoneme

        if temp_ph in ph_map:            
            ph = ph + (ph_map[temp_ph]) #appends corresponding IPA phoneme to stress marker (if one exists)
        else:
            ph = ph + "???"
        return ph
    elif type(arpa_phonemes) is list: 
        # print(arpa_phonemes)
        l = list()
        ph = str("")  
        temp_ph = str("") 
        for phone in arpa_phonemes:    
            if '0' in phone:
                temp_ph = phone[:-1]  #removes unstressed vowel marker
                ph = ""
            elif '1' in phone:
                temp_ph = phone[:-1]  #removes primary stress marker,
                ph = "ˈ"                  #adds IPA primary stress marker at beginning of phoneme
            elif '2' in phone:
                temp_ph = phone[:-1]  #removes secondary stress marker
                ph = "ˌ"                  #adds IPA secondary stress marker at beginning of phoneme
            else:
                temp_ph = phone
                ph = ""

            if temp_ph in ph_map:            
                ph = ph + (ph_map[temp_ph]) #appends corresponding IPA phoneme to stress marker (if one exists)
            else:
                ph = ph + "???"             #if unknown ARPA symbol, write "???" as placeholder
            # print(ph)
            l.append(ph)
        return l
    else:
        print("Please provide either a string or a list of strings composed of ARPAbet phonemes as input.")
        return None
