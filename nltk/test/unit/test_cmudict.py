import pytest
import nltk
from nltk.corpus import *
from nltk.corpus.reader import cmudict as cmu 

#unit tests for the cmudict corpus reader, particularly IPA transcription additions


def test_cmudict_ipa_all_sounds():
    #Testing a few words for their IPA transcriptions
    word_ipa_pairs = {
        #Sample words to show each possible sound in Standard American English
        #(Taken from header comments of cmudict.py)
        "odd"   :    ["ˈɑ", "d"],   
        "hut"   :    ["h", "ˈʌ", "t"],    
        "cow"   :    ["k", "ˈaʊ"],     
        "be"    :    ["b", "ˈi"],
        "dee"   :    ["d", "ˈi"],
        "ed"    :    ["ˈɛ", "d"],
        "ate"   :    ["ˈeɪ", "t"],
        "green" :    ["ɡ", "r", "ˈi", "n"],
        "it"    :    ["ˈɪ", "t"],
        "gee"   :    ["dʒ", "ˈi"],
        "lee"   :    ["l", "ˈi"],
        "knee"  :    ["n", "ˈi"],
        "oat"   :    ["ˈoʊ", "t"],
        "pee"   :    ["p", "ˈi"],
        "sea"   :    ["s", "ˈi"],
        "tea"   :    ["t", "ˈi"],
        "hood"  :    ["h", "ˈʊ", "d"],
        "vee"   :    ["v", "ˈi"],
        "yield" :    ["j", "ˈi", "l", "d"],
        "seizure":   ["s", "ˈi", "ʒ", "ɚ"],
        "at"    :    ["ˈæ", "t"],
        "ought" :    ["ˈɔ", "t"],
        "hide"  :    ["h", "ˈaɪ", "d"],
        "cheese":    ["tʃ", "ˈi", "z"],
        "thee"  :    ["ð", "ˈi"],
        "hurt"  :    ["h", "ˈɚ", "t"],
        "fee"   :    ["f", "ˈi"],
        "he"    :    ["h", "ˈi"],
        "eat"   :    ["ˈi", "t"],
        "key"   :    ["k", "ˈi"],
        "me"    :    ["m", "ˈi"],
        "ping"  :    ["p", "ˈɪ", "ŋ"],
        "toy"   :    ["t", "ˈɔɪ"],
        #Changed read to road since read is a homonym
        "road"  :    ["r", "ˈoʊ", "d"], 
        "she"   :    ["ʃ", "ˈi"], 
        "theta" :    ["θ", "ˈeɪ", "t", "ə"],
        "two"   :    ["t", "ˈu"],
        "we"    :    ["w", "ˈi"],
        "zee"   :    ["z", "ˈi"],
    }

    #Load the CMU Pronouncing Dictionary
    c = cmu.CMUDictCorpusReader(nltk.data.find("corpora/cmudict"), "cmudict")

    for word, expected_ipa in word_ipa_pairs.items():
        phones = c.dict("IPA")[word][0]
        assert phones == expected_ipa

def test_cmudict_ipa_example():    
    ph = ["ɪ", "ɡ", "z", "ˈæ", "m", "p", "ə", "l"]
    c = cmu.CMUDictCorpusReader(nltk.data.find("corpora/cmudict"), "cmudict")
    phones = c.dict("IPA")["example"][0]
    assert phones == ph


def test_cmudict_ipa_mid_vowels():
    #The word bubble has both the /ə/ and /ʌ/ sounds, differentiated here by stress on the initial one. Arpa represents these the same but with stress numbers.
    ph = ["b", "ˈʌ", "b", "ə", "l"]
    c = cmu.CMUDictCorpusReader(nltk.data.find("corpora/cmudict"), "cmudict")
    phones = c.dict("IPA")["bubble"][0]
    assert phones == ph

def test_cmudict_ipa_multi_stress():    
    ph = ["r", "ˌɛ", "k", "r", "i", "ˈeɪ", "ʃ", "ə", "n"]
    c = cmu.CMUDictCorpusReader(nltk.data.find("corpora/cmudict"), "cmudict")
    phones = c.dict("IPA")["recreation"][0]
    assert phones == ph