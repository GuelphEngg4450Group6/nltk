import pytest
import nltk
from nltk.corpus import *
from nltk.corpus.reader import cmudict as cmu 

#unit tests for the cmudict corpus reader, particularly IPA transcription additions


def test_issue_3260_no_space_before_closing_double_quote():
    detok = TreebankWordDetokenizer()
    s = ['hello', ',', "''", 'world', '.']
    assert detok.detokenize(s) == 'hello," world.'

def test_cmudict_ipa_transcription():
    # Load the CMU Pronouncing Dictionary
    c = cmu.CMUDictCorpusReader(nltk.data.find("corpora/cmudict"), "cmudict")

    # Test a few words for their IPA transcriptions
    word_ipa_pairs = {
        "example":   ["ɪ", "ɡ", "z", "ˈæ", "m", "p", "ə" "l"],
        "odd"   :    ["ˈɑ", "d"],   
        "hut"   :    ["h", "ˈʌ", "t"],    
        "cow"   :    ["k", "ˈaʊ"],     
        "be"    :    ["b", "i"],
        "dee"   :    ["d", "i"],
        "Ed"    :    ["ˈɛ", "d"],
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
        "read"  :    ["r", "ˈi", "d"],
        "she"   :    ["ʃ", "ˈi"], 
        "theta" :    ["θ", "ˈeɪ", "t", "ʌ"],
        "two"   :    ["t", "ˈu"],
        "we"    :    ["w", "ˈi"],
        "zee"   :    ["z", "ˈi"],
    }

    for word, expected_ipa in word_ipa_pairs.items():
        phones = c.dict("IPA")[word]
        assert phones == expected_ipa