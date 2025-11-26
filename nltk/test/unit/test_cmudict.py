import pytest
import nltk
from nltk.corpus.reader import cmudict as cmu
from nltk.corpus.reader.cmudict import CMUDictCorpusReader


class DummyCMU(CMUDictCorpusReader):
    def __init__(self):
        # root and fileids are not used because we override dict
        super().__init__(root="", fileids=[])

    def dict(self, transcription_format: str = "ARPA"):
        # Small fake dictionary to white box phones_to_word
        return {
            "cat": [["K", "AE1", "T"]],
            "bat": [["B", "AE1", "T"], ["B", "AE2", "T"]],
            "dog": [["D", "AO1", "G"]],
            "kat": [["K", "AE1", "T"]],
        }


# unit tests for the cmudict corpus reader, particularly IPA transcription additions

def test_cmudict_ipa_all_sounds():
    # Testing a few words for their IPA transcriptions
    word_ipa_pairs = {
        # Sample words to show each possible sound in Standard American English
        # (Taken from header comments of cmudict.py)
        "odd": ["ˈɑ", "d"],
        "hut": ["h", "ˈʌ", "t"],
        "cow": ["k", "ˈaʊ"],
        "be": ["b", "ˈi"],
        "dee": ["d", "ˈi"],
        "ed": ["ˈɛ", "d"],
        "ate": ["ˈeɪ", "t"],
        "green": ["ɡ", "r", "ˈi", "n"],
        "it": ["ˈɪ", "t"],
        "gee": ["dʒ", "ˈi"],
        "lee": ["l", "ˈi"],
        "knee": ["n", "ˈi"],
        "oat": ["ˈoʊ", "t"],
        "pee": ["p", "ˈi"],
        "sea": ["s", "ˈi"],
        "tea": ["t", "ˈi"],
        "hood": ["h", "ˈʊ", "d"],
        "vee": ["v", "ˈi"],
        "yield": ["j", "ˈi", "l", "d"],
        "seizure": ["s", "ˈi", "ʒ", "ɚ"],
        "at": ["ˈæ", "t"],
        "ought": ["ˈɔ", "t"],
        "hide": ["h", "ˈaɪ", "d"],
        "cheese": ["tʃ", "ˈi", "z"],
        "thee": ["ð", "ˈi"],
        "hurt": ["h", "ˈɚ", "t"],
        "fee": ["f", "ˈi"],
        "he": ["h", "ˈi"],
        "eat": ["ˈi", "t"],
        "key": ["k", "ˈi"],
        "me": ["m", "ˈi"],
        "ping": ["p", "ˈɪ", "ŋ"],
        "toy": ["t", "ˈɔɪ"],
        # Changed read to road since read is a homonym
        "road": ["r", "ˈoʊ", "d"],
        "she": ["ʃ", "ˈi"],
        "theta": ["θ", "ˈeɪ", "t", "ə"],
        "two": ["t", "ˈu"],
        "we": ["w", "ˈi"],
        "zee": ["z", "ˈi"],
    }

    # Load the CMU Pronouncing Dictionary
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
    # The word bubble has both the /ə/ and /ʌ/ sounds, differentiated here by stress on the initial one.
    ph = ["b", "ˈʌ", "b", "ə", "l"]
    c = cmu.CMUDictCorpusReader(nltk.data.find("corpora/cmudict"), "cmudict")
    phones = c.dict("IPA")["bubble"][0]
    assert phones == ph


def test_cmudict_ipa_multi_stress():
    # The word recreation has primary, secondary, and unstressed syllables.
    ph = ["r", "ˌɛ", "k", "r", "i", "ˈeɪ", "ʃ", "ə", "n"]
    c = cmu.CMUDictCorpusReader(nltk.data.find("corpora/cmudict"), "cmudict")
    phones = c.dict("IPA")["recreation"][0]
    assert phones == ph


def test_phones_to_word_integration():
    ph = ["p", "ə", "t", "ˈeɪ", "t", "ˌoʊ"]
    c = cmu.CMUDictCorpusReader(nltk.data.find("corpora/cmudict"), "cmudict")
    words = c.phones_to_word(ph, "IPA")
    assert "potato" in words


# white box tests for phones_to_word using DummyCMU

def test_phones_to_word_single_pron_multiple_words():
    c = DummyCMU()
    result = c.phones_to_word(["K", "AE1", "T"])
    assert set(result) == {"cat", "kat"}


def test_phones_to_word_multiple_prons_match_second():
    c = DummyCMU()
    result = c.phones_to_word(["B", "AE2", "T"])
    assert result == ["bat"]


def test_phones_to_word_no_match():
    c = DummyCMU()
    result = c.phones_to_word(["Z", "IY1"])
    assert result == []


# white box tests for tok_by_phone

def test_tok_by_phone_basic_ipa():
    text = "Hello world, this is a test, ppoottaattooeess."
    c = cmu.CMUDictCorpusReader(nltk.data.find("corpora/cmudict"), "cmudict")
    phones, unknown_words = c.tok_by_phone(text, "IPA")

    expected_sounds = {
        "h", "ə", "l", "ˈoʊ",    # hello
        "w", "ˈɚ", "d",          # world
        "ð", "ˈɪ", "s",          # this
        "t", "ˈɛ",               # test
    }

    for ph in expected_sounds:
        assert ph in phones

    assert "ppoottaattooeess" in unknown_words


def test_tok_by_phone_removes_duplicate_phones():
    text = "odd odd"
    c = cmu.CMUDictCorpusReader(nltk.data.find("corpora/cmudict"), "cmudict")

    phones, unknown_words = c.tok_by_phone(text, "IPA")

    single_pron = c.dict("IPA")["odd"][0]
    assert len(phones) <= len(single_pron)
    assert unknown_words == []


def test_tok_by_phone_arpa_transcription():
    text = "hello world"
    c = cmu.CMUDictCorpusReader(nltk.data.find("corpora/cmudict"), "cmudict")

    phones, unknown_words = c.tok_by_phone(text, "ARPA")

    assert any(ph.endswith(("0", "1", "2")) for ph in phones)
    assert "hello" not in unknown_words
    assert "world" not in unknown_words


def test_tok_by_phone_all_unknown():
    text = "qwertyuiop"
    c = cmu.CMUDictCorpusReader(nltk.data.find("corpora/cmudict"), "cmudict")

    phones, unknown_words = c.tok_by_phone(text, "IPA")

    assert phones == []
    assert "qwertyuiop" in unknown_words


# white box tests for arpa_to_ipa via module function

def test_arpa_to_ipa_string_unstressed():
    assert cmu.arpa_to_ipa("AH0") == "ə"


def test_arpa_to_ipa_string_primary():
    assert cmu.arpa_to_ipa("AE1") == "ˈæ"


def test_arpa_to_ipa_string_secondary():
    assert cmu.arpa_to_ipa("AO2") == "ˌɔ"


def test_arpa_to_ipa_string_unknown_symbol():
    assert cmu.arpa_to_ipa("ZZ1") == "ˈ???"


def test_arpa_to_ipa_list_mixed_stress():
    result = cmu.arpa_to_ipa(["K", "AE1", "T"])
    assert result == ["k", "ˈæ", "t"]


def test_arpa_to_ipa_list_unknown_inside():
    result = cmu.arpa_to_ipa(["B", "XX0", "AE1"])
    assert result == ["b", "???", "ˈæ"]


def test_arpa_to_ipa_list_consonant_no_stress():
    result = cmu.arpa_to_ipa(["SH"])
    assert result == ["ʃ"]


def test_arpa_to_ipa_wrong_type_returns_none():
    result = cmu.arpa_to_ipa(123)
    assert result is None
