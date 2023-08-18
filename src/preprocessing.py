"""Preprocessing functions."""

import re
import nltk
import unicodedata
import contractions
import spacy

nlp = spacy.load('en_core_web_sm',  disable=['parser', 'ner'])
ps = nltk.stem.SnowballStemmer(language='english')

def remove_urls(text):
    """Remove URLs.

    Args:
        text (string): text

    Returns:
        String : String after removing URLs
    """
    pattern = re.compile(r'https?://\S+|www\.\S+')
    text = pattern.sub(r'', str(text)).strip()
    return text

    
def strip_html_tags(text):
    """Remove HTML tags

    Args:
        text (String): text

    Returns:
        String: String without HTML tags
    """
    pattern = re.compile('<.*?>')
    return pattern.sub(r'', str(text))


def remove_accented_chars(text):
    """Remove accented characters.

    Examples ë - e, õ - o
    
    Args:
        text (String): text

    Returns:
        String: String without accented characters.
    """
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore') \
                                              .decode('utf-8', 'ignore')
    return text.strip()


def expand_contractions(text):
    """Expand contractions
    
    Example he’d - he would
    
    Args:
        text (String): String

    Returns:
        String: Expanded string
    """
    return contractions.fix(text).strip()


def spacy_lemmatize_text(text, lemmatizer):
    """Lemmatize text

    Args:
        text (String): text
        lemmatizer : Lemmatizer
    Returns:
        String: String after Lemmatization
    """
    text = lemmatizer(text)
    text = ' '.join([word.lemma_    if word.lemma_ != '-PRON-'
                                    else word.text
                                    for word in text])
    return text


def simple_stemming(text, stemmer):
    """Stemmer

    Args:
        text (String): text
        stemmer (Stemmer): Stemmer object

    Returns:
        String: String after stemming
    """
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text


def remove_special_characters(text, remove_digits=False):
    """Remove special characters.

    Args:
        text (String): text
        remove_digits (bool, optional): Whether to remove digits or not.
                                        Defaults to False.

    Returns:
        String: String without special characters
    """
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    text = re.sub(pattern, ' ', text).strip()
    return text


def remove_stopwords(text, stopwords=None):
    """Remove stopwords

    Args:
        text (String): text
        stopwords (list, optional): List of stopwords.
                                    If not passed nltk stopwords are used.

    Returns:
        String: string without stopwords
    """
    if not stopwords:
        stopwords = nltk.corpus.stopwords.words('english')
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token not in stopwords]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text.strip()

def text_pre_processor(text,
                       html_strip=True,
                       url_removal=True,
                       accented_char_removal=True,
                       contraction_expansion=True,
                       text_lower_case=True,
                       text_stemming=False,
                       stemmer=None,
                       text_lemmatization=False, 
                       lemmatizer=None,
                       special_char_removal=True,
                       remove_digits=False,
                       stopword_removal=False, 
                       stopword_list=None):
    """Text preprocessor.

    Args:
        text (String): text
        html_strip (bool, optional): Strip HTML tags. Defaults to True.
        url_removal (bool, optional): Remove URLs. Defaults to True.
        accented_char_removal (bool, optional): Remove accented characters. Defaults to True.
        contraction_expansion (bool, optional): Expand contracted words. Defaults to True.
        text_lower_case (bool, optional): Convert to lowercase. Defaults to True.
        text_stemming (bool, optional): Stemming. Defaults to True.
        stemmer (_type_, optional): Stemmer object. Defaults to None.
        text_lemmatization (bool, optional): Lemmatize. Defaults to False.
        lemmatizer (_type_, optional): Lemmatize object. Defaults to None.
        special_char_removal (bool, optional): Remove special char. Defaults to True.
        remove_digits (bool, optional): Remove digits. Defaults to False.
        stopword_removal (bool, optional): Remove stopwords. Defaults to False.
        stopword_list (_type_, optional): Stopword list. If not passed nltk is used.

    Returns:
        String: Processed text
    """
    # lowercase the text    
    if text_lower_case:
        text = text.lower()

    # strip HTML
    if html_strip:
        text = strip_html_tags(text)
    
    # Remove URL
    if url_removal:
        text = remove_urls(text)

    # remove extra newlines (in case of noisy text)
    text = text.translate(text.maketrans("\n\t\r", "   "))
    
    # remove accented characters
    if accented_char_removal:
        text = remove_accented_chars(text)
    
    # expand contractions    
    if contraction_expansion:
        text = expand_contractions(text)
    
    # remove special characters and\or digits    
    if special_char_removal:
        # insert spaces between special characters to isolate them    
        special_char_pattern = re.compile(r'([{.(-)!}])')
        text = special_char_pattern.sub(" \\1 ", text)
        text = remove_special_characters(text, remove_digits=remove_digits)  
        
    # remove stopwords
    if stopword_removal:
        text = remove_stopwords(text, stopwords=stopword_list)
    
    # lemmatize text
    if text_lemmatization:
        assert lemmatizer != None, "Pass a valid lemmatizer"
        text = spacy_lemmatize_text(text, lemmatizer) 
        
    # stem text
    if text_stemming and not text_lemmatization:
        assert stemmer != None, "Pass a valid stemmer"
        text = simple_stemming(text, stemmer)

    # remove extra whitespace
    text = re.sub(' +', ' ', text)
    text = text.strip()
    
    return text
