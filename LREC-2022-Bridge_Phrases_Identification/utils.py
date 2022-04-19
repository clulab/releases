# Author: Fan luo
 
from spacy.lang.en.stop_words import STOP_WORDS           
 
def lower(text):
    return text.lower()

def punc(text):
    text = text.replace( "–", "")              # remove range char, 2010–11, – is diff from -
    text = text.replace( "&quot;", "\"")
    text = text.replace( "&amp;", " ")
    text = text.replace( "'s ", " ")
    text = re.sub('(?<=\d),(?=\s\d)', '', text )  # remove , in date: january 2, 2012
    text = re.sub('(?<=\d\s),(?=\s\d)', '', text )  # remove , in date: january 2 , 2012
    text = re.sub('(?<=\d),(?=\d)', '', text)     # remove , in number: 1,000
    text = re.sub("([\^_&=\+])", ' ', text)  # remove special punctuations
#     text = re.sub("([\"])", '\'', text)  # replace " with ' for NER, such as "evolution" in question '5a877e5d5542993e715abf7d',
    text = re.sub("([()])", ',', text)   # replace () with , for NER, such as Bronwyn Kathleen Bishop ( née Setright ; born 19 October 1942 ) 
    text = re.sub(
        "(['?!,:`;{|}~<>\-\"])", r' \1 ', text
    )  # add space before and after punctuations, such as 'First Italo-Ethiopian War'
    
    # remove leading and tailing spaces and punctuations
    return text.strip().strip(string.punctuation.replace( "[", ""))  # do not strip "[" to avoid removing phrase prefix as [P9]


def remove_articles(text):
    return re.sub(r'\b(a|an|the)\b', ' ', text)


def white_space_fix(text):
    text = ' '.join(text.split())
    return text.strip()


def remove_punc(text):
    text = text.replace( "–", "")              # remove range char, 2010–11, – is diff from -
    text = text.replace( "&quot;", "\"")
    text = text.replace( "&amp;", " ")
    text = text.replace( "'s ", " ")
    text = re.sub('(?<=\d),(?=\s\d)', '', text )  # remove , in date: january 2, 2012
    text = re.sub('(?<=\d\s),(?=\s\d)', '', text )  # remove , in date: january 2 , 2012
    text = re.sub('(?<=\d),(?=\d)', '', text)     # remove , in number: 1,000
    exclude = set(string.punctuation)  | {u"‘", u"’", u"´", u"`", "_"}  
    return white_space_fix(''.join(ch if ch not in exclude else ' ' for ch in text ))
 
def remove_stop_words(text):
    all_stopwords = set(STOP_WORDS)
    text = re.sub(
        "(['?!,:`;(){|}~<>\-\"])", r' \1', text
    )  # add space before punct before stop words match, for example 's, city?
    text = ' '.join(word for word in text.split()
                    if word.lower() not in all_stopwords and singular_phrase(word).lower() not in all_stopwords)
    return text


wh_words = set([
    "what", "when", 'where', "which", "who", "whom", "whose", "why", "how",
    "whether", "What", "When", 'Where', "Which", "Who", "Whom", "Whose",
    "Why", "How", "Whether", 'other', 'same', 'another', 'many', 'much'
])
def remove_wh_words(text):
    return ' '.join(word if word not in wh_words else ', ' for word in text.split() )   # replce with , for NER. For example: Sugar Bowl where Louisville Cardinals defeated Florida 


def normalize_text(s):
    return white_space_fix(remove_punc(remove_articles(remove_wh_words(str(s)))))

def basic_normalize(s):
    return white_space_fix(punc(remove_wh_words(str(s))))


def singular_phrase(phrase):
    if phrase.strip() == '':
        return ''

    singular = p.singular_noun(phrase)
    if (singular):
        return singular 
    else:
        return phrase 
    
def singular_by_word(text):
    return ' '.join([singular_phrase(word) for word in text.split()])

def split_many(string, delimiters):
    string = basic_normalize(string)
    parts = [string]
    for d in delimiters:
        parts = sum((p.split(d) for p in parts), [])

    splits = [part.strip().lower() for i, part in enumerate(parts) if part.strip() != '' and (i == 0 or specific_entity_check(part))]
                                                                                              # len(part.split()) > 1)]
    
    return splits 

def remove_prefix(s, prefix, paraid=False):
    if paraid:
        return s[len(prefix) + 2:].strip() if s.startswith(prefix) else s.strip()  # prefix = '[P', but would like to match '[Pi] ' or '[p]'
    else:
        return s[len(prefix):].strip() if s.startswith(prefix) else s.strip()

def similar_in_length(p1, p2, limit=4): 
    p1_w_len = len(p1.split())
    p2_w_len = len(p2.split())
    w_len_ratio = float(max(p1_w_len, p2_w_len) / min(p1_w_len, p2_w_len)) 
    if w_len_ratio <= limit: 
        return True
    else:
        return False

    
    
def inclusion_idxs(query, choices):
    if (utils.full_process(query) and len(query) > 1 and choices != []):  # only exectute when query is valid. To avoid WARNING:root:Applied processor reduces input query to empty string, all comparisons will have score 0. 
        partial_inclusion_choices = [choices.index(simi_phrase) for (simi_phrase, similarity) in process.extractBests(query, choices, processor=None, scorer=fuzz.partial_ratio) if similarity== 100 and simi_phrase!= '']    

        token_set_inclusion_choices = [choices.index(simi_phrase) for (simi_phrase, similarity) in process.extractBests(query, choices, scorer=fuzz.token_set_ratio) if similarity== 100 and simi_phrase!= '']  

        inclusion_choice_idxs = list(set(partial_inclusion_choices) | set(token_set_inclusion_choices))
        return inclusion_choice_idxs
    return []

def inclusions(query, choices): # inclusion是双向的: 包含或者被包含
    inclusion_choices = []
    query = normalize_text(query)
    singular_query = normalize_text(singular_by_word(query))
    normalized_choices = [normalize_text(choice) for choice in choices]
    singular_choices = [normalize_text(singular_by_word(choice)) for choice in choices]
    if(len(query.split()) == 1):   # avoid match 'phrase' with 'ph', still allow match ‘[P1] businesswoman’ to ‘woman’ 
        choices =  [choice for singular_choice, choice in zip(singular_choices, choices) if len(singular_choice.split()) > 1 or singular_choice == singular_query] 
        normalized_choices = [normalized_choice for singular_choice, normalized_choice in zip(singular_choices, normalized_choices) if len(singular_choice.split()) > 1 or singular_choice == singular_query] 
        singular_choices = [singular_choice for singular_choice in singular_choices if len(singular_choice.split()) > 1 or singular_choice == singular_query] 
        
    
    inclusion_choice_idxs = inclusion_idxs(query, normalized_choices)
    inclusion_choice_idxs += inclusion_idxs(singular_query, normalized_choices)
    inclusion_choice_idxs += inclusion_idxs(query, singular_choices)
    inclusion_choice_idxs += inclusion_idxs(singular_query, singular_choices)

    return [choices[inclusion_choice_idx] for inclusion_choice_idx in set(inclusion_choice_idxs)]
 
def find_match_with_tolerance(short_text, text, tolerance=1): 
    if len(short_text) <= 1 or len(text) <= 1:
        return []
    matched_strs = []
    lower_white_fix_text = lower(white_space_fix(text))
    lower_white_fix_nopunc_text = lower(white_space_fix(remove_punc(text)))
    singular_lower_white_fix_nopunc_text = lower(white_space_fix(singular_by_word(remove_punc(text))))
    if(len(short_text.split()) == 1 or len(short_text) < 5):
        matches = find_near_matches(lower(white_space_fix(short_text)), lower_white_fix_text, max_l_dist=0)  # Guns N ' Roses  VS  Guns N Roses
        matches = [m for m in matches if m.matched !='' and (m.start==0 or (m.start > 0 and lower_white_fix_text[m.start-1].isalnum()==False)) and (m.end==len(lower_white_fix_text) or (m.end<len(lower_white_fix_text) and lower_white_fix_text[m.end].isalnum()==False)) ]  # whole word match
        k = len(matches)
        if(k == 0):        
            matches = find_near_matches(lower(white_space_fix(remove_punc(short_text))), lower_white_fix_nopunc_text, max_l_dist=0)  
            matches = [m for m in matches if m.matched !='' and (m.start==0 or (m.start > 0 and lower_white_fix_nopunc_text[m.start-1].isalnum()==False)) and (m.end==len(lower_white_fix_nopunc_text) or (m.end<len(lower_white_fix_nopunc_text) and lower_white_fix_nopunc_text[m.end].isalnum()==False)) ] # whole word match
            k = len(matches)
            if(k == 0):
                matches = find_near_matches(lower(white_space_fix(singular_by_word(remove_punc(short_text)))), singular_lower_white_fix_nopunc_text, max_l_dist=0)  
                matches = [m for m in matches if m.matched !='' and (m.start==0 or (m.start > 0 and singular_lower_white_fix_nopunc_text[m.start-1].isalnum()==False)) and (m.end==len(singular_lower_white_fix_nopunc_text) or (m.end<len(singular_lower_white_fix_nopunc_text) and singular_lower_white_fix_nopunc_text[m.end].isalnum()==False)) ]  # whole word match
                k = len(matches)
#         matched_strs = [m.matched for m in matches]  
    else:
        for t in range(0, tolerance+1):
            matches = find_near_matches(lower(white_space_fix(short_text)), lower_white_fix_text, max_l_dist=t)  # Guns N ' Roses  VS  Guns N Roses
            matches = [m for m in matches if m.matched !='' and (m.start==0 or (m.start > 0 and lower_white_fix_text[m.start-1].isalnum()==False)) and (m.end==len(lower_white_fix_text) or (m.end<len(lower_white_fix_text) and lower_white_fix_text[m.end].isalnum()==False)) ]
            k = len(matches)
            if(k > 0):
                break           
            matches = find_near_matches(lower(white_space_fix(remove_punc(short_text))), lower_white_fix_nopunc_text, max_l_dist=t)  # Guns N ' Roses  VS  Guns N Roses
            matches = [m for m in matches if m.matched !='' and (m.start==0 or (m.start > 0 and lower_white_fix_nopunc_text[m.start-1].isalnum()==False)) and (m.end==len(lower_white_fix_nopunc_text) or (m.end<len(lower_white_fix_nopunc_text) and lower_white_fix_nopunc_text[m.end].isalnum()==False)) ]
            k = len(matches)
            if(k > 0):
                break 
            matches = find_near_matches(lower(white_space_fix(singular_by_word(remove_punc(short_text)))), singular_lower_white_fix_nopunc_text, max_l_dist=t)  
            matches = [m for m in matches if m.matched !='' and (m.start==0 or (m.start > 0 and singular_lower_white_fix_nopunc_text[m.start-1].isalnum()==False)) and (m.end==len(singular_lower_white_fix_nopunc_text) or (m.end<len(singular_lower_white_fix_nopunc_text) and singular_lower_white_fix_nopunc_text[m.end].isalnum()==False)) ]
            k = len(matches)
            if(k > 0):
                break
    
    if(k > 0):
        matched_str = sorted(matches, key=lambda x: x.dist)[0].matched  # match with mini dist
        if(len(matched_str) > tolerance * 2):
            matches = find_near_matches(lower(matched_str), lower(text), max_l_dist=min(len(matched_str)-1, tolerance+2))  # tolerance for space and puncutations, but small than len(matched_str), for example: cbs
            matches = [m for m in matches if m.matched !='' and (m.start==0 or (m.start > 0 and text[m.start-1].isalnum()==False)) and (m.end==len(text) or (m.end<len(text) and text[m.end].isalnum()==False)) ]
            matches = sorted(matches, key=lambda x: x.dist)[:k]
        else:
            return []
    return matches 


def inclusion_best_match(query, choices): 
    inclusion_phrases = inclusions(query, choices)
    if(inclusion_phrases!= []):
        simi_phrase, similarity = process.extractOne(query, inclusion_phrases, scorer=fuzz.ratio) # most similar   
        if fuzz.WRatio(simi_phrase, query) >= 90: # similarity of '500 mile' and '41st international 500 mile sweepstakes' is 90
            return simi_phrase
 
    return None
 

def NN_check(word, tags = ['NN', 'NNP', 'LOC', 'RB', 'VB', 'CD']):  # 'North': LOC, 'north': NN, south': RB, 'play': VB, '1952': CD

    word = singular_phrase(word)
    if len(word.strip().split()) != 1:
        return False
    doc = nlp(word) 
    if(doc[0].tag_ in tags):  # such as 'method', 'woman'
        return True
    else:
        return False

    
def specific_entity_check(phrase, specified=[]):
# consider all the entitites, also match with title phrases
# do not consider noun_chunks which is more than an entity, for example: 'justin bieber charity work',  but consider 'New York city'

    if phrase.startswith('[P'):
        return False
    
    if phrase.lower() in specified:
        return True
    
    doc = nlp(phrase) 
    ents = doc.ents
    if len(ents) != 1:
        return False
      
    entity = doc.ents[0]   
    if entity.label_ in ['PERSON',  'ORG' ,'WORK_OF_ART', 'PRODUCT', 'GPE']:
        rest = normalize_text(phrase.replace(entity.text, ''))
        if rest == '' or NN_check(rest):
            return True
    
    return False



def phrase_cleaner(_span):   
    """Function to clean unnecessary words from the given phrase/string. (Punctuation mark, symbol, unknown, conjunction, determiner, subordinating or preposition and space)
# check pos, lower case, also return span start and end char-based position of phrase; ignore ones that non of the words' tag is nn
# For check has_noun, use word.pos_ in addition to NN_check. so that "medieval fortress" in 5a7d54165542995f4f402256 will be extracte
    """

    clean_phrase = []
    start = -1
    end = -1
    
    has_noun = False
    for word in _span: 
#         print(word, ' word.pos_: ', word.pos_)
        if word.pos_ not in ['PUNCT', 'SYM', 'X', 'CONJ', 'CCONJ', 'DET',  'SPACE', 'PRON'] and len(word.text) > 0:# 'ADJ' , 'VERB', 'ADP'.    'of' in University of Louisville is ADP
            if word.text.strip() != 'xxx' : 
                clean_phrase.append(word.text)
                if start == -1:
                    start = word.idx
                end = word.idx + len(word)
            if word.text.strip() == 'xxx' or word.pos_ == "NOUN" or word.pos_ == "PROPN" or NN_check(word.text.lower()):
                has_noun = True 
    if has_noun == False:
        return None, start, end
    
    if len(clean_phrase) > 0:        
        if(any([phrase.lower() not in STOP_WORDS and singular_phrase(phrase).lower() not in STOP_WORDS for phrase in clean_phrase])):        # not all STOP_WORD   
            clean_phrase_text = ' '.join(clean_phrase)
            if (len(clean_phrase_text) > 1):    # NS becomes n after singular 
                return clean_phrase_text, start, end

    return None, start, end
 

def replace_phrases_in_text(text, phrases):
     # do not macth 'civil war' when the text contains 'russian civil war' 
    for p1, p2 in list(itertools.combinations(phrases.copy(), 2)):
        if p1 != p2 and p2 in phrases and find_match_with_tolerance(p1, text, 0) and p2 in p1:  # phrase 'smyrna georgia' VS text 'Smyrna, Georgia'
            phrases.remove(p2)
        elif p1 != p2 and p1 in phrases and find_match_with_tolerance(p2, text, 0) and p1 in p2:
            phrases.remove(p1)
         
         
    # replace match phrase with ' xxx '
    # would not always replaced:  "song by Guns N ' Roses" would not be replace when match title 'guns n roses song' 
    for phrase in phrases.copy():   
        if len(phrase) > 2:  # avoid case such as phrase = 'V', which would be recognized in doc.ents
            # max_l_dist=1: match “Arthur ’s Magazine” VS the title “Arthur’s Magazine”
            matches = find_match_with_tolerance(phrase, text) 
            matched_phrases = [text[m.start:m.end] for m in matches if len(m.matched) > 2] 
            for matched_phrase in matched_phrases:
                if similar_in_length(matched_phrase, phrase, 4) == False and phrase in phrases:
                    phrases.remove(phrase)
                    break
                text = text.replace(matched_phrase, " xxx ")
    return text, phrases

# annotate bridge phrases 

def annotate_bridge_phrases(sp_phrases, sp_titles, sp_titles_phrases):
     
    bridge_phrases = []
    for ((sp1_id, sp1_phrases),(sp2_id, sp2_phrases)) in itertools.combinations(enumerate(sp_phrases), 2): # every pair of two sp sents
        overlap_phrases = set(sp1_phrases) & set(sp2_phrases)  # exact match
        bridge_phrases.extend(overlap_phrases)
        
        for phrase in sp1_phrases:
            inclusive_phrases = inclusions(phrase, sp2_phrases)  # inclusion 是双向的
            inclusive_phrases = set([inclusive_phrase for inclusive_phrase in inclusive_phrases if similar_in_length(phrase, inclusive_phrase) and inclusive_phrase])
            bridge_phrases.extend(inclusive_phrases)
            
            if(sp_titles[sp1_id] != sp_titles[sp2_id]):             # sp1 and sp2 not from same para
                if(phrase in sp_titles_phrases[sp2_id]):            # phrase is another sp para's title phrase
                    bridge_phrases.append(phrase) 
        
        for phrase in sp2_phrases:
            if(sp_titles[sp1_id] != sp_titles[sp2_id]):             # sp1 and sp2 not from same para
                if(phrase in sp_titles_phrases[sp1_id]):            # phrase is another sp para's title phrase
                    bridge_phrases.append(phrase)            

    return set(bridge_phrases)


# phrase_postpreocess 

def phrase_postpreocess(phrases, cased_text, specific_phrases):
    
    phrases = list(phrases)
    # restore the case in the original context
    for i, phrase in enumerate(phrases):
        matches = find_match_with_tolerance(phrase, cased_text, tolerance=0)
        if(len(matches) > 0):
            phrases[i] = cased_text[matches[0].start:matches[0].end]    
    
    cased_phrases = phrases.copy()
    
    if(len(specific_phrases) > 0):
        for phrase in phrases.copy():
            if specific_entity_check(phrase, specific_phrases):
                continue
            else:
                matched = inclusion_best_match(phrase.lower(), specific_phrases)   # 'will grace'  in 'will &amp; grace' ; 'christopher jonathan williams' VS Chris Williams 
                if matched and similar_in_length(phrase, matched) :
                    continue
                phrases.remove(phrase)
            
    return cased_phrases, phrases
 