## ENGLISH SUFFIXES
# Reference https://www.myenglishteacher.eu/blog/prefixes-suffixes-list/

suffixes = []

# NOUN SUFFIXES
suffixes.extend([
    "acy",                      # state or quality
    "al",                       # the action or process of
    "ance",                     # state or quality of
    "ence",                     # state or quality of
    "dom",                      # place or state of being
    "er",                       # person or object that does a specified action
    "or",                       # person or object that does a specified action
    "ism",                      # doctrine, belief
    "ist",                      # person or object that does a specified action
    "ity",                      # quality of
    "ty",                       # quality of
    "ment",                     # condition
    "ness",                     # state of being
    "ship",                     # position held
    "sion",                     # state of being
    "tion"                      # state of being
])

# VERB SUFFIXES
suffixes.extend([
    "ate",                      # become
    "en",                       # become
    "ify",                      # make or become
    "fy",                       # make or become
    "ise",                      # become
    "ize"                       # become
])

# ADJECTIVE SUFFIXES
suffixes.extend([
    "able",                     # capable of being
    "ible",                     # capable of being
    "al",                       # having the form or character of
    "esque",                    # in a manner of or resembling
    "ful",                      # notable for
    "ic",                       # having the form or character of
    "ical",                     # having the form or character of
    "ious",                     # characterised by
    "ous",                      # characterised by
    "ish",                      # having the quality of
    "ive",                      # having the nature of
    "less",                     # without
    "y"                         # characterised by
])

# ADVERB SUFFIXES
suffixes.extend([
    "ly",                       # related to or quality
    "ward",                     # direction
    "wards",                    # direction
    "wise"                      # in relation to
])

## ENGLISH PREFIXES
# Reference https://dictionary.cambridge.org/grammar/british-grammar/word-formation/prefixes

prefixes = [
    "anti",                     # against/opposed to
    "auto",                     # self
    "de",                       # reverse or change
    "dis",                      # reverse or remove
    "down",                     # reduce or lower
    "extra",                    # beyond
    "hyper",                    # extreme
    "il",                       # not
    "im",                       # not
    "in",                       # not
    "ir",                       # not
    "inter",                    # between
    "mega",                     # very big, important
    "mid",                      # middle
    "mis",                      # incorrectly, badly
    "non",                      # not
    "over",                     # too much
    "out",                      # go beyond
    "post",                     # after
    "pre",                      # before
    "pro",                      # in favour of
    "re",                       # again
    "semi",                     # half
    "sub",                      # under, below
    "super",                    # above, beyond
    "tele",                     # at a distance
    "trans",                    # across
    "ultra",                    # extremely
    "un",                       # remove, reverse, not
    "under",                    # less than, beneath
    "up"                        # make or move higher
]

def has_first_letter_captial(word):
    if len(word) > 0:
        return 1 if (word[0] >= 'A' and word[0] <= 'Z') else 0
    return 0

def has_hyphen(word):
    return 1 if ('-' in word) else 0

def starts_with_number(word):
    if len(word) > 0:
        return 1 if (word[0] >= '0' and word[0] <= '9') else 0
    return 0

## get the orthographic features for a word
def get_orth_feats(word):
    feats = []
    feats.append(has_first_letter_captial(word))
    feats.append(has_hyphen(word))
    feats.append(starts_with_number(word))
    lowercase_word = word.lower()
    feats.extend([1 if lowercase_word.endswith(s) else 0 for s in suffixes])
    # feats.extend([1 if lowercase_word.startswith(p) else 0 for p in prefixes])
    return feats

def get_num_orth_feats():
    return len(get_orth_feats("foobar"))

