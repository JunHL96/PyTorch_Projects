import random

def random_deletion(tokens, deletion_prob=0.1):
    """
    Randomly delete tokens with a given probability.
    If all tokens are deleted, returns the original list.
    """
    if not tokens:
        return tokens
    new_tokens = [token for token in tokens if random.random() > deletion_prob]
    return new_tokens if new_tokens else tokens

def augment_text(tokens, deletion_prob=0.1):
    """
    Augment a list of tokens using random deletion.
    """
    return random_deletion(tokens, deletion_prob)
