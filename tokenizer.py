# tokenizer.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to ShanghaiTech University, including a link 
# to https://i-techx.github.io/iTechX/courses?course_code=CS274A
# 
# Attribution Information: The NLP projects were developed at ShanghaiTech University.
# The core projects and autograders were adapted by Haoyi Wu (wuhy1@shanghaitech.edu.cn)


from typing import Dict, Tuple, List
import util

from tokenizers import Tokenizer
import tokenizers.models
import tokenizers.pre_tokenizers
import tokenizers.decoders
import string


def get_gpt2_tokenizer() -> Tokenizer:
    """
    Return a GPT-2 tokenizer.
    """
    vocab, merges = tokenizers.models.BPE.read_file("data/vocab.json", "data/merges.txt")
    clean_vocab(vocab, merges)
    tokenizer = Tokenizer(tokenizers.models.BPE(vocab, merges))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space = False)
    tokenizer.decoder = tokenizers.decoders.ByteLevel()

    return tokenizer


def clean_vocab(vocab: Dict[str, int], merges: List[Tuple[str, str]]):
    """
    Question:
        Given the vocabulary and merges of a BPE tokenizer, clean them up to avoid subtokens
        that consist of multiple digits. This would reduce the sparsity problem.

        This function does in-place modifications, so it should not return anything.

    Example:
        >>> vocab = {'Ġ': 0, '1': 1, '2': 2, 'Ġ1': 3, 'Ġ2': 4, '12': 5, 'Ġ12': 6}
        >>> merges = [('Ġ', '1'), ('Ġ', '2'), ('1', '2'), ('Ġ1', '2')]
        >>> clean_vocab(vocab, merges)
        >>> vocab
        {'Ġ': 0, '1': 1, '2': 2, 'Ġ1': 3, 'Ġ2': 4}

    Args:
        vocab (:obj:`Dict[str, int]`):
            A dictionnary of string keys and their ids, e.g.`{"am": 0,...}`

        merges (:obj:`List[Tuple[str, str]]`):
            A list of pairs of tokens (:obj:`Tuple[str, str]`), e.g. `[("a", "b"),...]`
    """
    
    """YOUR CODE HERE"""
    def is_multi_digit_subtoken(token_str: str) -> bool:
        """
        Checks if a token string, after stripping a potential leading 'Ġ' or '臓',
        consists solely of multiple digits.
        """
        effective_token = token_str
        if token_str.startswith('Ġ') or token_str.startswith('臓'):
            effective_token = token_str[1:]
        
        # Check if the effective token is all digits and has more than 1 digit
        return effective_token.isdigit() and len(effective_token) > 1

    # Remove multi-digit tokens from vocab while preserving index order
    tokens_to_remove = [token for token in vocab if is_multi_digit_subtoken(token)]
    for token in tokens_to_remove:
        del vocab[token]
    
    # Reindex vocabulary to maintain sequential indices
    sorted_items = sorted(vocab.items(), key=lambda x: x[1])
    vocab.clear()
    for i, (token, _) in enumerate(sorted_items):
        vocab[token] = i

    # Remove merge rules that would create multi-digit tokens
    # Keep merges where both components exist in vocab and result doesn't create multi-digit
    valid_merges = []
    for m1, m2 in merges:
        merged_token = m1 + m2
        # Keep the merge if:
        # 1. Both components still exist in vocab
        # 2. The result doesn't create a multi-digit token
        if m1 in vocab and m2 in vocab and not is_multi_digit_subtoken(merged_token):
            valid_merges.append((m1, m2))
    
    # Update merges in-place
    merges[:] = valid_merges


if __name__ == '__main__':

    print("Running tokenizer.py ...")

    tokenizer = get_gpt2_tokenizer()

    sentence = "Is 1029310928407 a multiple of 3?"
    print("      Sentence:", sentence)
    output = tokenizer.encode(sentence)
    print("After encoding:", output.tokens)
