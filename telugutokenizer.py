import torch.nn as nn
import torch
import unicodedata

from typing import Tuple
from torch import Tensor

class TeluguTokenizer:
    """
    Class for encoding and decoding labels
    """
    BLANK = "[B]"
    EOS = "[E]"
    PAD = "[P]"
    def __init__(self, threshold:float= 0.5, max_grps:int= 25):
        self.svar = ['అ', 'ఆ', 'ఇ', 'ఈ', 'ఉ', 'ఊ', 'ఋ', 'ఎ', 'ఏ', 'ఐ',
                     'ఒ', 'ఓ', 'ఔ']  # Vowels

        self.vyanjan = ['క', 'ఖ', 'గ', 'ఘ', 'ఙ',
                        'చ', 'ఛ', 'జ', 'ఝ', 'ఞ',
                        'ట', 'ఠ', 'డ', 'ఢ', 'ణ',
                        'త', 'థ', 'ద', 'ధ', 'న',
                        'ప', 'ఫ', 'బ', 'భ', 'మ',
                        'య', 'ర', 'ల', 'వ', 'శ',
                        'ష', 'స', 'హ', 'ళ', 'క్ష', 'ఱ']  # Consonants

        self.matras = ['ా', 'ి', 'ీ', 'ు', 'ూ', 'ృ', 'ె', 'ే', 'ై', 'ౖ','ొ', 'ో', 'ౌ', 'ం', 'ః',  " ై", "ఁ"]  # Matras
        self.halanth = '్'  # Halant (also known as virama)
        self.chinh =  ['₹', '।', '!', '$', '%', '?', '.', ',', "-", '(', ')']  # Punctuation
        self.ank = ['౦', '౧', '౨', '౩', '౪', '౫', '౬', '౭', '౮', '౯']  # Numbers

        self.special_consonants = ['క్ష', 'ఱ']  # Special consonants
        self.special_matra = ['ం', 'ః','ఽ']  # Special matras


        self._normalize_charset()
        self.threshold = threshold
        self.max_grps = max_grps
        self.h_c_classes = [TeluguTokenizer.EOS, TeluguTokenizer.PAD, TeluguTokenizer.BLANK] \
                            + self.vyanjan
        self.f_c_classes = [TeluguTokenizer.EOS, TeluguTokenizer.PAD, TeluguTokenizer.BLANK] \
                            + self.vyanjan + self.svar + self.ank + self.chinh + self.special_consonants
        self.d_classes =  [TeluguTokenizer.EOS, TeluguTokenizer.PAD] + self.matras # binary classification
        self.eos_id = 0
        self.pad_id = 1
        self.blank_id = 2
        # dict with class indexes as keys and characters as values
        self.h_c_label_map = {k:c for k,c in enumerate(self.h_c_classes, start = 0)}
        # 0 will be reserved for blank
        self.f_c_label_map = {k:c for k,c in enumerate(self.f_c_classes, start = 0)}
        # blank not needed for embedding as it is Binary classification of each diacritic
        self.d_c_label_map = {k:c for k,c in enumerate(self.d_classes, start = 0)}

        # dict with characters as keys and class indexes as values
        self.rev_h_c_label_map = {c:k for k,c in enumerate(self.h_c_classes, start = 0)}
        self.rev_f_c_label_map = {c:k for k,c in enumerate(self.f_c_classes, start = 0)}
        self.rev_d_label_map = {c:k for k,c in enumerate(self.d_classes, start= 0)}

    def get_charset(self)-> list:
        """
        returns the complete charset used by the tokenizer
        """
        return self.h_c_classes + self.f_c_classes + self.d_classes + [self.halanth,]

    def _normalize_charset(self)-> None:
        """
        NFKD Normalize the input charset
        """
        self.ank = [unicodedata.normalize("NFKD", char) for char in self.ank]
        self.chinh = [unicodedata.normalize("NFKD", char) for char in self.chinh]
        self.svar = [unicodedata.normalize("NFKD", char) for char in self.svar]
        self.vyanjan = [unicodedata.normalize("NFKD", char) for char in self.vyanjan]
        self.matras = [unicodedata.normalize("NFC", char) for char in self.matras]
        self.special_consonants = [unicodedata.normalize("NFKD", char) for char in self.special_consonants]
        self.special_matra = [unicodedata.normalize("NFKD", char) for char in self.special_matra]
        self.halanth = unicodedata.normalize("NFKD", self.halanth)

    def _check_h_c(self, label: str, idx: int) -> bool:
        """
        Method to check if the character at index idx in label is a half-char or not
        Returns:
        - bool: True if the current index is a half character or not
        """
        # check if the current char is in h_c_set and next char is halanth
        return idx < len(label) and label[idx] in self.rev_h_c_label_map and idx + 1 < len(label) and label[idx + 1] == self.halanth

    def _check_f_c(self, label, idx):
        """
        Method to check if the character at index idx in label is a full-char or not
        Returns:
        - bool: True if the current idx is a full character
        """
        # check if the current char is in f_c_set and
        # it is the last char of label or the following char is not halanth
        return idx < len(label) and label[idx] in self.rev_f_c_label_map \
            and (idx + 1 >= len(label) or label[idx + 1] != self.halanth)

    def _check_vyanjan(self, label, idx):
        """
        Method to check if the character at index idx in label is a full-vyanjan or not
        Returns:
        - bool: True if the current idx is a vyanjan
        """
        # check if the current char is in f_c_set and
        # it is the last char of label or the following char is not halanth
        return idx < len(label) and label[idx] in self.vyanjan \
            and (idx + 1 >= len(label) or label[idx + 1] != self.halanth)

    def _check_d_c(self, label, idx):
        """
        Function to check if the character at index idx in label is a diacritic or not
        Returns:
        - bool: True if the current idx is a diacritic
        """
        # check if the char belongs in d_c_set
        return idx < len(label) and label[idx] in self.rev_d_label_map

    def grp_sanity(self, label: str, grps: tuple) -> bool:
        """
        Checks whether the groups are properly formed
        for Telugu, each group should contain:
            1) at most 3 half-characters
            2) at most 2 diacritics
            3) at most 1 full-character
        Args:
        - label (str): label for which groups are provided
        - gprs (tuple): tuple containing the groups of the label

        Returns:
        - bool: True if all groups pass the sanity check else raises an Exception
        """
        for grp in grps:
            if grp[len(grp)-1] == self.halanth and grp in grps[:len(grps)-1]:
                print(
                    f"{grp} in label:{label} group should not end with Halanth if its not the last group in the word/label or overflow in the num: of half_characters in a group")
                return False
            # allowed character category counts
            h_c_count, f_c_count, d_c_count = 3, 1, 2
            d_seen = []
            i = 0
            while i < len(grp):
                # if special consonants is present then it should be the only character in the group
                if grp[i] in self.special_consonants:
                    if len(grp) != 1:
                        print(
                            f"spc cons is not the ending in the group {grp} for label {label}")
                        return False
                    else:
                        f_c_count -= 1
                        i += 1
                else:
                    if i + 1 < len(grp) and self.halanth == grp[i + 1] and grp[i] in self.rev_h_c_label_map and h_c_count > 0:
                        h_c_count -= 1
                        i += 1
                    elif grp[i] in self.rev_f_c_label_map and f_c_count > 0:
                        f_c_count -= 1
                    elif grp[i] in self.rev_d_label_map and d_c_count > 0 and grp[i] not in d_seen:
                        d_c_count -= 1

                    elif grp[i] in self.rev_d_label_map and d_c_count > 0 and grp[i] in d_seen:
                        print(
                            f"Duplicate Diacritic in group {grp} for label {label}")
                    elif grp[i]=='ె' and grp[i+1]=='ౖ' and d_c_count > 0 and grp[i] not in d_seen and grp[i+1] not in d_seen:
                        d_c_count -= 1


                    # elif grp[i] == self.halanth and grp[i-1] == unicodedata.normalize("NFKD", 'ു'):
                    #     i += 1
                    else:
                        if grp[i] not in self.rev_h_c_label_map and \
                                grp[i] not in self.rev_d_label_map and grp[i] not in self.rev_f_c_label_map:
                            print(f"Invalid {grp[i]} in group {grp} for label {label}")
                        if (h_c_count, f_c_count, d_c_count) == (3, 1, 2):
                            print(
                                f"Invalid number of half {h_c_count}, full {f_c_count} or diacritic characters {d_c_count} in {grp} for {label}")

                        return False
                    i += 1
            if f_c_count == 1:
                if grp[len(grp)-1:] == self.halanth:
                    return True
                print(f"There are no full character in group {grp} for {label} at {grps} OR")
                return False
        return True

    def label_transform(self, label: str) -> tuple:
        """
        Transform Telugu labels into groups
        Args:
        - label (str): label to transform
        Returns:
        - tuple: groups of the label
        """
        grps = ()
        running_grp = ""
        idx = 0
        while (idx < len(label)):
            t = idx
            # the group starts with a half-char or a full-char
            if label[idx] in self.special_consonants:
                running_grp += label[idx]
                idx += 1
            else:
                if self._check_h_c(label, idx):
                    # checks for half-characters
                    running_grp += label[idx:idx+2]
                    idx += 2
                    # there can be 3 half-char
                    if self._check_h_c(label, idx):
                        running_grp += label[idx: idx+2]
                        idx += 2
                        if self._check_h_c(label, idx):
                            running_grp += label[idx: idx+2]
                            idx += 2
                # half-char is followed by full character which is just vyanjan
                f_c_flag = False
                if self._check_f_c(label, idx) and label[idx] not in self.special_consonants:
                    if label[idx] not in (self.ank + self.chinh):
                        f_c_flag = True
                    # checks for 1 full character
                    running_grp += label[idx]
                    idx += 1
                # diacritics need not be always present
                if self._check_d_c(label, idx) and f_c_flag == True:
                  if label[idx] == 'ె' and (idx + 1) < len(label) and label[idx + 1] == 'ౖ':
                    running_grp += 'ై'
                    idx += 2
                    # checks for diacritics
                  else:
                      running_grp += label[idx]
                      idx += 1


                  # there can be 1 diacritics in a group  + ['ം' & 'ഃ' ] attached to it
                  if idx < len(label) and label[idx] in self.special_matra:
                      running_grp += (label[idx])
                      idx += 1

                if t == idx:
                    print(
                        f"{label} is Invalid label because of {label[t]} after {label[t-1]} at index {t}")
                    return ()
            if running_grp != "":
                grps = grps + (running_grp, )
            running_grp = ""
        return grps if self.grp_sanity(label, grps) else ()
