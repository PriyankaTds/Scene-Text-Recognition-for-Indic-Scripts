import torch.nn as nn
import torch
import unicodedata

from typing import Tuple
from torch import Tensor


class GujaratiTokenizer:
    BLANK = "[B]"
    EOS = "[E]"
    PAD = "[P]"

    def __init__(self, threshold:float= 0.5, max_grps:int= 25):
        self.vyanjan = ['ક', 'ખ', 'ગ', 'ઘ', 'ઙ',
                'ચ', 'છ', 'જ', 'ઝ', 'ઞ',
                'ટ', 'ઠ', 'ડ', 'ઢ', 'ણ',
                'ત', 'થ', 'દ', 'ધ', 'ન',
                'પ', 'ફ', 'બ', 'ભ', 'મ',
                'ય', 'ર', 'લ', 'ળ', 'વ',
                'શ', 'ષ', 'સ', 'હ'
                ]

        self.svar = ['અ', 'આ', 'ઇ', 'ઈ', 'ઉ',
             'ઊ', 'ઋ', 'એ', 'ઐ',
             'ઓ', 'ઔ', 'ઍ', 'ઑ'
            ]

        self.matras = ['ા', 'િ', 'ી', 'ુ', 'ૂ',
               'ૃ', 'ૄ', 'ે', 'ૈ', 'ો',
               'ૌ', 'ૅ', 'ૉ', 'ં', 'ઃ'
              ]

        self.chinh = ['ૐ', '₹', '।', '!', '$', ',', '.', '-', '%', '॥', '૰']

        self.ank = ['૦', '૧', '૨', '૩', '૪', '૫', '૬', '૭', '૮', '૯']

        self.halanth = '્'

        self.h_c_classes = [GujaratiTokenizer.EOS, GujaratiTokenizer.PAD, GujaratiTokenizer.BLANK] \
                            + self.vyanjan
        self.f_c_classes = [GujaratiTokenizer.EOS, GujaratiTokenizer.PAD] \
                            + self.vyanjan + self.svar + self.ank + self.chinh
        self.d_classes =  [GujaratiTokenizer.EOS, GujaratiTokenizer.PAD] + self.matras
        self.eos_id = 0
        self.pad_id = 1
        self.blank_id = 2
        self._normalize_charset()
        self.thresh = threshold
        self.max_grps = max_grps


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


    def _normalize_charset(self)-> None:
        """
        Function to normalize the charset provided and converts the charset from list to tuple
        """
        self.h_c_classes = tuple([unicodedata.normalize("NFKD", c) for c in self.h_c_classes])
        self.f_c_classes = tuple([unicodedata.normalize("NFKD", c) for c in self.f_c_classes])
        self.d_classes = tuple([unicodedata.normalize("NFKD", c) for c in self.d_classes])


    def get_charset(self)-> list:
        return self.h_c_classes + self.f_c_classes + self.d_classes + (self.halanth, )


    def _check_h_c(self, label:str, idx:int)-> bool:
        """
        Method to check if the character at index idx in label is a half-char or not
        Returns:
        - bool: True if the current index is a half character or not
        """
        # check if the current char is in h_c_set and next char is halanth
        return idx < len(label) and label[idx] in self.rev_h_c_label_map \
            and idx + 1 < len(label) and label[idx + 1] == self.halanth

    def _check_f_c(self, label, idx)-> bool:
        """
        Method to check if the character at index idx in label is a full-char or not
        Returns:
        - bool: True if the current idx is a full character
        """
        # check if the current char is in f_c_set and
        # it is the last char of label or the following char is not halanth
        return idx < len(label) and label[idx] in self.rev_f_c_label_map \
            and (idx + 1 >= len(label) or label[idx + 1] != self.halanth)


    def _check_d_c(self, label, idx)-> bool:
        """
        Function to check if the character at index idx in label is a diacritic or not
        Returns:
        - bool: True if the current idx is a diacritic
        """
        # check if the char belongs in d_c_set
        return idx < len(label) and label[idx] in self.rev_d_label_map


    def grp_sanity(self, label:str, grps:tuple)-> bool:
        """
        Checks whether the groups are properly formed
        for Gujarati, each group should contain:
            1) at most 2 half-characters
            2) at most 2 diacritics
            3) 1 full-character
        Args:
        - label (str): label for which groups are provided
        - gprs (tuple): tuple containing the groups of the label

        Returns:
        - bool: True if all groups pass the sanity check else raises an Exception
        """
        for grp in grps:
            # allowed character category counts
            h_c_count, f_c_count, d_c_count = 2, 1, 2
            d_seen = []
            i = 0
            while i < len(grp):
                if i + 1 < len(grp) and self.halanth == grp[i + 1] and grp[i] in self.rev_h_c_label_map and h_c_count > 0:
                    h_c_count -= 1
                    i += 1
                elif grp[i] in self.rev_f_c_label_map and f_c_count > 0:
                    f_c_count -= 1
                elif grp[i] in self.rev_d_label_map and d_c_count == 2:
                    d_c_count -= 1
                    d_seen.append(grp[i])
                elif grp[i] in self.rev_d_label_map and d_c_count != 2 and grp[i] not in d_seen:
                    d_c_count -= 1
                elif grp[i] in self.rev_d_label_map and d_c_count != 2 and grp[i] in d_seen:
                    print(f"Duplicate Diacritic in group {grp} for label {label}")
                    return False
                else:
                    if grp[i] not in self.rev_f_c_label_map and \
                        grp[i] not in self.rev_d_label_map and grp[i] not in self.rev_h_c_label_map:
                        print(f"Invalid {grp[i]} in group {grp} for label {label}")
                    else:
                        print(f"ill formed group {grp} in {label}")
                i += 1

            if f_c_count == 1 or (h_c_count, f_c_count, d_c_count) == (2, 1, 2):
                print(f"There are no full character in group {grp} for {label} OR")
                return False

        return True

    def label_transform(self, label:str)-> tuple:
        """
        Transform Gujarati labels into groups
        Args:
        - label (str): label to transform
        Returns:
        - tuple: groups of the label
        """
        grps = ()
        running_grp = ""
        idx = 0
        while(idx < len(label)):
            t = idx
            # the group starts with a half-char or a full-char
            if self._check_h_c(label, idx):
                # checks for half-characters
                running_grp += label[idx:idx+2]
                idx += 2
                # there can be 2 half-char
                if self._check_h_c(label, idx):
                    running_grp += label[idx: idx+2]
                    idx += 2

            # half-char is followed by full char
            if self._check_f_c(label, idx):
                # checks for 1 full character
                running_grp += label[idx]
                idx += 1

            # diacritics need not be always present
            if self._check_d_c(label, idx):
                # checks for diacritics
                running_grp += label[idx]
                idx += 1
                # there can be 2 diacritics in a group
                if self._check_d_c(label, idx):
                    running_grp += (label[idx])
                    idx += 1

            if t == idx:
                print(f"Invalid label {label}-{t}")
                return ()

            grps = grps + (running_grp, )
            running_grp = ""

        return grps if self.grp_sanity(label, grps) else ()
