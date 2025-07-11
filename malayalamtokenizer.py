class MalayalamTokenizer:
    """
    Class for encoding and decoding labels
    """
    BLANK = "[B]"
    EOS = "[E]"
    PAD = "[P]"
    def __init__(self, threshold:float= 0.5, max_grps:int= 25):
        self.svar = ['അ', 'ആ', 'ഇ', 'ഈ', 'ഉ', 'ഊ', 'ഋ', 'എ', 'ഏ', 'ഐ',
                    'ഒ', 'ഓ', 'ഔ']  # 'ൠ' has not been added as it has not been used in recent malayalam
        self.vyanjan = ['ക', 'ഖ', 'ഗ', 'ഘ', 'ങ',
                        'ച', 'ഛ', 'ജ', 'ഝ', 'ഞ',
                        'ട', 'ഠ', 'ഡ', 'ഢ', 'ണ',
                        'ത', 'ഥ', 'ദ', 'ധ', 'ന',
                        'പ', 'ഫ', 'ബ', 'ഭ', 'മ',
                        'യ', 'ര', 'ല', 'വ', 'ശ',
                        'ഷ', 'സ', 'ഹ', 'ള', 'ഴ', 'റ']
        self.matras = ['ാ', 'ി', 'ീ', 'ു', 'ൂ', 'ൃ', 'ൈ', 'ൗ', 'െ', 'േ', 'ം', 'ഃ', '഻', 'ു്']
        # halanth is also referred as chandrakala in malayalam
        self.halanth = '്'
        self.chinh =  ['₹', '।', '!', '$', '%', '?', '.', ',', "-", '(', ')']
        self.ank = ['൦', '൧', '൨', '൩', '൪', '൫', '൬', '൭', '൮', '൯']  # numbers

        self.chillaksharam = ['ൺ', 'ൻ', 'ർ', 'ൽ', 'ൾ']
        self.special_matra = ['ം', 'ഃ']
        self._normalize_charset()
        self.threshold = threshold
        self.max_grps = max_grps
        self.h_c_classes = [MalayalamTokenizer.EOS, MalayalamTokenizer.PAD, MalayalamTokenizer.BLANK] \
                            + self.vyanjan
        self.f_c_classes = [MalayalamTokenizer.EOS, MalayalamTokenizer.PAD, MalayalamTokenizer.BLANK] \
                            + self.vyanjan + self.svar + self.ank + self.chinh + self.chillaksharam
        self.d_classes =  [MalayalamTokenizer.EOS, MalayalamTokenizer.PAD] + self.matras # binary classification
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
        self.matras = [unicodedata.normalize("NFKD", char) for char in self.matras]
        self.chillaksharam = [unicodedata.normalize("NFKD", char) for char in self.chillaksharam]
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
        for Malayalam, each group should contain:
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
                    f"{grp} in label:{label} group should not end with Chandrakala if its not the last group in the word/label or overflow in the num: of half_characters in a group")
                return False
            # allowed character category counts
            h_c_count, f_c_count, d_c_count = 3, 1, 3
            d_seen = []
            i = 0
            while i < len(grp):
                # if chillaksharam is present then it should be the only character in the group
                if grp[i] in self.chillaksharam:
                    if len(grp) != 1:
                        print(
                            f"chill is not the ending in the group {grp} for label {label}")
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
                        d_seen.append(grp[i])
                    elif grp[i] in self.rev_d_label_map and d_c_count > 0 and grp[i] in d_seen:
                        print(
                            f"Duplicate Diacritic in group {grp} for label {label}")
                        return False
                    elif grp[i] == self.halanth and grp[i-1] == unicodedata.normalize("NFKD", 'ു'):
                        i += 1
                    else:
                        if grp[i] not in self.rev_h_c_label_map and \
                                grp[i] not in self.rev_d_label_map and grp[i] not in self.rev_f_c_label_map:
                            print(f"Invalid {grp[i]} in group {grp} for label {label}")
                        if (h_c_count, f_c_count, d_c_count) == (3, 1, 3):
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
        Transform malayalam labels into groups
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
            if label[idx] in self.chillaksharam:
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
                if self._check_f_c(label, idx) and label[idx] not in self.chillaksharam:
                    if label[idx] not in (self.ank + self.chinh):
                        f_c_flag = True
                    # checks for 1 full character
                    running_grp += label[idx]
                    idx += 1
                # diacritics need not be always present
                if self._check_d_c(label, idx) and f_c_flag == True:
                    # checks for diacritics
                    running_grp += label[idx]
                    idx += 1
                    # 'ു്' has been checked for,because of its presence in older datasets instead of just 'ു'
                    if idx < len(label) and label[idx-1] == unicodedata.normalize("NFKD", 'ു') and label[idx] == self.halanth:
                        running_grp += (label[idx])
                        idx += 1
                    
                    # the following if-condition is just to make sure we identify 'െെ', 'ൌ', 'ൊ', 'ോ'
                    elif idx < len(label):
                        if label[idx-1] == unicodedata.normalize("NFKD", 'െ'):
                            if label[idx] in [unicodedata.normalize("NFKD", 'െ'), unicodedata.normalize("NFKD", 'ൗ'), unicodedata.normalize("NFKD", 'ാ')]:
                                running_grp += (label[idx])
                                idx += 1
                            
                        elif label[idx-1] == unicodedata.normalize("NFKD", 'േ') and label[idx] == unicodedata.normalize("NFKD", 'ാ'):
                            running_grp += (label[idx])
                            idx += 1
                        elif label[idx-1] == unicodedata.normalize("NFKD",'ാ' ):
                             if label[idx] in [unicodedata.normalize("NFKD", 'െ'), unicodedata.normalize("NFKD", 'േ')]:
                                running_grp += (label[idx])
                                idx += 1
                        elif label[idx-1] == unicodedata.normalize("NFKD",'ൗ' ):
                             if label[idx] == unicodedata.normalize("NFKD", 'െ'):
                                running_grp += (label[idx])
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
    
    def grp_class_encoder(self, grp: str)-> Tuple[int, int, int, Tensor]:
        """
        Encodes the group into class labels for Half-character 2, 1 and full character;
        One-Hot encodes diacritics.
        Args:
        - grp (str): Character group

        Returns:
        - tuple(Tensor, Tensor, Tensor, Tensor): class label for half-character 2,
                                                half-character 1, full-character,
                                                and diacritics (one hot encoded)
        """
        # get the character groupings
        h_c_3_target, h_c_2_target, h_c_1_target, f_c_target, d_target = (
                                                            self.blank_id,
                                                            self.blank_id,
                                                            self.blank_id,
                                                            self.blank_id,
                                                            torch.tensor(
                                                                [0 for _ in range(len(self.d_classes))],
                                                                  dtype= torch.long
                                                                )
                                                            )
        halanth_cntr = 0
        i = 0
        while i < len(grp):
            if self._check_h_c(grp, i) and halanth_cntr < 3:
                # for half character occurence
                if h_c_1_target == self.blank_id:
                    # half_character1 will always track the half-character
                    # which is joined with the complete character
                    h_c_1_target = self.rev_h_c_label_map[grp[i]]
                    
                elif h_c_2_target == self.blank_id: # there are more than 1 half-characters
                    # half_character2 will keep track of half-character 
                    # not joined with the complete character
                    h_c_2_target = h_c_1_target
                    # the first half-char will be joined to the complete character
                    h_c_1_target = self.rev_h_c_label_map[grp[i]]
                else:
                    # for the third half-char
                    h_c_3_target = h_c_2_target
                    h_c_2_target = h_c_1_target
                    h_c_1_target = self.rev_h_c_label_map[grp[i]]

            elif self._check_f_c(grp, i):
                assert f_c_target == self.blank_id,\
                       f"2 full Characters have occured {grp}-{grp[i]}"
                f_c_target = self.rev_f_c_label_map[grp[i]]
            
            elif i + 1 < len(grp) and (grp[i] == unicodedata.normalize("NFKD", 'െ') and grp[i+1] == unicodedata.normalize("NFKD", 'െ')):
                # cannot combine the above matra 'െെ' as it is a repitition of the same െ
                assert d_target[self.rev_d_label_map[unicodedata.normalize("NFKD",'ൈ')]] == 0, \
                    f"2 same matras occured {grp}-{grp[i]}-{d_target}"
                d_target[self.rev_d_label_map['ൈ']] = 1.
                i += 1

            elif i + 1 < len(grp) and (grp[i] == unicodedata.normalize("NFKD", 'ു') and grp[i+1] == self.halanth):
                assert d_target[self.rev_d_label_map[grp[i] + grp[i + 1]]] == 0, \
                    f"2 same matras occured {grp}-{grp[i]}-{d_target}"
                d_target[self.rev_d_label_map[grp[i] + grp[i + 1]]] = 1.
                i += 1 # skip the next iteration as it is part of matra
                
            elif self._check_d_c(grp, i):
                # for diacritic occurence
                assert d_target[self.rev_d_label_map[grp[i]]] == 0, \
                    f"2 same matras occured {grp}-{grp[i]}-{d_target}"
                
                d_target[self.rev_d_label_map[grp[i]]] = 1.

            elif grp[i] == self.halanth and halanth_cntr < 3 and grp[i-1] != unicodedata.normalize("NFKD", 'ു'):
                halanth_cntr += 1

            elif grp[i] == self.halanth and halanth_cntr >=3 :
                raise Exception(f"More than 3 half-grp[i]acters occured: {grp} at index {i}")

            elif grp[i] not in self.rev_h_c_label_map and grp[i] not in self.rev_f_c_label_map \
                and grp[i] not in self.rev_d_label_map and grp[i] != self.halanth:
                raise Exception(f"grp[i]acter {grp[i]} not found in vocabulary for grp {grp} at index {i}")

            else:
                raise Exception(f"Invalid group {grp} due to grp[i] at {i}- {grp[i]}")
            
            i += 1
            
        assert torch.sum(d_target, dim = -1) <= 3, f"More than 2 diacritics occured {grp}-{d_target}"

        return h_c_3_target, h_c_2_target, h_c_1_target, f_c_target, d_target

    def label_encoder(self, label:str, device:torch.device)-> Tuple[Tensor, Tensor, Tensor, Tensor, int]:
        """
        Converts the text label into classes indexes for classification
        Args:
        - label (str): The label to be encoded
        - device (torch.device): Device in which the encodings should be saved

        Returns:
        - tuple(int, int, int, Tensor): half-char 2 class index, half-char 1
                                        class index, full char class index,
                                        diacritic one hot encoding
        """
        grps = self.label_transform(label= label)
        h_c_3_target, h_c_2_target, h_c_1_target, f_c_target, d_target = (
                                                            torch.full((self.max_grps,), self.pad_id, dtype= torch.long),
                                                            torch.full((self.max_grps,), self.pad_id, dtype= torch.long), 
                                                            torch.full((self.max_grps,), self.pad_id, dtype= torch.long),
                                                            torch.full((self.max_grps,), self.pad_id, dtype= torch.long),
                                                            torch.zeros(self.max_grps, len(self.d_classes), dtype= torch.long)
                                                        )
        d_target[:,self.pad_id] = 1.
        # truncate grps if grps exceed max_grps
        if len(grps) < self.max_grps:
            eos_idx = len(grps)
            # assign eos after the last group
            h_c_3_target[eos_idx], h_c_2_target[eos_idx], h_c_1_target[eos_idx], f_c_target[eos_idx] = self.eos_id, self.eos_id, self.eos_id, self.eos_id
            d_target[eos_idx, self.eos_id] = 1
                                                                                        
        else:
            grps = grps[:self.max_grps]
       
        for idx,grp in enumerate(grps, start= 0):
            h_c_3_target[idx], h_c_2_target[idx], h_c_1_target[idx], f_c_target[idx], d_target[idx] = self.grp_class_encoder(grp=grp)
        return h_c_3_target.to(device), h_c_2_target.to(device), h_c_1_target.to(device), f_c_target.to(device), d_target.to(device), len(grps)
    
    def _decode_grp(self, h_c_3_pred:Tensor, h_c_2_pred:Tensor, h_c_1_pred:Tensor, f_c_pred:Tensor, 
                    d_pred:Tensor, d_max:Tensor)-> str:
        """
        Method which takes in class predictions of a single group and decodes
        the group
        Args:
        - h_c_3_pred (Tensor): Index of max logit (prediction) of half-char 3; shape torch.Size(1)
        - h_c_2_pred (Tensor): Index of max logit (prediction) of half-char 2; shape torch.Size(1)
        - h_c_1_pred (Tensor): Index of max logit (prediction) of half-char 1; shape torch.Size(1)
        - f_c_pred (Tensor): Index of max logit (prediction) of full-char; shape torch.Size(1)
        - d_pred (Tensor): Index of 2 probability values > threshold; shape torch.Size(2)
        - d_max (Tensor): The corresponding values of d_pred in binary mask

        Returns:
        - str: the group formed
        """
        assert len(d_pred) == len(d_max) == 3, \
            "Diacritic preds and max must contain 2 elements for 2 diacritics"
        
        grp = ""
        grp += self.h_c_label_map[int(h_c_3_pred.item())] + self.halanth \
                if self.h_c_label_map[int(h_c_3_pred.item())] != MalayalamTokenizer.BLANK \
                    and self.h_c_label_map[int(h_c_2_pred.item())] != MalayalamTokenizer.PAD else ""
        grp += self.h_c_label_map[int(h_c_2_pred.item())] + self.halanth \
                if self.h_c_label_map[int(h_c_2_pred.item())] != MalayalamTokenizer.BLANK \
                    and self.h_c_label_map[int(h_c_2_pred.item())] != MalayalamTokenizer.PAD else ""
        
        grp += self.h_c_label_map[int(h_c_1_pred.item())] + self.halanth \
                if self.h_c_label_map[int(h_c_1_pred.item())] != MalayalamTokenizer.BLANK \
                    and self.h_c_label_map[int(h_c_1_pred.item())] != MalayalamTokenizer.PAD else ""
        
        grp += self.f_c_label_map[int(f_c_pred.item())]
        d1 = self.d_c_label_map[int(d_pred[0].item())] if d_max[0] else ""
        d2 = self.d_c_label_map[int(d_pred[1].item())] if d_max[1] else ""
        d3 = self.d_c_label_map[int(d_pred[2].item())] if d_max[2] else ""
        # ['ാ', 'ി', 'ീ', 'ു', 'ൂ', 'ൃ', 'ൈ', 'ൗ', 'െ', 'േ', '഻', 'ു്'] 'ം', 'ഃ'
        if d1 in unicodedata.normalize("NFKD",  'ം' + 'ഃ'):
            if d2 in unicodedata.normalize("NFKD", 'ാ') and d3 in unicodedata.normalize("NFKD", 'െ'+ 'േ'):
                grp += d3 + d2 + d1
            elif d3 in unicodedata.normalize("NFKD", 'ാ') and d2 in unicodedata.normalize("NFKD", 'െ'+ 'േ'):
                grp += d2 + d3 + d1
            elif d2 == 'െ' and d3 == 'ൗ':
                grp += d2 + d3 + d1
            elif d3 == 'െ' and d2 == 'ൗ':
                grp += d3 + d2 + d1
            elif d3 == '' or d2 == '':
                grp += d2 + d3 + d1
            else:
                grp += d1
                    
        elif d2 in unicodedata.normalize("NFKD",  'ം' + 'ഃ'):
            if d1 in unicodedata.normalize("NFKD", 'ാ') and d3 in unicodedata.normalize("NFKD", 'െ'+ 'േ'):
                grp += d3 + d1 + d2
            elif d3 in unicodedata.normalize("NFKD", 'ാ') and d1 in unicodedata.normalize("NFKD", 'െ'+ 'േ'):
                grp += d1 + d3 + d2
            elif d1 == 'െ' and d3 == 'ൗ':
                grp += d1 + d3 + d2
            elif d3 == 'െ' and d1 == 'ൗ':
                grp += d3 + d1 + d2
            elif d3 == '' or d1 == '':
                grp += d1 + d3 + d2
            else:
                grp += d1

        elif d3 in unicodedata.normalize("NFKD",  'ം' + 'ഃ'):
            if d1 in unicodedata.normalize("NFKD", 'ാ') and d2 in unicodedata.normalize("NFKD", 'െ'+ 'േ'):
                grp += d2 + d1 + d3
            elif d2 in unicodedata.normalize("NFKD", 'ാ') and d1 in unicodedata.normalize("NFKD", 'െ'+ 'േ'):
                grp += d1 + d2 + d3
            elif d1 == 'െ' and d2 == 'ൗ':
                grp += d1 + d2 + d3
            elif d2 == 'െ' and d1 == 'ൗ':
                grp += d2 + d1 + d3
            elif d2 == '' or d1 == '':
                grp += d1 + d2 + d3
            else:
                grp += d1
        
        elif d1 == '':
            if d2 in unicodedata.normalize("NFKD", 'ാ') and d3 in unicodedata.normalize("NFKD", 'െ'+ 'േ'):
                grp += d3 + d2
            elif d3 in unicodedata.normalize("NFKD", 'ാ') and d2 in unicodedata.normalize("NFKD", 'െ'+ 'േ'):
                grp += d2 + d3 # handle the null case
            elif d2 == 'െ' and d3 == 'ൗ':
                grp += d2 + d3
            elif d3 == 'െ' and d2 == 'ൗ':
                grp += d3 + d2
            elif d3 == '' or d2 == '':
                grp += d2 + d3
            else:
                grp += d2

        elif d2 == '':
            if d1 in unicodedata.normalize("NFKD", 'ാ') and d3 in unicodedata.normalize("NFKD", 'െ'+ 'േ'):
                grp += d3 + d1
            elif d3 in unicodedata.normalize("NFKD", 'ാ') and d1 in unicodedata.normalize("NFKD", 'െ'+ 'േ'):
                grp += d1 + d3 # handle the null case
            elif d1 == 'െ' and d3 == 'ൗ':
                grp += d2 + d3
            elif d3 == 'െ' and d1 == 'ൗ':
                grp += d3 + d1
            elif d3 == '' or d1 == '':
                grp += d1 + d3
            else:
                grp += d1
        elif d3 == '':
            if d2 in unicodedata.normalize("NFKD", 'ാ') and d1 in unicodedata.normalize("NFKD", 'െ'+ 'േ'):
                grp += d1 + d2
            elif d1 in unicodedata.normalize("NFKD", 'ാ') and d2 in unicodedata.normalize("NFKD", 'െ'+ 'േ'):
                grp += d2 + d1 # handle the null case
            elif d2 == 'െ' and d1 == 'ൗ':
                grp += d2 + d1
            elif d1 == 'െ' and d2 == 'ൗ':
                grp += d1 + d2
            elif d1 == '' or d2 == '':
                grp += d2 + d1
            else:
                grp += d2
        else:
            grp += d1
                
        return grp.replace(MalayalamTokenizer.BLANK, "").replace(MalayalamTokenizer.PAD, "") # remove all [B], [P] occurences
                
    def decode(self, logits:Tuple[Tensor, Tensor, Tensor, Tensor, Tensor])-> tuple:
        """
        Method to decode the labels of a batch given the logits
        Args:-
        - logits (tuple(Tensor, Tensor, Tensor, Tensor)): the logits of the model in the order
                                                        half-char 3, half-char 2, half-char 1, full-char, diacritic
                                                        logits
        Returns:
        - tuple: the labels of each batch item
        """
        # logits shape: BS x Max Grps x # of classes
        (h_c_3_logits, h_c_2_logits, h_c_1_logits, f_c_logits, d_logits) = logits
        batch_size = h_c_2_logits.shape[0]

        # greedy decoding for half and full chars
        h_c_3_preds = torch.argmax(h_c_3_logits, dim= 2)
        h_c_2_preds = torch.argmax(h_c_2_logits, dim= 2)
        h_c_1_preds = torch.argmax(h_c_1_logits, dim= 2)
        f_c_preds = torch.argmax(f_c_logits, dim= 2)

        # threshold filter for diacritic
        d_bin_mask = nn.functional.sigmoid(d_logits) > self.threshold
        d_max, d_preds = torch.topk(d_bin_mask.int(), k= 3, dim= 2, largest= True) # top k requires scalar

        # get the predictions
        pred_labels = []    
        for i in range(batch_size):
            label = ""
            for j in range(self.max_grps):
                grp = self._decode_grp(
                                h_c_3_pred= h_c_3_preds[i,j],
                                h_c_2_pred= h_c_2_preds[i,j],
                                h_c_1_pred= h_c_1_preds[i,j],
                                f_c_pred= f_c_preds[i,j],
                                d_pred= d_preds[i,j],
                                d_max= d_max[i,j]
                            )
                if MalayalamTokenizer.EOS in grp:
                    break
                else:
                    label += grp
                    
            pred_labels.append(label)
            label = ""
        
        return tuple(pred_labels)
