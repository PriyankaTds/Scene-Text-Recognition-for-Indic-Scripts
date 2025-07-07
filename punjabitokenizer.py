import unicodedata

class GurmukhiTokenizer:
    """
    Class for encoding and decoding Gurmukhi labels
    """
    BLANK = "[B]"
    EOS = "[E]"
    PAD = "[P]"

    def __init__(self, threshold: float = 0.5, max_grps: int = 25):
        """
        Args:
        - threshold (float): classification threshold (default: 0.5)
        """
        self.vyanjan = ['ਸ','ਹ',
                        'ਕ','ਖ','ਗ','ਘ','ਙ',
                        'ਚ','ਛ','ਜ','ਝ','ਞ',
                        'ਟ','ਠ','ਡ','ਢ','ਣ',
                        'ਤ','ਥ','ਦ','ਧ','ਨ',
                        'ਪ','ਫ','ਬ','ਭ','ਮ',
                        'ਯ','ਰ','ਲ','ਵ','ੜ']
        self.svar = [ 'ੳ', 'ਅ', 'ੲ', 'ਔ', 'ਐ', 'ਆ', 'ਈ', 'ਊ', 'ਓ', 'ਏ', 'ਇ', 'ਉ']
        self.matras = ['ਾ', 'ਿ', 'ੀ', 'ੁ', 'ੂ', 'ੇ', 'ੈ', 'ੋ', 'ੌ', 'ੰ', 'ੱ', 'ਂ', 'ਃ']
        self.chinh = ['ॐ', '₹', '।', '!', '$', ',', '.', '-', '%', '॥','ॽ','ੴ']
        self.ank = ['੦', '੧', '੨', '੩', '੪', '੫', '੬', '੭', '੮', '੯']
        self.subjoined_characters = ['ਯ','ਰ','ਹ','ਵ']
        self.special_consonants = ['ਸ਼', 'ਖ਼', 'ਗ਼', 'ਜ਼', 'ਫ਼', 'ਲ਼']
        self.halanth = '੍'
        self.h_c_classes = [GurmukhiTokenizer.EOS, GurmukhiTokenizer.PAD, GurmukhiTokenizer.BLANK] \
                            + self.subjoined_characters
        self.f_c_classes = [GurmukhiTokenizer.EOS, GurmukhiTokenizer.PAD] \
                            + self.vyanjan + self.svar + self.ank + self.chinh + self.special_consonants
        self.d_classes = [GurmukhiTokenizer.EOS, GurmukhiTokenizer.PAD] + self.matras # binary classification
        self.eos_id = 0
        self.pad_id = 1
        self.blank_id = 2
        self._normalize_charset()
        self.thresh = threshold
        self.max_grps = max_grps

        # dict with class indexes as keys and characters as values
        self.h_c_label_map = {k:c for k,c in enumerate(self.h_c_classes, start=0)}
        # 0 will be reserved for blank
        self.f_c_label_map = {k:c for k,c in enumerate(self.f_c_classes, start=0)}
        # blank not needed for embedding as it is Binary classification of each diacritic
        self.d_c_label_map = {k:c for k,c in enumerate(self.d_classes, start=0)}

        # dict with characters as keys and class indexes as values
        self.rev_h_c_label_map = {c:k for k,c in enumerate(self.h_c_classes, start=0)}
        self.rev_f_c_label_map = {c:k for k,c in enumerate(self.f_c_classes, start=0)}
        self.rev_d_label_map = {c:k for k,c in enumerate(self.d_classes, start=0)}

    def _normalize_charset(self) -> None:
        """
        Function to normalize the charset provided and converts the charset from list to tuple
        """
        self.h_c_classes = tuple([unicodedata.normalize("NFKD", c) for c in self.h_c_classes])
        self.f_c_classes = tuple([unicodedata.normalize("NFKD", c) for c in self.f_c_classes])
        self.d_classes = tuple([unicodedata.normalize("NFKD", c) for c in self.d_classes])

    def get_charset(self) -> list:
        return self.h_c_classes + self.f_c_classes + self.d_classes + (self.halanth, )

    def _check_h_c(self, label: str, idx: int) -> bool:
        """
        Method to check if the character at index idx in label is a half-char or not
        Returns:
        - bool: True if the current index is a half character or not
        """
        # check if the current char is halanth and next char is in h_c_set
        return idx + 1 < len(label) and label[idx] == self.halanth and label[idx + 1] in self.rev_h_c_label_map

    def _check_f_c(self, label, idx) -> bool:
        """
        Method to check if the character at index idx in label is a full-char or not
        Returns:
        - bool: True if the current idx is a full character
        """
        if idx < len(label) and label[idx] in self.rev_f_c_label_map:
            # Check if the next character is '਼' and if so, consider it as part of the full character
            if idx + 1 < len(label) and label[idx + 1] == '਼':
                return True
            return True
        return False

    def _check_d_c(self, label, idx) -> bool:
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
        for Gurmukhi, each group should contain:
            1) at most 1 half-characters
            2) at most 2 diacritics
            3) 1 full-character
        Args:
        - label (str): label for which groups are provided
        - grps (tuple): tuple containing the groups of the label

        Returns:
        - bool: True if all groups pass the sanity check else raises an Exception
        """
        for grp in grps:
            # allowed character category counts
            h_c_count, f_c_count, d_c_count = 1, 1, 2
            d_seen = []
            i = 0
            while i < len(grp):
                if grp[i] in self.rev_f_c_label_map and f_c_count > 0:
                    f_c_count -= 1
                    if i + 1 < len(grp) and grp[i + 1] == '਼':
                        i += 1  # skip the '਼'
                elif grp[i] == self.halanth and h_c_count > 0:
                    if i + 1 < len(grp) and grp[i + 1] in self.rev_h_c_label_map:
                        h_c_count -= 1
                        i += 1  # skip the next character as it is part of the halanth combination
                elif grp[i] in self.rev_d_label_map and d_c_count > 0:
                    d_c_count -= 1
                    if grp[i] in d_seen:
                        print(f"Duplicate Diacritic in group {grp} for label {label}")
                        return False
                    d_seen.append(grp[i])
                else:
                    print(f"Invalid character {grp[i]} in group {grp} for label {label}")
                    return False
                i += 1

            if f_c_count == 1 or (h_c_count == 1 and f_c_count == 1 and d_c_count == 2):
                print(f"There are no full character in group {grp} for {label}")
                return False

        return True

    def label_transform(self, label: str) -> tuple:
        """
        Transform Gurmukhi labels into groups
        Args:
        - label (str): label to transform
        Returns:
        - tuple: groups of the label
        """
        grps = ()
        running_grp = ""
        idx = 0
        while idx < len(label):
            t = idx
            # Check for full character
            if self._check_f_c(label, idx):
                running_grp += label[idx]
                idx += 1
                # Check if the next character is '਼' and include it
                if idx < len(label) and label[idx] == '਼':
                    running_grp += label[idx]
                    idx += 1

            # Check for half character
            if self._check_h_c(label, idx):
                running_grp += label[idx:idx + 2]
                idx += 2

            # Check for diacritic
            if self._check_d_c(label, idx):
                running_grp += label[idx]
                idx += 1
                if self._check_d_c(label, idx):
                    running_grp += label[idx]
                    idx += 1

            if t == idx:
                print(f"Invalid label {label} at index {t}")
                return ()

            grps = grps + (running_grp,)
            running_grp = ""

        return grps if self.grp_sanity(label, grps) else ()
