class BengaliTokenizer:
    BLANK = "[B]"
    EOS = "[E]"
    PAD = "[P]"

    def __init__(self, threshold:float = 0.5, max_grps:int = 25):
        # Bengali consonants (vyanjans)
        self.vyanjan = ['ক', 'খ', 'গ', 'ঘ', 'ঙ',
                       'চ', 'ছ', 'জ', 'ঝ', 'ঞ',
                       'ট', 'ঠ', 'ড', 'ঢ', 'ণ',
                       'ত', 'থ', 'দ', 'ধ', 'ন',
                       'প', 'ফ', 'ব', 'ভ', 'ম',
                       'য', 'র', 'ল', 'শ', 'ষ',
                       'স', 'হ', 'ড়', 'ঢ়', 'য়', 'ৱ', 'ৎ']

        # Bengali vowels (svar)
        self.svar = ['অ', 'আ', 'ই', 'ঈ', 'উ',
                     'ঊ', 'ঋ', 'এ', 'ঐ',
                     'ও', 'ঔ']
        self.svar1 = ['অ', 'ই', 'ঈ', 'উ',
                     'ঊ', 'ঋ', 'এ', 'ঐ',
                     'ও', 'ঔ']

        # Bengali vowel signs (matras/diacritics)
        self.matras = ['া', 'ি', 'ী', 'ু', 'ূ',
                      'ৃ', 'ে', 'ৈ', 'ো', 'ৌ',
                      'ং', 'ঃ', '়', 'ঁ','্' ]
        self.matras1 = ['া', 'ি', 'ী', 'ু', 'ূ',
                      'ৃ', 'ে', 'ৈ', 'ো', 'ৌ',
                      'ং', 'ঃ', '়', 'ঁ' ]

        # Special characters and punctuation
        self.chinh = ['ॐ', '₹', '।', '॥', '!', '$', ',', '.', '-', '%']

        # Bengali numerals
        self.ank = ['০', '১', '২', '৩', '৪', '৫', '৬', '৭', '৮', '৯']

        # Halant/Hasant (used to remove inherent vowel)
        self.halanth = '্'

        # Define character classes
        self.h_c_classes = [BengaliTokenizer.EOS, BengaliTokenizer.PAD, BengaliTokenizer.BLANK] + self.vyanjan
        self.f_c_classes = [BengaliTokenizer.EOS, BengaliTokenizer.PAD] + self.vyanjan + self.svar + self.ank + self.chinh
        self.d_classes = [BengaliTokenizer.EOS, BengaliTokenizer.PAD] + self.matras

        self.eos_id = 0
        self.pad_id = 1
        self.blank_id = 2

        self._normalize_charset()
        self.thresh = threshold
        self.max_grps = max_grps

        # Create mapping dictionaries
        self.h_c_label_map = {k:c for k,c in enumerate(self.h_c_classes)}
        self.f_c_label_map = {k:c for k,c in enumerate(self.f_c_classes)}
        self.d_c_label_map = {k:c for k,c in enumerate(self.d_classes)}

        self.rev_h_c_label_map = {c:k for k,c in enumerate(self.h_c_classes)}
        self.rev_f_c_label_map = {c:k for k,c in enumerate(self.f_c_classes)}
        self.rev_d_label_map = {c:k for k,c in enumerate(self.d_classes)}

    def _normalize_charset(self) -> None:
        """Normalize the character sets using NFC normalization"""
        self.h_c_classes = tuple([unicodedata.normalize("NFC", c) for c in self.h_c_classes])
        self.f_c_classes = tuple([unicodedata.normalize("NFC", c) for c in self.f_c_classes])
        self.d_classes = tuple([unicodedata.normalize("NFC", c) for c in self.d_classes])
        # self.ank = [unicodedata.normalize("NFKD", char) for char in self.ank]
        # self.chinh = [unicodedata.normalize("NFKD", char) for char in self.chinh]
        # self.svar = [unicodedata.normalize("NFKD", char) for char in self.svar]
        # self.vyanjan = [unicodedata.normalize("NFKD", char) for char in self.vyanjan]
        # self.matras = [unicodedata.normalize("NFC", char) for char in self.matras]


    def get_charset(self) -> list:
        """Return the complete character set"""
        return self.h_c_classes + self.f_c_classes + self.d_classes + (self.halanth,)

    def _check_h_c(self, label: str, idx: int) -> bool:
        """Check if character at idx is a half-character"""
        return idx < len(label) and label[idx] in self.rev_h_c_label_map \
            and idx + 1 < len(label) and label[idx + 1] == self.halanth

    def _check_f_c(self, label: str, idx: int) -> bool:
        """Check if character at idx is a full-character"""
        if idx + 1 < len(label) and label[idx] == 'অ' and label[idx + 1] == '্':
            return True  # "অ্য" is a valid full character

        return idx < len(label) and label[idx] in self.rev_f_c_label_map \
            and (idx + 1 >= len(label) or label[idx + 1] != self.halanth)

    def _check_d_c(self, label: str, idx: int) -> bool:
        """Check if character at idx is a diacritic"""
        return idx < len(label) and label[idx] in self.rev_d_label_map

    def grp_sanity(self, label: str, grps: tuple) -> bool:
        """
        Check if groups follow Bengali rules:
        1. Max 2 half characters at start
        2. Single root character
        3. Max 2 diacritics at end
        """
        for grp in grps:
            h_c_count, f_c_count, d_c_count = 2, 1, 2  # Maximum allowed counts
            d_seen = []
            i = 0

            while i < len(grp):
                if i + 1 < len(grp) and self.halanth == grp[i + 1] and grp[i] in self.rev_h_c_label_map and h_c_count > 0:
                    h_c_count -= 1
                    i += 2
                elif grp[i] in self.rev_f_c_label_map and f_c_count > 0:
                    f_c_count -= 1
                    i += 1
                elif grp[i] in self.rev_d_label_map and d_c_count > 0:
                    if grp[i] not in d_seen:
                        d_seen.append(grp[i])
                        d_c_count -= 1
                        i += 1
                    else:
                        print(f"Duplicate Diacritic in group {grp} for label {label}")
                        return False
                else:
                    print(f"Invalid character {grp[i]} in group {grp} for label {label}")
                    return False

            # Verify Bengali-specific rules
            if f_c_count == 1:  # Must have exactly one root character
                print(f"Missing root character in group {grp}")
                return False
            if len(d_seen) > 2:  # Cannot have more than 2 diacritics
                print(f"Too many diacritics in group {grp}")
                return False

        return True

    def label_transform(self, label: str) -> tuple:
        """Transform Bengali text into character groups"""
        grps = ()
        running_grp = ""
        idx = 0

        while idx < len(label):
            t = idx
            # Check for half characters (max 2)
            half_char_count = 0
            while self._check_h_c(label, idx) and half_char_count < 2:
                running_grp += label[idx:idx+2]
                idx += 2
                half_char_count += 1

            # Check for full character (root), including exceptions
            if self._check_f_c(label, idx):
                running_grp += label[idx]
                idx += 1

            # Check for diacritics (max 2)
            diacritic_count = 0
            while self._check_d_c(label, idx) and diacritic_count < 2:
                running_grp += label[idx]
                idx += 1
                diacritic_count += 1

            if t == idx:
                print(f"Invalid label {label} at position {t}")
                return ()

            grps = grps + (running_grp,)
            running_grp = ""

        return grps if self.grp_sanity(label, grps) else ()
