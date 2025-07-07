import unicodedata

class TamilTokenizer:
    """
    Class for decoding and encoding Tamil Labels
    """
    BLANK = "[B]"
    EOS = "[E]"
    PAD = "[P]"

    def __init__(self, threshold: float = 0.5, max_grps: int = 25):
        self.uyir_eluthu = ['அ', 'ஆ', 'இ', 'ஈ', 'உ', 'ஊ', 'எ', 'ஏ', 'ஐ', 'ஒ', 'ஓ', 'ஔ']
        self.mei_eluthu = ['க', 'ங', 'ச', 'ஞ', 'ட', 'ண', 'த', 'ந', 'ப', 'ம', 'ய', 'ர', 'ல', 'வ', 'ழ', 'ள', 'ற', 'ன']
        self.uyirmei_eluthu = [c + m for c in self.mei_eluthu for m in ['', 'ா', 'ி', 'ீ', 'ு', 'ூ', 'ெ', 'ே', 'ை', 'ொ', 'ோ', 'ௌ']]
        self.grantha_eluthu = ['ஜ', 'ஷ', 'ஸ', 'ஹ']
        self.pulli = '்'
        self.aytham = 'ஃ'
        self.numbers = ['௦', '௧', '௨', '௩', '௪', '௫', '௬', '௭', '௮', '௯', '௰', '௱', '௲']
        self.symbols = ['ௐ', '₹', '।', '!', '$', ',', '.', '-', '%']

        self.threshold = threshold
        self.max_grps = max_grps
        self._normalize_charset()

        self.f_c_classes = [TamilTokenizer.EOS, TamilTokenizer.PAD, TamilTokenizer.BLANK] + \
                           self.uyir_eluthu + self.mei_eluthu + self.uyirmei_eluthu + \
                           self.grantha_eluthu + [self.aytham] + self.numbers + self.symbols
        self.d_classes = [TamilTokenizer.EOS, TamilTokenizer.PAD] + ['ா', 'ி', 'ீ', 'ு', 'ூ', 'ெ', 'ே', 'ை', 'ொ', 'ோ', 'ௌ', self.pulli]

        self.eos_id = 0
        self.pad_id = 1
        self.blank_id = 2

        self.f_c_label_map = {k: c for k, c in enumerate(self.f_c_classes, start=0)}
        self.d_c_label_map = {k: c for k, c in enumerate(self.d_classes, start=0)}

        self.rev_f_c_label_map = {c: k for k, c in enumerate(self.f_c_classes, start=0)}
        self.rev_d_label_map = {c: k for k, c in enumerate(self.d_classes, start=0)}

    def get_charset(self) -> list:
        """
        Returns the complete charset used by the tokenizer
        """
        return self.f_c_classes + self.d_classes

    def _normalize_charset(self):
        """
        NFKD Normalize the input charset
        """
        self.uyir_eluthu = [unicodedata.normalize("NFKD", char) for char in self.uyir_eluthu]
        self.mei_eluthu = [unicodedata.normalize("NFKD", char) for char in self.mei_eluthu]
        self.uyirmei_eluthu = [unicodedata.normalize("NFKD", char) for char in self.uyirmei_eluthu]
        self.grantha_eluthu = [unicodedata.normalize("NFKD", char) for char in self.grantha_eluthu]
        self.numbers = [unicodedata.normalize("NFKD", char) for char in self.numbers]
        self.symbols = [unicodedata.normalize("NFKD", char) for char in self.symbols]

    def _check_f_c(self, label: str, idx: int) -> bool:
        """
        Checks whether the passed idx is a full char or not in the label
        """
        if idx < len(label) and label[idx] in self.f_c_classes:
            return True
        return False

    def _check_diac(self, label: str, idx: int) -> bool:
        """
        Checks whether the passed idx is a diacritic or not in the label
        """
        if idx < len(label) and label[idx] in self.d_classes:
            return True
        return False

    def grp_sanity(self, label: str, grps: tuple) -> bool:
        """
        Checks whether the groups are properly formed
        for Tamil, each group should contain:
            1) at most 2 diacritics
            2) 1 full-character
        """
        for idx, grp in enumerate(grps):
            f_c_count, d_c_count = 1, 2
            d_seen = ''
            i = 0
            while i < len(grp):
                if self._check_f_c(grp, i) and f_c_count > 0:
                    f_c_count -= 1
                elif self._check_diac(grp, i) and d_c_count > 0:
                    if f_c_count != 0:
                        print(f"No full char to attach diacritic: {grp} in {label}")
                        return False

                    if d_c_count == 2:  # First diacritic
                        d_c_count -= 1
                        d_seen = grp[i]
                    elif grp[i] != d_seen:  # 2nd diacritic
                        d_c_count -= 1
                    else:
                        print(f"Duplicate Diacritic in group {grp} for label {label}")
                        return False
                else:
                    if grp[i] not in self.rev_f_c_label_map and grp[i] not in self.rev_d_label_map:
                        print(f"Invalid {grp[i]} in group {grp} for label {label}")
                    elif d_c_count < 0:
                        print(f"More than 2 diacritics found {grp} in {label}")
                    else:
                        print(f"Ill formed group {grp} in {label}")
                    return False
                i += 1

            if (f_c_count, d_c_count) == (1, 2):
                print(f"Invalid group: {grp} for {label} OR Empty group")
                return False
            elif f_c_count == 1 and idx != len(grps) - 1:
                print(f"There is no full char in grp: {grp} in {label}")
                return False

        return True

    def label_transform(self, label: str) -> tuple:
        """
        Transform Tamil labels into groups
        """
        grps = ()
        running_grp = ""
        idx = 0
        while idx < len(label):
            t = idx
            # The group starts with a full-char
            if self._check_f_c(label, idx):
                running_grp += label[idx]
                idx += 1

            # Diacritics need not be always present
            if self._check_diac(label, idx):
                running_grp += label[idx]
                idx += 1
                # There can be 2 diacritics in a group
                if idx < len(label) and self._check_diac(label, idx):
                    running_grp += label[idx]
                    idx += 1

            if t == idx:
                print(f"Invalid label {label}-{t}")
                return ()

            grps = grps + (running_grp,)
            running_grp = ""

        return grps if self.grp_sanity(label, grps) else ()
