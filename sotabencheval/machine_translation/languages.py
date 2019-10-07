from enum import Enum

_full_forms = {
    "en": "English",
    "fr": "French",
    "de": "German",
}


class Language(Enum):
    English = "en"
    French = "fr"
    German = "de"

    @property
    def fullname(self):
        return _full_forms[self.value]
