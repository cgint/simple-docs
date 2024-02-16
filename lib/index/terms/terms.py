from typing import List


characters_to_replace_before = ["\"", "</", "<", ">", "\n", "\r", "\t", "?", "!", ".", ",", ":", ";", "(", ")", "[", "]", "{", "}", "+", "="]
characters_to_ignore_when_equals_term = ["#", "-"]
characters_replace_by = " "

def terms_from_txt(text: str) -> List[str]:
    for char in characters_to_replace_before:
        text = text.replace(char, characters_replace_by)
    terms = text.split()
    return [term.lower().strip() for term in terms if term not in characters_to_ignore_when_equals_term]
