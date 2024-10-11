import re
import numpy as np

def unicode(text):
    # Define the replacement mappings
    replacements = {
        "òa": "oà", "óa": "oá", "ỏa": "oả", "õa": "oã", "ọa": "oạ",
        "òe": "oè", "óe": "oé", "ỏe": "oẻ", "õe": "oẽ", "ọe": "oẹ",
        "ùy": "uỳ", "úy": "uý", "ủy": "uỷ", "ũy": "uỹ", "ụy": "uỵ",
        "Ủy": "Uỷ"
    }
    
    # Define a function to apply the replacements to a single text
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text
    

def dupplicate_punctuation(text, pos = []):
    if pos == []:
        pos = list(range(len(text)))

    def replace(text, pattern, replacement, pos ):
        matches = [0]  # Initialize a list to track the positions of matches.



        # Nested function to handle each regex match.
        def capture_and_replace(match, ret):
            matches.extend([match.start() + 1, match.end()])  # Store the start+1 and end positions of the match.
            return ret  # Return the replacement text for the match.

        # Get the length of the original text.
        l = len(text)

        # Use `re.sub` to find all occurrences of the pattern and replace them.
        # `capture_and_replace` is used as a callback to record match positions.
        text = re.sub(pattern, lambda match: capture_and_replace(match, replacement), text, flags=re.IGNORECASE)

        # Add the length of the modified text to the matches.
        matches.append(l)

        # Split the matches list into pairs of start and end positions.
        slices = np.array_split(matches, int(len(matches) / 2))

        # Adjust the `pos` list according to the changes made in the text.
        res = []
        for s in slices:
            res += pos[s[0]:s[1]]  # Extend `res` with the corresponding slice of `pos`.

        # Ensure the length of the updated `text` matches the updated `pos` list.
        assert len(text) == len(res)

        return text, res  # Return the updated text and the adjusted `pos` list.

    
    # collapse duplicated punctuations 
    punc = ',. !?\"\''
    for c in punc:
        pat = '([' + c + ']{2,})'
        text, pos = replace(text, pat, c, pos)
    assert len(text) == len(pos)
    return text, pos


#text = "Hello!! How are you???"
#preprocess(text) # 'Hello! How are you?' , pos
