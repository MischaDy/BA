def clean_str(string, to_lower=True):
    clean_string = string.strip()
    if to_lower:
        return clean_string.lower()
    return clean_string.upper()
