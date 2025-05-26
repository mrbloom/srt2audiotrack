import re


def two_cases(title):
    if not title:
       return '',''
    return title[0].upper() + title[1:], title[0].lower() + title[1:]

def parse_vocabular_file(vocabular_path):
    """
    Parses a vocabular file with lines like:
        Kiyv<=>Kiev
        Ekaterina II<=>Ekaterina druga
    Returns a list of tuples [("Kiyv","Kiev"), ("Ekaterina II","Ekaterina druga")].
    """
    replacements = []
    with open(vocabular_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            # Expect a separator <=>
            if '<=>' in line:
                old, new = line.split('<=>', 1)
                new_upper, new_lower = two_cases(new.strip())
                old_strip = old.strip()
                replacements.append((old_strip, new_upper))
                replacements.append((old_strip, new_lower))
    # Sort by length of the old string, descending (longest first).
    replacements.sort(key=lambda x: len(x[0]), reverse=True)
    return replacements


def modify_subtitles_with_vocabular_wholefile_even_partishally(subtitle_path, vocabular_path, output_path):
    replacements = parse_vocabular_file(vocabular_path)

    with open(subtitle_path, 'r', encoding='utf-8') as infile:
        text = infile.read()

    for old, new in replacements:
        text = text.replace(old, new)

    with open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.write(text)

    return text

def modify_subtitles_with_vocabular_wholefile(subtitle_path, vocabular_path, output_path):
    replacements = parse_vocabular_file(vocabular_path)

    with open(subtitle_path, 'r', encoding='utf-8') as infile:
        text = infile.read()

    # Use regular expressions with word boundaries (\b)
    for old, new in replacements:
        # Escape 'old' to avoid special regex characters
        old_escaped = re.escape(old)
        # Build the pattern: \bOLD\b ensures we only match full words
        pattern = rf"\b{old_escaped}\b"
        text = re.sub(pattern, new, text)

    with open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.write(text)

    return text

def apply_replacements(line, replacements):
    """
    Applies each replacement (old->new) in order to a single line.
    Because we sorted by length in parse_vocabular_file,
    longer strings get replaced first.
    """
    for old, new in replacements:
        line = line.replace(old, new)
    return line

def modify_subtitles_with_vocabular(subtitle_path, vocabular_path, output_path):
    """
    Reads `subtitle_path` line-by-line, applies the replacements
    from `vocabular_path`, and writes to `output_path`.
    """
    # Get the replacements
    replacements = parse_vocabular_file(vocabular_path)

    with open(subtitle_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # Apply the replacements on the current line
            new_line = apply_replacements(line, replacements)
            outfile.write(new_line)