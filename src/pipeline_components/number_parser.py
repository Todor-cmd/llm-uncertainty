import re

def extract_number_from_text(text, bottom_up=False, prefix=None, prefix_only=False):
    """
    Extract the first number from text, handling various formats.

    Args:
        text (str): Text that may contain numbers
        bottom_up (bool): If True and prefix search fails, search from end of text backwards
        prefix (str): If provided, look for numbers directly after this prefix first
        
    Returns:
        float: Extracted number, or mark as -1.0 if no number found
    """
    # Remove any whitespace and convert to string
    text = str(text).strip()

    # First, try to find number after prefix if provided
    if prefix is not None:
        # Look for the prefix followed by optional whitespace and then a number
        prefix_pattern = re.escape(prefix) + r'\s*(-?\d+(?:\.\d+)?)'
        prefix_match = re.search(prefix_pattern, text, re.IGNORECASE)
        if prefix_match:
            number = float(prefix_match.group(1))
            if 0 <= number <= 100:
                return number
            # If out of range, continue to other methods
            if prefix_only:
                return -1.0

    # Try to find numbers in the text using regex
    # This pattern matches positive and negative integers and decimals
    number_pattern = r'-?\d+(?:\.\d+)?'
    matches = re.findall(number_pattern, text)

    if matches:
        if bottom_up:
            # Search from the end backwards
            for match in reversed(matches):
                number = float(match)
                if 0 <= number <= 100:
                    return number
        else:
            # Search from the beginning (original behavior)
            for match in matches:
                number = float(match)
                if 0 <= number <= 100:
                    return number

    # If no number found, try to extract digits only (keeping minus sign)
    digits_only = re.sub(r'[^\d.-]', '', text)
    if digits_only and digits_only not in ['.', '-', '.-', '-.']:
        try:
            number = float(digits_only)
            if 0 <= number <= 100:
                return number
        except ValueError:
            pass

    # Default fallback if no number can be extracted
    # print(f"Warning: Could not extract number from '{text}', marking inalid as -1.0")
    return -1.0


if __name__ == "__main__":
    print(extract_number_from_text("The number is 23.45"))
    print(extract_number_from_text("The number is 123"))
    print(extract_number_from_text("The number is -1"))
    print(extract_number_from_text("The number is 23"))
    print(extract_number_from_text("The number is 12 01"))
    print(extract_number_from_text("The number is 01,45"))
    print(extract_number_from_text("The number 1 and 52"))
    print(extract_number_from_text("The number is"))
    print(extract_number_from_text("3"))
    print(extract_number_from_text("The number is 23.45\n score: 3", prefix = "score:"))
    print(extract_number_from_text("The number is 23.45\n score: 3", bottom_up=True))
    print(extract_number_from_text("The number is 23.45\n SCORE: 85", prefix = "score:"))
    print(extract_number_from_text("The number is 23.45\n Score: 75", prefix = "SCORE:"))
    print(extract_number_from_text("Analysis text 3\nUncertainty Score: 92", prefix = "uncertainty score:"))
    