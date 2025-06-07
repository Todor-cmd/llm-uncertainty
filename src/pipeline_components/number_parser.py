import re

def extract_number_from_text(text):
    """
    Extract the first number from text, handling various formats.

    Args:
        text (str): Text that may contain numbers
        
    Returns:
        float: Extracted number, or mark as -1.0 if no number found
    """
    # Remove any whitespace and convert to string
    text = str(text).strip()

    # Try to find numbers in the text using regex
    # This pattern matches positive and negative integers and decimals
    number_pattern = r'-?\d+(?:\.\d+)?'
    matches = re.findall(number_pattern, text)

    if matches:
        # Return the first number found
        number = float(matches[0])
        if number > 100 or number < 0:
            # print(f"Warning: Extracted number {number} is out of range, marking inalid as -1.0")
            return -1.0
        return number

    # If no number found, try to extract digits only (keeping minus sign)
    digits_only = re.sub(r'[^\d.-]', '', text)
    if digits_only and digits_only not in ['.', '-', '.-', '-.']:
        try:
            return float(digits_only)
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