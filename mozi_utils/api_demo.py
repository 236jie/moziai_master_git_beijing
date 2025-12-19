# demo.py

def greet(name: str) -> str:
    """
    Greet a person by name.

    Args:
        name (str): The name of the person to greet.

    Returns:
        str: A greeting message.
    """
    return f"Hello, {name}!"

def add(a: int, b: int) -> int:
    """
    Add two numbers together.

    Args:
        a (int): The first number.
        b (int): The second number.

    Returns:
        int: The sum of a and b.
    """
    return a + b

def is_palindrome(s: str) -> bool:
    """
    Check if a string is a palindrome.

    A palindrome is a word, phrase, number, or other sequence of characters
    that reads the same forward and backward.

    Args:
        s (str): The string to check.

    Returns:
        bool: True if the string is a palindrome, False otherwise.
    """
    return s == s[::-1]