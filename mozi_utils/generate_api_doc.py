# doc_generator.py

import inspect
from typing import List, Callable


def generate_api_docs(module) -> str:
    """
    Generate API documentation for all functions in a given module.

    Args:
        module: The module containing the functions to document.

    Returns:
        str: Markdown formatted API documentation.
    """
    docs = "# API Documentation\n\n"

    functions: List[Callable] = [obj for name, obj in inspect.getmembers(module) if inspect.isfunction(obj)]

    for func in functions:
        docs += f"## `{func.__name__}`\n\n"

        # Get the docstring
        docstring = inspect.getdoc(func)
        if docstring:
            docs += f"{docstring}\n\n"

        # Get the function signature
        signature = inspect.signature(func)
        docs += f"### Signature\n\n```python\n{func.__name__}{signature}\n```\n\n"

        # Get parameter details
        docs += "### Parameters\n\n"
        for param_name, param in signature.parameters.items():
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else "Not specified"
            docs += f"- `{param_name}`: {param_type}\n"

        docs += "\n"

        # Get return type
        return_type = signature.return_annotation if signature.return_annotation != inspect.Signature.empty else "Not specified"
        docs += f"### Returns\n\n- {return_type}\n\n"

        docs += "---\n\n"  # Add a separator between functions

    return docs


# Example usage
if __name__ == "__main__":
    import api_demo

    api_docs = generate_api_docs(api_demo)
    print(api_docs)

    # Optionally, save to a file
    with open("api_docs.md", "w") as f:
        f.write(api_docs)