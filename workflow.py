from dotenv import load_dotenv

load_dotenv()


"""
Add workflow function for per_datasets
"""

def add(a: float, b: float) -> float:
    """
    ## add
    
    Adds two numbers together as a simple workflow example.
    
    ### **parameters**
    
    a : float
        First number to add
        
    b : float
        Second number to add
        
    ### **returns**
    
    float
        Sum of a and b
        
    ### **examples**
    
    >>> from per_datasets.workflows.add import add
    >>> result = add(2.5, 3.7)
    >>> print(result)
    6.2
    """
    return a + b