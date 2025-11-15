# Workflow Template - Create Your Own Custom Workflows

A template repository for creating and managing custom workflow functions in Python. This template provides the structure and documentation standards for building reusable workflow components.

## 1. How to Fork This Template

### Create Your Own Workflow Repository
1. **Click the "Use this template" button** on GitHub
2. **Create a new repository** with your desired name (e.g., "my-custom-workflows")
3. **Clone your new repository:**
```bash
git clone https://github.com/P-E-R-D/server.git
cd server
```

## 2. How to Set Up Your Custom Workflow

### Prerequisites
- Python 3.7+
- pip package manager

### Initial Setup
1. **Rename the package** (optional but recommended):
   - Change all occurrences of "per_datasets" to your package name
   - Update folder structure accordingly

2. **Install base dependencies:**
```bash
cd workflows/add
pip install -r requirements.txt
```

3. **Install your package in development mode:**
```bash
pip install -e .
```

## 3. How to Test Your Workflow

### Test the Example Function
```python
from src.workflows.add.workflow import add

# Test the included example 
result = add(2.5, 3.7)
print(f"Result: {result}")  # Should print: 6.2
```

### Create Your Own Tests
Create a `test.py` file:
```python
# test.py file
from src.workflows.add.workflow import add

# Test the included example 
result = add(2.5, 3.7)
print(f"Result: {result}")  # Should print: 6.2
```

Run with:
```bash
python -m src.workflows.add.test
```

## 4. How to Modify and Create Your Own Workflows

### Step 1: Create a New Workflow Folder
```bash
# Create a new workflow directory
mkdir workflows/my_custom_workflow
cd workflows/my_custom_workflow
```

### Step 2: Create Your Workflow Files
**Create `workflow.py`:**
```python
"""
Your custom workflow function
"""

def my_custom_workflow(input_data: dict) -> dict:
    """
    ## my_custom_workflow
    
    Describe what your workflow does here.
    
    ### **parameters**
    
    input_data : dict
        Input data description
        
    ### **returns**
    
    dict
        Processed output data
        
    ### **examples**
    
    >>> from your_package.workflows.my_custom_workflow import my_custom_workflow
    >>> result = my_custom_workflow({"input": "data"})
    >>> print(result)
    {'processed': 'result'}
    """
    # Your workflow logic here
    processed_data = {"processed": input_data.get("input", "") + "_processed"}
    return processed_data
```

**Create `__init__.py`:**
```python
"""
My custom workflow module
"""

from .workflow import my_custom_workflow

__all__ = ['my_custom_workflow']
```

**Create `requirements.txt`** (if your workflow needs specific packages):
```
pandas>=1.3.0
numpy>=1.21.0
requests>=2.25.0
```

### Step 3: Update Package Exports
Edit the main `workflows/__init__.py`:
```python
"""
Workflows module for your_package_name
"""

from .add import add
from .substract import subtract
from .my_custom_workflow import my_custom_workflow  # Add your new workflow

__all__ = ['add', 'subtract', 'my_custom_workflow']  # Update this list
```

## 5. How to Upload and Distribute Your Workflow


## Workflow Structure Template
```

├── workflows/
│   ├── __init__.py                 # Main exports
│   ├── add/                        # Example workflow
│   │   ├── __init__.py
│   │   ├── workflow.py
│   │   └── requirements.txt
│   └── my_custom_workflow/         # Your custom workflow
│       ├── __init__.py
│       ├── workflow.py
│       └── requirements.txt             
└── README.md                      
```