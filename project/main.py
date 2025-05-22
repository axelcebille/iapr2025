import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

def run_notebook(notebook_path):
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})

    print(f"Notebook {notebook_path} executed successfully.")

if __name__ == "__main__":
    notebook_path = os.path.join('src', 'report.ipynb')
    run_notebook(notebook_path)