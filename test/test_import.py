import os
import sys

# Add the project root to sys.path to allow importing llamafactory and EasyCL
# Assumes this script is run from the LLaMA-Factory root directory or EasyCL/test
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to sys.path")

print("Attempting imports...")

try:
    # Try importing a core llamafactory module
    from llamafactory.model import load_model, load_tokenizer
    print("Successfully imported load_model and load_tokenizer from llamafactory.model")

    # Try importing a core EasyCL hparams module
    from EasyCL.hparams.cl_finetuning_args import CLFinetuningArguments
    print("Successfully imported from EasyCL.hparams")

    # Try importing a specific CL method's trainer from EasyCL
    from EasyCL.cl.ewc.ewc_trainer import EWCSeq2SeqTrainer
    print("Successfully imported EWCSeq2SeqTrainer from EasyCL.cl.ewc")

    # Try importing the CL Evaluator manager from EasyCL
    from EasyCL.cl_workflow.evaluator import CLEvaluator
    print("Successfully imported from EasyCL.cl_workflow")

    print("\nImport tests passed!")

except ImportError as e:
    print(f"\nImport test failed: {e}")
    # Print sys.path for debugging
    print("\nCurrent sys.path:")
    for path in sys.path:
        print(f"- {path}")

except Exception as e:
    print(f"\nAn unexpected error occurred: {e}") 