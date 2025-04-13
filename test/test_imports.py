import pytest

def test_hparams_import():
    """Test importing core hyperparameter classes."""
    try:
        from easycl.hparams import CLFinetuningArguments, CLEvaluationArguments
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import hparams: {e}")

def test_cl_workflow_import():
    """Test importing CL workflow components."""
    try:
        from easycl.cl_workflow import CLWorkflow, CLEvaluator
        from easycl.cl_workflow.cl_eval import CLEvalEvaluator, CLMetricsCalculator
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import cl_workflow components: {e}")

def test_cl_methods_import():
    """Test importing a few CL method workflows and trainers."""
    imports_to_test = {
        "ewc": ["run_sft_ewc", "EWCSeq2SeqTrainer"],
        "lwf": ["run_sft_lwf", "LWFTrainer"],
        "replay": ["run_sft_replay", "ReplaySeq2SeqTrainer"],
        "abscl": ["run_sft_abscl", "ABSCLTrainer"],
        # Add more methods here if needed
    }
    failed_imports = []
    for method, components in imports_to_test.items():
        try:
            module = __import__(f"easycl.cl.{method}", fromlist=components)
            for component in components:
                assert hasattr(module, component), f"Component {component} not found in easycl.cl.{method}"
        except ImportError as e:
            failed_imports.append(f"easycl.cl.{method}: {e}")
        except AssertionError as e:
            failed_imports.append(str(e))
            
    if failed_imports:
        pytest.fail(f"Failed to import CL method components:\n" + "\n".join(failed_imports))
    else:
        assert True 