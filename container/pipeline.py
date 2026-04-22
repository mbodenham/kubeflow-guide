from kfp import dsl
from kfp import compiler
from kfp import kubernetes

# =============================================================================
# 🤖 COMPONENT: Qwen Text Generation
# =============================================================================

@dsl.component(base_image='mbodenham/kubeflow-tut')
def generate_text(prompt: str) -> None:
    """Run Qwen3.5-0.8B model inference and save output to mounted volume.
    
    Args:
        prompt: Input text prompt for the model
    """
    import os
    import subprocess
    from datetime import datetime

    # Generate unique filename with timestamp
    time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    mount_path = "/mnt/data"
    file_name = f"output_{time_now}.txt"
    full_path = os.path.join(mount_path, file_name)

    # Ensure mount directory exists
    os.makedirs(mount_path, exist_ok=True)

    # Run main.py with prompt and output path
    cmd = [
        "python", "main.py",
        prompt,
        "--output", full_path,
        "--model", "Qwen/Qwen3.5-0.8B"
    ]
    print(f"🤖 Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd="/workspace")

    if result.returncode != 0:
        raise RuntimeError(f"❌ Command failed with exit code {result.returncode}")


# =============================================================================
# 📐 PIPELINE DEFINITION
# =============================================================================

@dsl.pipeline(
    name='qwen-text-generation-pipeline',
    description='Generate text using Qwen3.5-0.8B model on Kubernetes.'
)
def qwen_pipeline(prompt: str = "Hello, how are you?"):
    """Pipeline that generates text using Qwen3.5-0.8B and saves to PVC.
    
    Args:
        prompt: Input text prompt (default: "Hello, how are you?")
    """
    generation_task = generate_text(prompt=prompt)

    # Mount PVC for persistent storage
    kubernetes.mount_pvc(
        task=generation_task,
        pvc_name='tutorial',
        mount_path='/mnt/data'
    )

    # Configure resources for GPU inference
    generation_task.set_cpu_request('4')
    generation_task.set_cpu_limit('8')
    generation_task.set_memory_request('16G')
    generation_task.set_memory_limit('32G')
    generation_task.set_accelerator_limit(1)
    generation_task.add_node_selector_constraint('nvidia.com/gpu')


# =============================================================================
# ⚙️ COMPILATION
# =============================================================================

if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=qwen_pipeline,
        package_path='compiled_qwen_pipeline.yaml'
    )
    print("✅ Saved to 'compiled_qwen_pipeline.yaml'.")
