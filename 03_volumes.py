from kfp import dsl
from kfp import compiler
from kfp import kubernetes


@dsl.component(
    base_image='python:3.13-slim',
)
def generate(
    test_str: str,
) -> None:
    """Write a test string to a file on the mounted persistent volume."""
    import os
    from datetime import datetime

    # Generate timestamped filename
    time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    mount_path = "/mnt/data"
    file_name = f"example_{time_now}.txt"
    full_path = os.path.join(mount_path, file_name)
    
    # Ensure mount directory exists
    os.makedirs(mount_path, exist_ok=True)
    
    # Write test string to file
    with open(full_path, 'w') as f:
        f.write(test_str)


@dsl.pipeline(
    name='volume-pipeline',
    description='Write a test file to a persistent volume.'
)

# Pipeline parameters can be specified via the Kubeflow UI when creating a Run!
def volume_pipeline(test_str: str = "This is a test string for the volume pipeline."):
    """Pipeline task that writes a test string to a PVC-mounted volume."""
    generation_task = generate(
        test_str=test_str
    )

    # Mount PVC to container
    kubernetes.mount_pvc(
        task=generation_task,
        pvc_name='tutorial',
        mount_path='/mnt/data'
    )

    # Configure resource requests and limits
    generation_task.set_cpu_request('2')
    generation_task.set_cpu_limit('4')
    generation_task.set_memory_request('4G')
    generation_task.set_memory_limit('8G')


if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=volume_pipeline,
        package_path='compiled_volume_pipeline.yaml'
    )
    print("Saved to 'compiled_volume_pipeline.yaml'.")