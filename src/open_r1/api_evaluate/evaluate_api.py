from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager

from model_api import CustomAPIMLModel

# API 配置
api_url = "http://dify-z100.adcp.lab.zverse.space/v1/chat-messages"
api_key = "app-RpZcf0LdOOZscFh9LTHun4C6"

# 配置评估跟踪
evaluation_tracker = EvaluationTracker(
    output_dir="./results",
    save_details=True,
    push_to_hub=False
)

# 配置管道参数
pipeline_params = PipelineParameters(
    launcher_type=ParallelismManager.NONE,
    override_batch_size=1,
    max_samples=3,  # 可根据需求调整
    custom_tasks_directory="../evaluate_astro.py"
)

custom_model = CustomAPIMLModel(api_url=api_url, api_key=api_key)

pipeline = Pipeline(
    tasks="custom|gpqa:astro|0|0",
    pipeline_parameters=pipeline_params,
    evaluation_tracker=evaluation_tracker,
    model=custom_model,
    model_config=None,
)

# 执行评估
pipeline.evaluate()
pipeline.save_and_push_results()
pipeline.show_results()
