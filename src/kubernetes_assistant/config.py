import os

from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    model_host: str = Field(default=os.environ.get("MODEL_HOST", "http://localhost:11434"))
    model_id: str = Field(default=os.environ.get("MODEL_ID", "qwen3:latest"))

class KubernetesAssistantConfig(BaseModel):
    llm_config: ModelConfig = Field(default_factory=ModelConfig)
    cluster_name: str = Field(default=os.environ.get("CLUSTER_NAME", "The Cluster"))
    agent_name: str = Field(default=os.environ.get("AGENT_NAME", "KubeBot"))
    agent_role: str = Field(default=os.environ.get("AGENT_ROLE", "intern system administrator"))
    kubeconfig_path: str = Field(default=os.environ.get("KUBE_CONFIG_PATH", "./k3s.yaml"))
    discord_token: str = Field(default=os.environ["DISCORD_TOKEN"])
