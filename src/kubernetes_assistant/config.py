from pydantic import Field
from pydantic_settings import BaseSettings


class ModelConfig(BaseSettings):
    model_host: str = Field(default="http://localhost:11434", alias="MODEL_HOST")
    model_id: str = Field(default="qwen3:latest", alias="MODEL_ID")


class KubernetesAssistantConfig(BaseSettings):
    llm_config: ModelConfig = Field(default_factory=ModelConfig)
    cluster_name: str = Field(default="The Cluster", alias="CLUSTER_NAME")
    agent_name: str = Field(default="KubeBot", alias="AGENT_NAME")
    agent_role: str = Field(default="intern system administrator", alias="AGENT_ROLE")
    kubeconfig_path: str = Field(default="./k3s.yaml", alias="KUBE_CONFIG_PATH")
    discord_token: str = Field(alias="DISCORD_TOKEN")
