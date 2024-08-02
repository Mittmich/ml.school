"""Helper classes for aws pipeline creation."""
from pydantic import BaseModel, computed_field, Field, ConfigDict
from typing import List, Dict, Union, Optional
from sagemaker.workflow.pipeline_context import LocalPipelineSession, PipelineSession



class Configuration(BaseModel):
    """Pipeline configuration."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    local_mode: bool = Field(default=False, description="Run pipeline in local mode.", exclude=True)
    default_instance: str = Field(default="ml.m5.xlarge", description="Sagemaker instance type. Ignored in local mode.", exclude=True)
    arm64: bool = Field(default=False, exclude=True)
    framework_version: str = Field(default="2.12", description="Sagemaker framework version.")
    python_version: str = Field(default="py310", description="Python version.")
    bucket: str = Field(..., exclude=True)

    @computed_field
    def instance_type(self) -> str:
        """Return instance type."""
        return "local" if self.local_mode else self.default_instance
    
    @computed_field
    def session(self) -> PipelineSession:
        """Return pipeline session."""
        return LocalPipelineSession(default_bucket=self.bucket) if self.local_mode else PipelineSession(default_bucket=self.bucket)

    @computed_field
    def image(self) -> Optional[str]:
        """image to be used"""
        if self.arm64 and self.local_mode:
            return "sagemaker-tensorflow-toolkit-local"
        return None

