from enum import Enum
from pydantic import BaseModel, Field, validator
from typing import Optional


class TrainingParameters(BaseModel):
    n_clusters: int = Field(description='number of clusters')
    max_iter: int = Field(description="maximum iteration")

class TestingParameters(BaseModel):
    show_progress: Optional[int] = Field(default=1, description="number of iterations to progress report")
