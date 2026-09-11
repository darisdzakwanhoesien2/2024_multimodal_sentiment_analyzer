from pydantic import BaseModel


class JobCreatedResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: str
    error: str | None = None
    file_name: str


class HFVerifyResult(BaseModel):
    model: str
    accessible: bool
    detail: str


class HealthResponse(BaseModel):
    status: str
    packages: dict
