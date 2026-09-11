from fastapi import FastAPI

app = FastAPI(title="Project Tutor")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
