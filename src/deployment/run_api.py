import uvicorn
from .config import settings

def main():
    """Run the FastAPI server"""
    uvicorn.run(
        "src.deployment.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )

if __name__ == "__main__":
    main() 