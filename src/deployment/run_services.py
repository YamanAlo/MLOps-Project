import uvicorn
from multiprocessing import Process
from .config import settings

def run_api():
    """Run the main API service"""
    uvicorn.run(
        "src.deployment.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )


def main():
    """Run both API and monitoring services"""
    # Start API service
    api_process = Process(target=run_api)
    api_process.start()
    
  
    try:
        # Wait for processes to complete
        api_process.join()
        
    except KeyboardInterrupt:
        print("\nShutting down services...")
        api_process.terminate()
        
        api_process.join()
       

if __name__ == "__main__":
    main() 