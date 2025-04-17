import uvicorn
from main import create_db_and_tables

if __name__ == "__main__":
    # Create database tables
    create_db_and_tables()
    
    # Run the application
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 