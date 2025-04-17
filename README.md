# Checklists API

A FastAPI application for managing checklists with categories, items, and files.

## Features

- Create, read, update, and delete checklists with categories and items
- Add file upload fields to items, configurable to accept one or multiple files
- Clone existing checklists
- Share checklists via public links
- File uploads with proper handling of file types and storage

## Setup

1. Create a virtual environment:

   ```
   python -m venv venv
   ```

2. Activate the virtual environment:

   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```
   - On Windows:
     ```
     venv\Scripts\activate
     ```

3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Setup PostgreSQL:

   a. Ensure PostgreSQL is installed and running

   b. Create a database for the application:

   ```sql
   CREATE DATABASE termblocks;
   ```

   c. Create a `.env` file with your PostgreSQL database URL:

   ```
   DB_URL=postgresql://username:password@localhost:5432/termblocks
   ```

   Replace `username` and `password` with your actual PostgreSQL credentials.

5. Initialize the database:

   ```
   python main.py
   ```

## Running the Application

You can run the application in two ways:

1. Using the run script:

   ```
   python run.py
   ```

2. Or directly with uvicorn:

   ```
   uvicorn main:app --reload
   ```

3. Access the API documentation and testing interface at http://localhost:8000/docs

## Testing the API

FastAPI provides an interactive API documentation that allows you to:

1. See all available endpoints
2. Try out endpoints directly in your browser
3. View request and response formats
4. Upload files and test all functionality

To test the application:

1. Open your browser and go to http://localhost:8000/docs
2. Use the "Try it out" button on any endpoint
3. Fill in the required parameters and execute the request
4. View the results directly in the browser

## API Endpoints

### Checklists

- `GET /checklists`: List all checklists
- `GET /checklists/{checklist_id}`: Get a specific checklist with all related data
- `POST /checklists`: Create a new checklist
- `PUT /checklists/{checklist_id}`: Update a checklist
- `POST /checklists/{checklist_id}/clone`: Clone an existing checklist

### Categories

- `POST /checklists/{checklist_id}/categories`: Add a category to a checklist
- `PUT /categories/{category_id}`: Update a category
- `DELETE /categories/{category_id}`: Delete a category and its items

### Items

- `POST /categories/{category_id}/items`: Add an item to a category
- `PUT /items/{item_id}`: Update an item
- `DELETE /items/{item_id}`: Delete an item and its files

### Files

- `POST /upload/{item_id}`: Upload a file to an item
- `DELETE /files/{file_id}`: Delete a file

### Sharing

- `GET /shared/{share_token}`: View a shared checklist by its token

## Database Structure

The database has the following models:

- `Checklist`: Contains categories, with sharing options
- `Category`: Contains items
- `Item`: Contains files, can be configured as a file upload field that allows single or multiple files
- `File`: Represents a file with metadata and URL

## Frontend Integration

The API includes CORS configuration to allow cross-origin requests from the React frontend.

## Troubleshooting

If you encounter database connection issues:

1. Make sure PostgreSQL is running
2. Verify your database credentials in the `.env` file
3. Check that the database exists:
   ```sql
   SELECT datname FROM pg_database WHERE datname = 'termblocks';
   ```
4. Ensure your PostgreSQL user has appropriate permissions

If you have issues with file uploads, make sure the `uploads` directory is writable by the application.
