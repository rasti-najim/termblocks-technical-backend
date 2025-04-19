# TermBlocks Checklists API

A FastAPI application for managing checklists with categories, items, and file uploads.

## 📋 Overview

This API provides a complete backend solution for creating and managing checklists with:

- Hierarchical organization (checklists → categories → items)
- File upload capabilities for individual items
- Checklist sharing via public links
- Checklist cloning functionality

## 🛠️ Setup

### Prerequisites

- Python 3.8+
- PostgreSQL

### Installation

1. Clone the repository:

   ```bash
   git clone [repository URL]
   cd termblocks-technical-backend
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv

   # On macOS/Linux
   source venv/bin/activate

   # On Windows
   venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure the database:

   - Create a PostgreSQL database:
     ```sql
     CREATE DATABASE termblocks;
     ```
   - Create a `.env` file with your database connection string:
     ```
     DB_URL=postgresql://username:password@localhost:5432/termblocks
     ```

5. Initialize the database:
   ```bash
   python main.py
   ```

## 🚀 Running the Application

Start the server with one of these commands:

```bash
# Option 1: Using the run script
python run.py

# Option 2: Using uvicorn directly
uvicorn main:app --reload
```

Access the API at: http://localhost:8000
Interactive documentation: http://localhost:8000/docs

## 🔍 API Endpoints

### Checklists

- `GET /checklists` - List all checklists
- `GET /checklists/{checklist_id}` - Get a specific checklist
- `POST /checklists` - Create a new checklist
- `PUT /checklists/{checklist_id}` - Update a checklist
- `POST /checklists/{checklist_id}/clone` - Clone an existing checklist

### Categories

- `POST /checklists/{checklist_id}/categories` - Add a category
- `PUT /categories/{category_id}` - Update a category
- `DELETE /categories/{category_id}` - Delete a category

### Items

- `POST /categories/{category_id}/items` - Add an item
- `PUT /items/{item_id}` - Update an item
- `DELETE /items/{item_id}` - Delete an item

### Files

- `POST /upload/{item_id}` - Upload a file to an item
- `DELETE /files/{file_id}` - Delete a file

### Sharing

- `GET /shared/{share_token}` - Access a shared checklist

## 🗄️ Database Structure

The application uses these core models:

- **Checklist**: Top-level container with sharing capabilities
- **Category**: Groups related items within a checklist
- **Item**: Individual checklist entry, can accept file uploads
- **File**: Uploaded file with metadata and storage information

## 🔗 Frontend Integration

The API includes CORS configuration to allow integration with frontend applications.

## ⚠️ Troubleshooting

### Database Issues

- Verify PostgreSQL is running
- Check credentials in the `.env` file
- Ensure the database exists
- Confirm proper user permissions

### File Upload Issues

- Ensure the `uploads` directory exists and is writable
- Check file size limits in your configuration

## 📄 License

[Add license information]

## 👥 Contact

[Add contact information]
