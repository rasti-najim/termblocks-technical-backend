from typing import List, Optional, Dict, Any
import os
import logging
import uuid
import shutil
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File as FastAPIFile, Form, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field as PydanticField
from sqlmodel import SQLModel, Field, create_engine, Session, Relationship, select, column
from dotenv import load_dotenv
from sqlalchemy.exc import SQLAlchemyError, ProgrammingError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Define link tables for many-to-many relationships
class ItemFileLink(SQLModel, table=True):
    file_id: Optional[int] = Field(default=None, foreign_key="file.id", primary_key=True)
    item_id: Optional[int] = Field(default=None, foreign_key="item.id", primary_key=True)

class CategoryItemLink(SQLModel, table=True):
    category_id: Optional[int] = Field(default=None, foreign_key="category.id", primary_key=True)
    item_id: Optional[int] = Field(default=None, foreign_key="item.id", primary_key=True)

class ChecklistCategoryLink(SQLModel, table=True):
    checklist_id: Optional[int] = Field(default=None, foreign_key="checklist.id", primary_key=True)
    category_id: Optional[int] = Field(default=None, foreign_key="category.id", primary_key=True)

class File(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    url: str
    content_type: Optional[str] = None
    size: Optional[int] = None
    uploaded_at: datetime = Field(default_factory=datetime.now)
    
    items: List["Item"] = Relationship(back_populates="files", link_model=ItemFileLink)

class Item(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    allow_multiple_files: bool = Field(default=False)
    is_file_upload_field: bool = Field(default=False)
    
    files: List[File] = Relationship(back_populates="items", link_model=ItemFileLink)
    categories: List["Category"] = Relationship(back_populates="items", link_model=CategoryItemLink)

class Category(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    
    items: List[Item] = Relationship(back_populates="categories", link_model=CategoryItemLink)
    checklists: List["Checklist"] = Relationship(back_populates="categories", link_model=ChecklistCategoryLink)

class Checklist(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    share_token: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), index=True, unique=True)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    is_public: bool = Field(default=False)
    
    categories: List[Category] = Relationship(back_populates="checklists", link_model=ChecklistCategoryLink)

# Request models
class FileCreate(BaseModel):
    name: str
    url: str
    content_type: Optional[str] = None
    size: Optional[int] = None

class ItemCreate(BaseModel):
    name: str
    allow_multiple_files: bool = False
    is_file_upload_field: bool = False
    files: Optional[List[FileCreate]] = None

class CategoryCreate(BaseModel):
    name: str
    items: Optional[List[ItemCreate]] = None

class ChecklistCreate(BaseModel):
    name: str
    categories: Optional[List[CategoryCreate]] = None

class FileUploadRequest(BaseModel):
    item_id: int

# Dependency to get database session
def get_db():
    with Session(engine) as session:
        yield session

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/checklists")
def read_checklists(db: Session = Depends(get_db)):
    checklists = db.exec(select(Checklist)).all()
    return {"checklists": [{"id": checklist.id, "name": checklist.name, "share_token": checklist.share_token} for checklist in checklists]}

@app.get("/checklists/{checklist_id}")
def read_checklist(checklist_id: int, db: Session = Depends(get_db)):
    checklist = db.get(Checklist, checklist_id)
    if not checklist:
        raise HTTPException(status_code=404, detail="Checklist not found")
    
    # Build the full response with categories, items, and files
    categories = []
    for category in checklist.categories:
        items = []
        for item in category.items:
            files = [{"id": file.id, "name": file.name, "url": file.url, "content_type": file.content_type, "size": file.size, "uploaded_at": file.uploaded_at} for file in item.files]
            items.append({
                "id": item.id,
                "name": item.name,
                "allow_multiple_files": item.allow_multiple_files,
                "is_file_upload_field": item.is_file_upload_field,
                "files": files
            })
        categories.append({
            "id": category.id,
            "name": category.name,
            "items": items
        })
    
    return {
        "id": checklist.id,
        "name": checklist.name,
        "share_token": checklist.share_token,
        "is_public": checklist.is_public,
        "created_at": checklist.created_at,
        "updated_at": checklist.updated_at,
        "categories": categories
    }

@app.get("/shared/{share_token}")
def read_shared_checklist(share_token: str, db: Session = Depends(get_db)):
    """Access a checklist using its share token (public link)"""
    checklist = db.exec(select(Checklist).where(Checklist.share_token == share_token)).first()
    if not checklist:
        raise HTTPException(status_code=404, detail="Shared checklist not found")
    
    if not checklist.is_public:
        raise HTTPException(status_code=403, detail="This checklist is not shared publicly")
    
    # Use the same response format as the regular checklist endpoint
    return read_checklist(checklist.id, db)

@app.post("/checklists")
def create_checklist(checklist_data: ChecklistCreate, db: Session = Depends(get_db)):
    """Create a new checklist with its categories and items"""
    new_checklist = Checklist(name=checklist_data.name)
    db.add(new_checklist)
    db.commit()
    db.refresh(new_checklist)
    
    # Add categories if provided
    if checklist_data.categories:
        for cat_data in checklist_data.categories:
            new_category = Category(name=cat_data.name)
            db.add(new_category)
            db.commit()
            db.refresh(new_category)
            
            # Link category to checklist
            new_checklist.categories.append(new_category)
            
            # Add items if provided
            if cat_data.items:
                for item_data in cat_data.items:
                    new_item = Item(
                        name=item_data.name,
                        allow_multiple_files=item_data.allow_multiple_files,
                        is_file_upload_field=item_data.is_file_upload_field
                    )
                    db.add(new_item)
                    db.commit()
                    db.refresh(new_item)
                    
                    # Link item to category
                    new_category.items.append(new_item)
                    
                    # Add files if provided
                    if item_data.files:
                        for file_data in item_data.files:
                            new_file = File(
                                name=file_data.name,
                                url=file_data.url,
                                content_type=file_data.content_type,
                                size=file_data.size
                            )
                            db.add(new_file)
                            db.commit()
                            db.refresh(new_file)
                            
                            # Link file to item
                            new_item.files.append(new_file)
    
    db.commit()
    return {"checklist_id": new_checklist.id, "share_token": new_checklist.share_token}

@app.post("/checklists/{checklist_id}/clone")
def clone_checklist(checklist_id: int, db: Session = Depends(get_db)):
    """Clone an existing checklist with all its categories, items, and structure (not files)"""
    original = db.get(Checklist, checklist_id)
    if not original:
        raise HTTPException(status_code=404, detail="Checklist not found")
    
    # Create new checklist with a copy of the original name
    clone = Checklist(name=f"Copy of {original.name}")
    db.add(clone)
    db.commit()
    db.refresh(clone)
    
    # Clone all categories and their items
    for category in original.categories:
        new_category = Category(name=category.name)
        db.add(new_category)
        db.commit()
        db.refresh(new_category)
        
        # Link to the new checklist
        clone.categories.append(new_category)
        
        # Clone all items in this category
        for item in category.items:
            new_item = Item(
                name=item.name,
                allow_multiple_files=item.allow_multiple_files,
                is_file_upload_field=item.is_file_upload_field
            )
            db.add(new_item)
            db.commit()
            db.refresh(new_item)
            
            # Link to the new category
            new_category.items.append(new_item)
    
    db.commit()
    return {"checklist_id": clone.id, "share_token": clone.share_token}

@app.put("/checklists/{checklist_id}")
def update_checklist(checklist_id: int, checklist_data: Dict[str, Any], db: Session = Depends(get_db)):
    """Update a checklist's basic properties"""
    checklist = db.get(Checklist, checklist_id)
    if not checklist:
        raise HTTPException(status_code=404, detail="Checklist not found")
    
    # Update basic attributes
    if "name" in checklist_data:
        checklist.name = checklist_data["name"]
    
    if "is_public" in checklist_data:
        checklist.is_public = checklist_data["is_public"]
    
    checklist.updated_at = datetime.now()
    db.commit()
    
    return {"checklist_id": checklist.id}

@app.post("/checklists/{checklist_id}/categories")
def add_category(checklist_id: int, category_data: CategoryCreate, db: Session = Depends(get_db)):
    """Add a new category to a checklist"""
    checklist = db.get(Checklist, checklist_id)
    if not checklist:
        raise HTTPException(status_code=404, detail="Checklist not found")
    
    new_category = Category(name=category_data.name)
    db.add(new_category)
    db.commit()
    db.refresh(new_category)
    
    # Link to checklist
    checklist.categories.append(new_category)
    
    # Add items if provided
    if category_data.items:
        for item_data in category_data.items:
            new_item = Item(
                name=item_data.name,
                allow_multiple_files=item_data.allow_multiple_files,
                is_file_upload_field=item_data.is_file_upload_field
            )
            db.add(new_item)
            db.commit()
            db.refresh(new_item)
            
            # Link to category
            new_category.items.append(new_item)
    
    # Update checklist timestamp
    checklist.updated_at = datetime.now()
    db.commit()
    
    return {"category_id": new_category.id}

@app.put("/categories/{category_id}")
def update_category(category_id: int, category_data: Dict[str, Any], db: Session = Depends(get_db)):
    """Update a category's properties"""
    category = db.get(Category, category_id)
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    
    # Update name if provided
    if "name" in category_data:
        category.name = category_data["name"]
    
    # Update linked checklist timestamps
    for checklist in category.checklists:
        checklist.updated_at = datetime.now()
    
    db.commit()
    return {"category_id": category.id}

@app.delete("/categories/{category_id}")
def delete_category(category_id: int, db: Session = Depends(get_db)):
    """Delete a category and its items"""
    category = db.get(Category, category_id)
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    
    # Update linked checklist timestamps
    for checklist in category.checklists:
        checklist.updated_at = datetime.now()
    
    # Delete the category (SQLModel will handle cascade deletion)
    db.delete(category)
    db.commit()
    
    return {"message": "Category deleted successfully"}

@app.post("/categories/{category_id}/items")
def add_item(category_id: int, item_data: ItemCreate, db: Session = Depends(get_db)):
    """Add a new item to a category"""
    category = db.get(Category, category_id)
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    
    new_item = Item(
        name=item_data.name,
        allow_multiple_files=item_data.allow_multiple_files,
        is_file_upload_field=item_data.is_file_upload_field
    )
    db.add(new_item)
    db.commit()
    db.refresh(new_item)
    
    # Link to category
    category.items.append(new_item)
    
    # Update linked checklist timestamps
    for checklist in category.checklists:
        checklist.updated_at = datetime.now()
    
    db.commit()
    
    return {"item_id": new_item.id}

@app.put("/items/{item_id}")
def update_item(item_id: int, item_data: Dict[str, Any], db: Session = Depends(get_db)):
    """Update an item's properties"""
    item = db.get(Item, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Update fields if provided
    if "name" in item_data:
        item.name = item_data["name"]
    
    if "allow_multiple_files" in item_data:
        item.allow_multiple_files = item_data["allow_multiple_files"]
    
    if "is_file_upload_field" in item_data:
        item.is_file_upload_field = item_data["is_file_upload_field"]
    
    # Update linked checklist timestamps
    for category in item.categories:
        for checklist in category.checklists:
            checklist.updated_at = datetime.now()
    
    db.commit()
    
    return {"item_id": item.id}

@app.delete("/items/{item_id}")
def delete_item(item_id: int, db: Session = Depends(get_db)):
    """Delete an item and its files"""
    item = db.get(Item, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Update linked checklist timestamps
    for category in item.categories:
        for checklist in category.checklists:
            checklist.updated_at = datetime.now()
    
    # Delete the item (SQLModel will handle cascade deletion)
    db.delete(item)
    db.commit()
    
    return {"message": "Item deleted successfully"}

# Create upload directory if it doesn't exist
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload/{item_id}")
async def upload_file(
    item_id: int, 
    file: UploadFile = FastAPIFile(...),
    db: Session = Depends(get_db)
):
    """Upload a file and link it to an item"""
    # Verify the item exists
    item = db.get(Item, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Check if the item allows files
    if not item.is_file_upload_field:
        raise HTTPException(status_code=400, detail="This item does not accept file uploads")
    
    # Check if multiple files are allowed when files already exist
    if not item.allow_multiple_files and len(item.files) > 0:
        raise HTTPException(status_code=400, detail="This item only accepts a single file")
    
    # Generate a unique filename to prevent collisions
    file_ext = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    
    # Create item-specific directory
    item_upload_dir = os.path.join(UPLOAD_DIR, f"item_{item_id}")
    os.makedirs(item_upload_dir, exist_ok=True)
    
    # Save the file
    file_path = os.path.join(item_upload_dir, unique_filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Create file record in database
    relative_url = f"/uploads/item_{item_id}/{unique_filename}"
    db_file = File(
        name=file.filename,
        url=relative_url,
        content_type=file.content_type,
        size=os.path.getsize(file_path),
    )
    db.add(db_file)
    db.commit()
    db.refresh(db_file)
    
    # Link file to item
    item.files.append(db_file)
    
    # Update linked checklist timestamps
    for category in item.categories:
        for checklist in category.checklists:
            checklist.updated_at = datetime.now()
    
    db.commit()
    
    return {
        "file_id": db_file.id,
        "name": db_file.name,
        "url": db_file.url,
        "content_type": db_file.content_type,
        "size": db_file.size
    }

@app.delete("/files/{file_id}")
def delete_file(file_id: int, db: Session = Depends(get_db)):
    """Delete a file"""
    file = db.get(File, file_id)
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Update linked checklist timestamps
    for item in file.items:
        for category in item.categories:
            for checklist in category.checklists:
                checklist.updated_at = datetime.now()
    
    # Delete the physical file if it exists
    if file.url and file.url.startswith("/uploads/"):
        file_path = os.path.join(os.getcwd(), file.url.lstrip("/"))
        if os.path.exists(file_path):
            os.remove(file_path)
    
    # Delete the database record
    db.delete(file)
    db.commit()
    
    return {"message": "File deleted successfully"}

# Configure PostgreSQL engine with connection pool settings
database_url = os.getenv("DB_URL")
if not database_url:
    raise ValueError("DB_URL environment variable is not set")

# Configure engine with PostgreSQL specific settings
engine = create_engine(
    database_url,
    echo=True,  # Log SQL queries (set to False in production)
    pool_pre_ping=True,  # Check connection before using it
    pool_recycle=300,  # Recycle connections every 5 minutes
    pool_size=5,  # Maximum number of connections to keep in the pool
    max_overflow=10  # Maximum number of connections to create beyond pool_size
)

def create_db_and_tables():
    try:
        SQLModel.metadata.create_all(engine)
        logger.info("Database tables created successfully")
    except ProgrammingError as e:
        logger.error(f"Error creating database tables: {e}")
        # For first time setup, you might need to create the database
        if "does not exist" in str(e):
            logger.warning("Database might not exist. Please create it first")
            raise HTTPException(status_code=500, detail="Database does not exist. Please create it first")
        raise

# Run this if the file is executed directly
if __name__ == "__main__":
    # Create database tables
    create_db_and_tables()
    
    # Print instructions
    print("\n✅ Database tables created!\n")
    print("To run the application, use one of these commands:")
    print("    uvicorn main:app --reload")
    print("    python run.py\n")
    print("Then visit http://localhost:8000/docs to test the API\n")