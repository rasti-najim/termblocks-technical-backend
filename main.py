from typing import List, Optional, Dict, Any
import os
import logging
import uuid
import shutil
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, UploadFile, File as FastAPIFile, Form, Depends, Body, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field as PydanticField, EmailStr
from sqlmodel import SQLModel, Field, create_engine, Session, Relationship, select, column
from dotenv import load_dotenv
from sqlalchemy.exc import SQLAlchemyError, ProgrammingError
from sqlalchemy import func, select as sqlalchemy_select
from passlib.context import CryptContext
from jose import JWTError, jwt
import secrets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Security settings
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = None  # Set to None to disable token expiration

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

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

# Dependency to get database session
def get_db():
    with Session(engine) as session:
        yield session

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

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(unique=True, index=True)
    hashed_password: str
    full_name: Optional[str] = None
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.now)
    
    checklists: List["Checklist"] = Relationship(back_populates="owner")

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
    owner_id: Optional[int] = Field(default=None, foreign_key="user.id")
    
    categories: List[Category] = Relationship(back_populates="checklists", link_model=ChecklistCategoryLink)
    owner: Optional[User] = Relationship(back_populates="checklists")

# Request models
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str

class UserRead(UserBase):
    id: int
    is_active: bool
    created_at: datetime

class FileCreate(BaseModel):
    name: str
    url: str
    content_type: Optional[str] = None
    size: Optional[int] = None

class ItemCreate(BaseModel):
    name: str
    files: Optional[List[FileCreate]] = None

class CategoryCreate(BaseModel):
    name: str
    items: Optional[List[ItemCreate]] = None

class ChecklistCreate(BaseModel):
    name: str
    categories: Optional[List[CategoryCreate]] = None

class FileUploadRequest(BaseModel):
    item_id: int

# Request models for combined responses
class UserWithToken(BaseModel):
    user: UserRead
    access_token: str
    token_type: str

# Security functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(db: Session, email: str, password: str):
    user = db.exec(select(User).where(User.email == email)).first()
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
        to_encode.update({"exp": expire})
    # No expiration if expires_delta is None
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception
    user = db.exec(select(User).where(User.email == token_data.email)).first()
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Auth endpoints
@app.post("/register", response_model=UserWithToken)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user and return a token"""
    # Check if user already exists
    db_user = db.exec(select(User).where(User.email == user.email)).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user with hashed password
    hashed_password = get_password_hash(user.password)
    db_user = User(
        email=user.email,
        hashed_password=hashed_password,
        full_name=user.full_name
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Generate token for the new user
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=None
    )
    
    # Create response with both user data and token
    return {
        "user": db_user,
        "access_token": access_token,
        "token_type": "bearer"
    }

@app.post("/token", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Login to get access token"""
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Pass None for expires_delta to create a token that never expires
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=None
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=UserRead)
def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get current user information"""
    return current_user

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/checklists")
def read_checklists(db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    # Query checklists owned by the current user
    checklist_with_category_counts = (
        db.execute(
            sqlalchemy_select(
                Checklist.id,
                Checklist.name,
                Checklist.share_token,
                Checklist.created_at,
                Checklist.updated_at,
                func.count(ChecklistCategoryLink.category_id).label("category_count")
            )
            .outerjoin(ChecklistCategoryLink, Checklist.id == ChecklistCategoryLink.checklist_id)
            .where(Checklist.owner_id == current_user.id)
            .group_by(Checklist.id)
        )
        .all()
    )
    
    # Create a mapping of checklist IDs to their data including category counts
    checklist_data = {
        row[0]: {
            "id": row[0],
            "name": row[1],
            "share_token": row[2],
            "created_at": row[3],
            "updated_at": row[4],
            "category_count": row[5],
            "item_count": 0  # Will be updated in the next query
        }
        for row in checklist_with_category_counts
    }
    
    # Get checklist IDs for the second query
    checklist_ids = list(checklist_data.keys())
    
    if checklist_ids:  # Only run second query if we have checklists
        # Query to get item counts for each checklist in one go
        item_counts = (
            db.execute(
                sqlalchemy_select(
                    Checklist.id,
                    func.count(Item.id).label("item_count")
                )
                .join(ChecklistCategoryLink, Checklist.id == ChecklistCategoryLink.checklist_id)
                .join(Category, Category.id == ChecklistCategoryLink.category_id)
                .join(CategoryItemLink, Category.id == CategoryItemLink.category_id)
                .join(Item, Item.id == CategoryItemLink.item_id)
                .where(Checklist.id.in_(checklist_ids))
                .group_by(Checklist.id)
            )
            .all()
        )
        
        # Update the checklist data with item counts
        for checklist_id, item_count in item_counts:
            if checklist_id in checklist_data:
                checklist_data[checklist_id]["item_count"] = item_count
    
    # Convert to a list and return
    result = list(checklist_data.values())
    return {"checklists": result}

@app.get("/checklists/{checklist_id}")
def read_checklist(checklist_id: int, db: Session = Depends(get_db), current_user: Optional[User] = Depends(get_current_user)):
    checklist = db.get(Checklist, checklist_id)
    if not checklist:
        raise HTTPException(status_code=404, detail="Checklist not found")
    
    # Check permissions
    if checklist.owner_id != current_user.id and not checklist.is_public:
        raise HTTPException(status_code=403, detail="You don't have permission to view this checklist")
    
    # Build the full response with categories, items, and files
    categories = []
    for category in checklist.categories:
        items = []
        for item in category.items:
            files = [{"id": file.id, "name": file.name, "url": file.url, "content_type": file.content_type, "size": file.size, "uploaded_at": file.uploaded_at} for file in item.files]
            items.append({
                "id": item.id,
                "name": item.name,
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
        "categories": categories,
        "owner_id": checklist.owner_id
    }

@app.get("/shared/{share_token}")
def read_shared_checklist(share_token: str, db: Session = Depends(get_db)):
    """Access a checklist using its share token (public link)"""
    checklist = db.exec(select(Checklist).where(Checklist.share_token == share_token)).first()
    if not checklist:
        raise HTTPException(status_code=404, detail="Shared checklist not found")
    
    if not checklist.is_public:
        raise HTTPException(status_code=403, detail="This checklist is not shared publicly")
    
    # Build the full response with categories, items, and files
    categories = []
    for category in checklist.categories:
        items = []
        for item in category.items:
            files = [{"id": file.id, "name": file.name, "url": file.url, "content_type": file.content_type, "size": file.size, "uploaded_at": file.uploaded_at} for file in item.files]
            items.append({
                "id": item.id,
                "name": item.name,
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
        "categories": categories,
        "owner_id": checklist.owner_id
    }

@app.post("/checklists")
def create_checklist(checklist_data: ChecklistCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    """Create a new checklist with its categories and items"""
    new_checklist = Checklist(
        name=checklist_data.name,
        owner_id=current_user.id
    )
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
                        name=item_data.name
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
def clone_checklist(checklist_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    """Clone an existing checklist with all its categories, items, and structure (not files)"""
    original = db.get(Checklist, checklist_id)
    if not original:
        raise HTTPException(status_code=404, detail="Checklist not found")
    
    # Check if the user has permission to view the original (must be owner or public)
    if original.owner_id != current_user.id and not original.is_public:
        raise HTTPException(status_code=403, detail="You don't have permission to clone this checklist")
    
    # Create new checklist with a copy of the original name
    clone = Checklist(
        name=f"Copy of {original.name}",
        owner_id=current_user.id  # The clone is owned by the current user
    )
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
                name=item.name
            )
            db.add(new_item)
            db.commit()
            db.refresh(new_item)
            
            # Link to the new category
            new_category.items.append(new_item)
    
    db.commit()
    return {"checklist_id": clone.id, "share_token": clone.share_token}

@app.put("/checklists/{checklist_id}")
def update_checklist(checklist_id: int, checklist_data: Dict[str, Any], db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    """Update a checklist's basic properties"""
    checklist = db.get(Checklist, checklist_id)
    if not checklist:
        raise HTTPException(status_code=404, detail="Checklist not found")
    
    # Check ownership
    if checklist.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="You don't have permission to update this checklist")
    
    # Update basic attributes
    if "name" in checklist_data:
        checklist.name = checklist_data["name"]
    
    if "is_public" in checklist_data:
        checklist.is_public = checklist_data["is_public"]
    
    checklist.updated_at = datetime.now()
    db.commit()
    
    return {"checklist_id": checklist.id}

@app.post("/checklists/{checklist_id}/categories")
def add_category(checklist_id: int, category_data: CategoryCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    """Add a new category to a checklist"""
    checklist = db.get(Checklist, checklist_id)
    if not checklist:
        raise HTTPException(status_code=404, detail="Checklist not found")
    
    # Check ownership
    if checklist.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="You don't have permission to modify this checklist")
    
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
                name=item_data.name
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
def update_category(category_id: int, category_data: Dict[str, Any], db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    """Update a category's properties"""
    category = db.get(Category, category_id)
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    
    # Check if the user owns any checklists that contain this category
    user_owns_category = False
    for checklist in category.checklists:
        if checklist.owner_id == current_user.id:
            user_owns_category = True
            break
    
    if not user_owns_category:
        raise HTTPException(status_code=403, detail="You don't have permission to update this category")
    
    # Update name if provided
    if "name" in category_data:
        category.name = category_data["name"]
    
    # Update linked checklist timestamps
    for checklist in category.checklists:
        if checklist.owner_id == current_user.id:
            checklist.updated_at = datetime.now()
    
    db.commit()
    return {"category_id": category.id}

@app.delete("/categories/{category_id}")
def delete_category(category_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    """Delete a category and its items"""
    category = db.get(Category, category_id)
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    
    # Check if the user owns any checklists that contain this category
    user_owns_category = False
    checklists_to_update = []
    for checklist in category.checklists:
        if checklist.owner_id == current_user.id:
            user_owns_category = True
            checklists_to_update.append(checklist)
    
    if not user_owns_category:
        raise HTTPException(status_code=403, detail="You don't have permission to delete this category")
    
    # Update linked checklist timestamps
    for checklist in checklists_to_update:
        checklist.updated_at = datetime.now()
    
    # Delete the category (SQLModel will handle cascade deletion)
    db.delete(category)
    db.commit()
    
    return {"message": "Category deleted successfully"}

@app.post("/categories/{category_id}/items")
def add_item(category_id: int, item_data: ItemCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    """Add a new item to a category"""
    category = db.get(Category, category_id)
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    
    # Check if the user owns any checklists that contain this category
    user_owns_category = False
    checklists_to_update = []
    for checklist in category.checklists:
        if checklist.owner_id == current_user.id:
            user_owns_category = True
            checklists_to_update.append(checklist)
    
    if not user_owns_category:
        raise HTTPException(status_code=403, detail="You don't have permission to modify this category")
    
    new_item = Item(
        name=item_data.name
    )
    db.add(new_item)
    db.commit()
    db.refresh(new_item)
    
    # Link to category
    category.items.append(new_item)
    
    # Update linked checklist timestamps
    for checklist in checklists_to_update:
        checklist.updated_at = datetime.now()
    
    db.commit()
    
    return {"item_id": new_item.id}

@app.put("/items/{item_id}")
def update_item(item_id: int, item_data: Dict[str, Any], db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    """Update an item's properties"""
    item = db.get(Item, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Check if the user owns the checklist that contains this item
    user_owns_item = False
    checklists_to_update = []
    for category in item.categories:
        for checklist in category.checklists:
            if checklist.owner_id == current_user.id:
                user_owns_item = True
                checklists_to_update.append(checklist)
    
    if not user_owns_item:
        raise HTTPException(status_code=403, detail="You don't have permission to update this item")
    
    # Update fields if provided
    if "name" in item_data:
        item.name = item_data["name"]
    
    # Update linked checklist timestamps
    for checklist in checklists_to_update:
        checklist.updated_at = datetime.now()
    
    db.commit()
    
    return {"item_id": item.id}

@app.delete("/items/{item_id}")
def delete_item(item_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    """Delete an item and its files"""
    item = db.get(Item, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Check if the user owns the checklist that contains this item
    user_owns_item = False
    checklists_to_update = []
    for category in item.categories:
        for checklist in category.checklists:
            if checklist.owner_id == current_user.id:
                user_owns_item = True
                checklists_to_update.append(checklist)
    
    if not user_owns_item:
        raise HTTPException(status_code=403, detail="You don't have permission to delete this item")
    
    # Update linked checklist timestamps
    for checklist in checklists_to_update:
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
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Upload a file and link it to an item"""
    # Verify the item exists
    item = db.get(Item, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Check if the user owns the checklist that contains this item
    user_owns_item = False
    checklists_to_update = []
    for category in item.categories:
        for checklist in category.checklists:
            if checklist.owner_id == current_user.id:
                user_owns_item = True
                checklists_to_update.append(checklist)
    
    if not user_owns_item:
        raise HTTPException(status_code=403, detail="You don't have permission to upload to this item")
    
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
    for checklist in checklists_to_update:
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
def delete_file(file_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    """Delete a file"""
    file = db.get(File, file_id)
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Check if the user owns the checklist that contains this file
    user_owns_file = False
    checklists_to_update = []
    for item in file.items:
        for category in item.categories:
            for checklist in category.checklists:
                if checklist.owner_id == current_user.id:
                    user_owns_file = True
                    checklists_to_update.append(checklist)
    
    if not user_owns_file:
        raise HTTPException(status_code=403, detail="You don't have permission to delete this file")
    
    # Update linked checklist timestamps
    for checklist in checklists_to_update:
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