from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw, ImageFont
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
import os, uuid
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = "postgresql://postgres:1203@localhost:5432/thumbnail_db"

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True  
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class ImageDescription(Base):
    __tablename__ = "image_descriptions"

    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String, nullable=False)
    thumbnail_path = Column(String, nullable=False)
    description = Column(String, nullable=False)
    platform = Column(String, nullable=False)

app = FastAPI()

@app.on_event("startup")
def startup():
    print("📦 Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print(" Tables ready")

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
output_parser = StrOutputParser()

thumbnail_prompt = PromptTemplate(
    template="Generate a catchy and engaging text for a youtube thumbnail or instagram reel based on the user text and the platform, maximum 5 words, curiosity based. Platform: {platform}. User text: {user_text}",
    input_variables=["user_text", "platform"]
)

thumbnail_chain = thumbnail_prompt | model | output_parser

UPLOAD_DIR = "uploads"
THUMB_DIR = "thumbnails"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(THUMB_DIR, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/thumbnails", StaticFiles(directory=THUMB_DIR), name="thumbnails")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print("🔥 GLOBAL ERROR:", exc)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc)}
    )
@app.get("/")
def root():
    return {"status": "API is running"}


def create_thumbnail(image_path, text, platform):
    img = Image.open(image_path).convert("RGB")

    if platform.lower() == "youtube":
        img = img.resize((1280, 720))
        font_size = 85
        text_y = 520
    elif platform.lower() == "instagram":
        img = img.resize((1080, 1920))
        font_size = 90
        text_y = 1400
    else:
        raise ValueError("Platform must be 'youtube' or 'instagram'")

    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arialbd.ttf", font_size)
    except:
        font = ImageFont.load_default()

    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
    else:
        text_width, _ = draw.textsize(text, font=font)

    x = (img.width - text_width) // 2

    draw.text(
        (x, text_y),
        text,
        font=font,
        fill="white",
        stroke_width=6,
        stroke_fill="black"
    )

    thumb_name = f"{uuid.uuid4().hex}.jpg"
    thumb_path = os.path.join(THUMB_DIR, thumb_name)
    img.save(thumb_path, quality=95)

    return thumb_name


@app.post("/create-thumbnail")
async def create_thumbnail_api(
    file: UploadFile = File(...),
    user_text: str = Form(...),
    platform: str = Form(...)
):
    contents = await file.read()
    image_name = f"{uuid.uuid4().hex}.jpg"
    image_path = os.path.join(UPLOAD_DIR, image_name)

    with open(image_path, "wb") as f:
        f.write(contents)

    thumbnail_text = thumbnail_chain.invoke({
        "user_text": user_text,
        "platform": platform
    })

    thumb_name = create_thumbnail(image_path, thumbnail_text, platform)

    db = SessionLocal()
    record = ImageDescription(
        image_path=image_name,
        thumbnail_path=thumb_name,
        description=thumbnail_text,
        platform=platform
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    db.close()

    return {
        "id": record.id,
        "thumbnail_url": f"/thumbnails/{thumb_name}"
    }


@app.get("/thumbnail/{thumbnail_id}")
def get_thumbnail(thumbnail_id: int):
    db = SessionLocal()
    record = db.query(ImageDescription).filter(ImageDescription.id == thumbnail_id).first()
    db.close()

    if not record:
        return {"error": "Thumbnail not found"}

    return {
        "thumbnail_url": f"/thumbnails/{record.thumbnail_path}",
        "description": record.description,
        "platform": record.platform
    }
