from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import uvicorn

import imgtokens

app = FastAPI()
app.state.initialized = False  # Add this line

@app.on_event("startup")
async def startup_event():
    print("Preloading models...")
    imgtokens.preload_models()  # this is fast
    print("Creating ImagePatchWordTokenizer...")
    ipwt = imgtokens.ImagePatchWordTokenizer()  # this is slow on first run
    # TODO: Figure out why
    print("Initializing model...")
    ipwt._init_model()
    print("Done initializing")
    app.state.ipwt = ipwt
    app.state.initialized = True  # Set to True when initialization is done

@app.get("/readiness")
async def readiness():
    if app.state.initialized:
        return JSONResponse(content={"status": "ready"})
    else:
        return JSONResponse(content={"status": "initializing"}, status_code=503)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")


@app.post("/process-image/")
async def process_image(request: Request, file: UploadFile = File(...), num_words: int = 8):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        img = Image.open(BytesIO(await file.read()))
    except Exception as e:
        raise HTTPException(status_code=400, detail="Could not open image")

    ipwt = request.app.state.ipwt  # Access the shared ipwt object
    words = ipwt.process_img(img, num_words)

    words_serializable = [[str(w) for w in row] for row in words]
    return JSONResponse(content={"words": words_serializable})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
