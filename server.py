from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image
import uvicorn

import imgtokens

app = FastAPI()
app.state.initialized = False  # Add this line

@app.on_event("startup")
async def startup_event():
    # imgtokens.preload_models()  # this will OOM
    print("Creating ImagePatchWordTokenizer...")
    ipwt = imgtokens.ImagePatchWordTokenizer()  # this is slow on first run
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

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")


@app.post("/process-image/")
async def process_image(request: Request, file: UploadFile = File(...), similarity: str = "omp", num_words: int = 10):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")

    if num_words > 20:
        raise HTTPException(status_code=400, detail="Too many words")

    try:
        img = Image.open(BytesIO(await file.read()))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open image: {e}")

    ipwt: imgtokens.ImagePatchWordTokenizer = request.app.state.ipwt
    words = ipwt.process_img(img, similarity=similarity, num_words=num_words)
    fig = ipwt.draw_with_plotly(words)
    html = fig.to_html(include_plotlyjs="cdn")
    return HTMLResponse(content=html)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

