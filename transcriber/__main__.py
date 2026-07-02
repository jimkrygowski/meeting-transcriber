import uvicorn

uvicorn.run("transcriber.server:app", host="127.0.0.1", port=8484)
