# this is main file, for test code see get_token.py file
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from livekit import api
from starlette.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import cast

load_dotenv(".env.local")

app = FastAPI()

app = FastAPI(
    middleware=[
        Middleware(
            cast(type, CORSMiddleware),
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ]
)

class TokenRequest(BaseModel):
    room_name: str
    participant_name: str

@app.post("/getToken")
async def get_token(req: TokenRequest):
    token = api.AccessToken(
        os.getenv("LIVEKIT_API_KEY"),
        os.getenv("LIVEKIT_API_SECRET")
    ).with_grants(
        api.VideoGrants(
            room_join=True,
            room=req.room_name,
            can_publish=True,
            can_subscribe=True
        )
    ).with_identity(req.participant_name).with_name(req.participant_name)

    return {"token": token.to_jwt()}

# To run: uvicorn server:app --host 0.0.0.0 --port 3000
