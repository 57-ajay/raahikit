import os
import asyncio
from dotenv import load_dotenv
from livekit.api import AccessToken, VideoGrants

load_dotenv(".env.local")

API_KEY = os.getenv("LIVEKIT_API_KEY")
API_SECRET = os.getenv("LIVEKIT_API_SECRET")

def create_token():
    # We create a token for a human user to join the room "test-room"
    grant = VideoGrants(
        room_join=True,
        room="test-room",
        can_publish=True,
        can_subscribe=True,
    )

    token = AccessToken(API_KEY, API_SECRET) \
        .with_grants(grant) \
        .with_identity("human-tester") \
        .with_name("Human Tester") \
        .to_jwt()

    print("\n=== YOUR TOKEN BELOW ===")
    print(token)
    print("========================\n")

if __name__ == "__main__":
    create_token()
