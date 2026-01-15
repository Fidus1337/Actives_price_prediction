from dotenv import load_dotenv
import os

## ENV VARIABLES

load_dotenv("dev.env")

BASE_URL = os.getenv("COIN_GLASS_ENDPOINT")

