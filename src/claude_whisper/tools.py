import base64
import io

import mss
from claude_agent_sdk import create_sdk_mcp_server, tool
from PIL import Image

MAX_DIMENSION = 1024
JPEG_QUALITY = 60


def _capture_and_encode() -> str:
    """Capture the screen and return a base64-encoded JPEG string, downscaled to stay under SDK buffer limits."""
    with mss.mss() as sct:
        monitor = sct.monitors[0]
        sct_img = sct.grab(monitor)
        image = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

    # Downscale large screenshots so the JSON payload stays under the 1MB SDK buffer limit
    # w, h = image.size
    # if max(w, h) > MAX_DIMENSION:
    #     scale = MAX_DIMENSION / max(w, h)
    #     image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")


@tool(
    name="screenshot",
    description=(
        "Take a screenshot of the user's current screen and return it as an image. "
        "Use this tool whenever the user asks you to look at, read, gather, grab, extract, or take "
        "something from their screen, desktop, window, or display. The returned image can be analyzed "
        "to answer questions about what is visible on screen."
    ),
    input_schema={},
)
async def screenshot(args):
    """Capture the current screen and return it as a base64-encoded image for analysis."""
    try:
        image_data = _capture_and_encode()
        return {
            "content": [
                {"type": "image", "data": image_data, "mimeType": "image/jpeg"},
            ],
        }
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Screenshot error: {e}"}],
            "is_error": True,
        }


claude_whisper_mcp_server = create_sdk_mcp_server(
    name="utils",
    tools=[screenshot],
)
