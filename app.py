"""
Compatibility entrypoint for legacy local scripts (benchmark.py).

The canonical deployed FastAPI app is server.app:app.
Docker/HF Space runs: uvicorn server.app:app --host 0.0.0.0 --port 7860
This wrapper allows `python app.py` to work for local testing.
"""

import os

from server.app import app


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
