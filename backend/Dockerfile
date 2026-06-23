# =============================================================================
#  ClearClaims Hail Report API — container image
#  Python + the system libraries WeasyPrint needs (Pango/Cairo) + brand fonts.
#  (eccodes/cfgrib install via pip wheels, which bundle the GRIB binary.)
# =============================================================================
FROM python:3.11-slim-bookworm

# --- System libraries for WeasyPrint (PDF text layout) + networking certs ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpango-1.0-0 libpangocairo-1.0-0 libcairo2 \
        libgdk-pixbuf-2.0-0 libffi8 shared-mime-info \
        fonts-dejavu-core curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Brand fonts (DM Serif Display + Outfit) embedded in the PDF -------------
RUN mkdir -p /app/fonts \
 && curl -fsSL -o /app/fonts/DMSerifDisplay-Regular.ttf \
      "https://github.com/google/fonts/raw/main/ofl/dmserifdisplay/DMSerifDisplay-Regular.ttf" \
 && curl -fsSL -o "/app/fonts/Outfit[wght].ttf" \
      "https://github.com/google/fonts/raw/main/ofl/outfit/Outfit%5Bwght%5D.ttf" \
 || echo "WARNING: brand font download failed; PDF will use a fallback font."

# --- Python dependencies -----------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Application code ---------------------------------------------------------
COPY hail_core.py pipeline.py app.py ./

ENV FONT_DIR=/app/fonts \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

# Render provides $PORT; default to 8000 for local runs.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
