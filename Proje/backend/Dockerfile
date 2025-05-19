# Python imajını kullan
FROM python:3.10-slim

# Sistem güncelleme ve bağımlılıkları yükle
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizini oluştur
WORKDIR /app

# Gerekli dosyaları kopyala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Tüm proje dosyalarını kopyala
COPY . .

# PDF klasörü oluştur
RUN mkdir -p uploaded_pdfs

# Uygulamayı başlat
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
