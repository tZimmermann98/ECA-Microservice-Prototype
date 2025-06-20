# Embodied Conversational Agent (ECA) Microservice Prototype

This repository contains the source code for a multi-service prototype of an **Embodied Conversational Agent (ECA)**.
The architecture is designed to be **modular and scalable**, separating concerns for perception, conversation, vocalization, and embodiment into individual microservices, orchestrated by a central engine.

---

## 🧰 Prerequisites

Before you begin, ensure you have the following installed:

* [Docker](https://www.docker.com/products/docker-desktop)
* Docker Compose (included with Docker Desktop)

> 💡 **Recommendation**: Configure Docker Desktop to use a significant portion of your system’s resources, especially RAM (e.g., 16GB+), as the Perception and Transcription models are memory-intensive.

---

## 🚀 First-Time Setup

### 1. Clone the Repository

```bash
git clone https://github.com/tZimmermann98/ECA-Microservice-Prototype
cd ECA-Microservice-Prototype
```

---

### 2. Configure Local Hostname

The system uses a reverse proxy with SSL at `https://ai-face-to-face.local`.

#### On macOS / Linux:

```bash
sudo nano /etc/hosts
```

Add this line at the end:

```
127.0.0.1   ai-face-to-face.local
```

#### On Windows:

1. Open Notepad as Administrator.
2. Open the file:
   `C:\Windows\System32\drivers\etc\hosts`
3. Add:

```
127.0.0.1   ai-face-to-face.local
```

---

### 3. Create SSL Certificate for Local Development

The NGINX reverse proxy requires an SSL certificate to serve the application over HTTPS locally.

Run the following OpenSSL command from the project root. This will create the nginx-selfsigned.key and nginx-selfsigned.crt files in the nginx/certs/ directory.

```Bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
-keyout ./nginx/certs/nginx-selfsigned.key \
-out ./nginx/certs/nginx-selfsigned.crt \
-subj "/C=DE/ST=NRW/L=Muenster/O=Local Dev/CN=ai-face-to-face.local"
```

🔐 Your browser will show a security warning because the certificate is self-signed. You can safely accept the warning to proceed to the site.

---

### 4. Create and Configure the `.env` File

1. In the project root, create a file named `.env`.
2. Copy the content from `example.env` and replace all placeholders with your actual credentials.

#### Example `.env` structure:

```env
OPENAI_API_KEY=your_openai_api_key
FISHAUDIO_API_KEY=your_fishaudio_api_key
HEYGEN_API_KEY=your_heygen_api_key
DB_NAME=eca_prototype
DB_USER=admin
DB_PASSWORD=your_super_secret_password
S3_ACCESS_KEY_ID=minioadmin
S3_SECRET_ACCESS_KEY=minioadmin
OAUTH_COOKIE_SECRET=your_long_random_cookie_secret_string
OAUTH_CLIENT_ID=your_oauth_client_id
OAUTH_CLIENT_SECRET=your_oauth_client_secret
```

---

### 5. Update Admin User Email and Avatar/Voice IDs

Before starting the application:

* **Change the admin email** in the `init_db.py` script to the one you’ll use to log in via your OIDC provider (e.g., a Gmail address for Google login).
* **Replace the placeholder `provider_voice_id` and `provider_avatar_id`** with the actual values from your [FishAudio](https://fish.audio) and [HeyGen](https://heygen.com) accounts.

```python
# In `init_db.py`
admin_user = User(
    user_email="your_email@example.com",  # ← Update this!
    ...
)

audio_prov = AudioProvider(
    ...
    provider_voice_id="your_real_voice_id",  # ← Update this!
)

video_prov = VideoProvider(
    ...
    provider_avatar_id="your_real_avatar_id",  # ← Update this!
)
```

You can find these IDs in your FishAudio and HeyGen dashboards after account creation.

---

## ▶️ Running the Application

From the project root, start all services with:

```bash
docker compose up --build
```

> 🔧 The `--build` flag forces Docker to rebuild the containers. Use this on the first run and after any code changes.

> 🕒 **Note**: The first run may take some time as Docker downloads images and AI model weights (DeepFace, Whisper, SpeechBrain). Future runs will be much faster due to caching.

---

## 🌐 Accessing the Services

* **Main Frontend**: [https://ai-face-to-face.local](https://ai-face-to-face.local)
  *(OIDC login required — use the email set in `init_db.py`!)*

---

## 🛑 Stopping the Application

* Stop all services with:

```bash
Ctrl + C
```

* To remove containers and network:

```bash
docker compose down
```

* To also remove the database and cached model volumes:

```bash
docker compose down -v
```

---

## 📝 Notes for New Users

* After first start:

  * Create accounts at [OpenAI](https://openai.com), [FishAudio](https://fish.audio), and [HeyGen](https://heygen.com).
  * Set the respective API keys in your `.env` file.
  * Update the `init_db.py` script as described above for:

    * Your **OIDC login email**
    * Your **FishAudio voice ID**
    * Your **HeyGen avatar ID**

This ensures the demo avatar and authentication work correctly from the beginning.
