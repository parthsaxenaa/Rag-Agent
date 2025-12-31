# How to Host on Render ☁️

Follow these steps to deploy your bot for free on Render.com.

## 1. Prepare your Repository
1. Create a **GitHub** (or GitLab) repository.
2. Upload all files from this folder to the repository.
   - **Crucial**: Make sure to upload/commit the `faiss_index` folder! 
   - *Why?* Creating the index takes a long time and might hit rate limits. Uploading the pre-made index makes the live app start instantly.
   - Also ensure `complete data.pdf` is uploaded if you want it to be processed again in the future.
   - **Do NOT** upload your `.env` file (it contains your secret key).

## 2. Deploy to Render
1. Sign up/Log in to [Render.com](https://render.com).
2. Click **New +** -> **Web Service**.
3. Connect your GitHub repository.
4. **Configuration**:
   - **Name**: `vellicate-bot` (or any name)
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. **Environment Variables** (Advanced Settings):
   - Scroll down to "Environment Variables".
   - Click "Add Environment Variable".
   - Key: `GOOGLE_API_KEY`
   - Value: *Paste your actual Gemini API Key here*
6. Click **Create Web Service**.

## 3. Done!
Render will build your app and give you a URL (e.g., `https://vellicate-bot.onrender.com`). It might take a few minutes to start.
