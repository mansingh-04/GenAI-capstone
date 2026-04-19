# Deployment Guide: Intelligent News Credibility API on Render

This guide walks you through deploying the FastAPI backend to Render.

## Prerequisites

1. **Render Account**: Sign up at https://render.com
2. **GitHub Repository**: Push your code to GitHub (Render integrates with GitHub)
3. **API Keys**:
   - Groq API Key: Get from https://console.groq.com
   - NewsAPI Key: Get from https://newsapi.org
4. **Git**: Latest version installed locally

## Step 1: Prepare Your Repository

1. Ensure your code is pushed to GitHub:

```bash
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

2. Verify the `render.yaml` file exists in the root of your repository:

```bash
ls render.yaml
```

## Step 2: Connect GitHub to Render

1. Go to https://dashboard.render.com
2. Click **"New +"** → **"Web Service"**
3. Select **"Connect a repository"**
4. Search for your repository and connect it
5. Render will auto-detect the `render.yaml` file

## Step 3: Configure Environment Variables

After connecting your repository, set these environment variables in the Render dashboard:

### Required (Secret) Variables

Go to **Dashboard** → **Your Service** → **Environment** and add these as **Secret** variables:

```
GROQ_API_KEY=your_groq_api_key_here
NEWSAPI_KEY=your_newsapi_key_here
API_SECRET_KEY=generate_a_random_secret_key_here
```

### To Generate a Secure API_SECRET_KEY:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Optional (Secret) Variables

```
HUGGINGFACE_API_KEY=your_huggingface_key_here  # If using HuggingFace models
```

## Step 4: Deploy

1. Click **"Deploy"** on the Render dashboard
2. Render will:
   - Clone your repository
   - Run the `buildCommand` to install dependencies
   - Run the `startCommand` to start the FastAPI server
3. Monitor the build logs in the Render dashboard
4. Once deployed, you'll get a live URL: `https://intelligent-news-credibility-api.onrender.com`

## Step 5: Verify Deployment

Test your API:

```bash
# Health check
curl https://intelligent-news-credibility-api.onrender.com/

# Access API documentation
# Visit: https://intelligent-news-credibility-api.onrender.com/docs
```

## Step 6: Configure Frontend (Streamlit)

Update your Streamlit frontend's `.env` to point to the deployed backend:

```env
UI_BACKEND_HOST=https://intelligent-news-credibility-api.onrender.com
API_SECRET_KEY=same_key_as_backend
```

## Render.yaml Configuration Explained

| Setting           | Purpose                                                     |
| ----------------- | ----------------------------------------------------------- |
| `buildCommand`    | Installs dependencies from requirements.txt                 |
| `startCommand`    | Starts FastAPI server on the assigned port                  |
| `pythonVersion`   | Sets Python version (3.11 recommended)                      |
| `healthCheckPath` | Endpoint for health checks (/)                              |
| `timeout`         | Request timeout in seconds (120s for long-running analyses) |
| `plan`            | Render pricing tier (standard = free tier alternative)      |
| `autoDeploy`      | Auto-redeploy on GitHub push                                |

## Troubleshooting

### Build Fails

- **Check logs**: View full build logs in Render dashboard
- **Missing dependencies**: Ensure `requirements.txt` is up-to-date
- **Python version**: Verify Python 3.11 compatibility

### API Returns 500 Error

- **Check environment variables**: Ensure all secret keys are set
- **Check logs**: Look for errors in the Render service logs
- **Verify keys**: Test API keys independently

### Slow Response Times

- **Cold starts**: Render free tier has slower cold starts. Consider upgrading.
- **API call timeouts**: Render's default timeout might be too short. Check `render.yaml` timeout setting.
- **External API latency**: NewsAPI/Groq API calls may be slow. Add retry logic if needed.

### Database Issues

- Render services don't persist data across restarts by default
- SQLite database (`history.db`) will be reset on redeploy
- Consider using Render PostgreSQL service for production

## Production Best Practices

1. **Use a Database Service**:

   ```yaml
   - type: pserv
     name: credibility-db
     ipAllowList: []
     plan: free
   ```

2. **Set up Environment-Specific Configs**:
   - Development: Free tier on Render
   - Production: Paid tier with more resources

3. **Enable HTTPS**: Render does this automatically

4. **Monitor Logs**:
   - Visit Render dashboard → Logs tab
   - Set up log alerting if available

5. **Update Dependencies Regularly**:
   - Keep `requirements.txt` updated
   - Test locally before pushing to production

## Scaling on Render

- **More instances**: Set `numInstances: 2` (or more) in `render.yaml`
- **Larger plan**: Upgrade from `standard` to `pro` or `platinum`
- **Database**: Switch from SQLite to PostgreSQL for production

## Cost Estimation

- **Web Service**:
  - Free tier: ≤0.1 CPU / 512MB RAM (auto-sleep after 15 min inactivity)
  - Standard: $7/month
  - Pro: $25/month
- **External API calls**: Groq (~$0) + NewsAPI (varies by plan)

For production workloads, expect $20-50/month depending on traffic.

## Monitoring

1. **Render Dashboard**: View real-time metrics, logs, and health status
2. **Set up alerts**: Configure email notifications for deployment failures
3. **Monitor API usage**: Track Groq and NewsAPI usage in their respective dashboards

## Next Steps

- Deploy Streamlit frontend to Streamlit Cloud (free)
- Set up a custom domain pointing to your Render service
- Configure automated backups if using Render PostgreSQL
- Set up CI/CD pipeline for automated testing before deployment

---

**Support Links:**

- Render Docs: https://render.com/docs
- FastAPI Docs: https://fastapi.tiangolo.com
- GitHub Integration: https://render.com/docs/github
