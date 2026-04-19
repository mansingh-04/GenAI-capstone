# Render Deployment Quick Checklist

## Pre-Deployment ✓

- [ ] Push code to GitHub (`git push origin main`)
- [ ] Verify `render.yaml` exists in repository root
- [ ] Test locally: `uvicorn milestone2/api/main:app --reload`
- [ ] All dependencies in `milestone2/requirements.txt`
- [ ] No hardcoded API keys in code or `.env`

## Environment Variables to Set in Render Dashboard ✓

**Secret Variables** (click "Add Secret" for each):

```
GROQ_API_KEY=your_key_here
NEWSAPI_KEY=your_key_here
API_SECRET_KEY=generate_with: python -c "import secrets; print(secrets.token_urlsafe(32))"
```

**Optional Secret Variables**:

```
HUGGINGFACE_API_KEY=your_key_here  (if using HuggingFace models)
```

## Deployment Steps ✓

1. [ ] Go to https://dashboard.render.com
2. [ ] Click "New +" → "Web Service"
3. [ ] Connect your GitHub repository
4. [ ] Render will auto-detect `render.yaml`
5. [ ] Click "Deploy"
6. [ ] Wait for build to complete (5-10 minutes)
7. [ ] Get your live URL from dashboard

## Verify Deployment ✓

- [ ] Visit `https://your-service-url/` → Should return `{"status": "online", ...}`
- [ ] Visit `https://your-service-url/docs` → Should show Swagger UI
- [ ] Test API endpoint:
  ```bash
  curl -X POST https://your-service-url/api/predict \
    -H "X-API-Key: your_api_secret_key" \
    -H "Content-Type: application/json" \
    -d '{"text": "Test claim"}'
  ```

## Configure Frontend ✓

Update Streamlit `milestone2/.env` with:

```env
UI_BACKEND_HOST=https://your-service-url
API_SECRET_KEY=same_key_as_backend
```

## Post-Deployment ✓

- [ ] Monitor logs in Render dashboard for errors
- [ ] Test all main endpoints
- [ ] Set up auto-redeploy from GitHub (enabled by default)
- [ ] Consider upgrading to paid tier for production use
- [ ] Set up custom domain (optional)

## Troubleshooting Quick Fixes ✓

| Problem        | Solution                                                     |
| -------------- | ------------------------------------------------------------ |
| Build fails    | Check `requirements.txt`, ensure `cd milestone2` in commands |
| 500 errors     | Verify all secret environment variables are set              |
| Slow response  | This is expected on Render free tier (cold starts)           |
| Import errors  | Ensure `sys.path` is set correctly in code                   |
| Database empty | SQLite gets reset on redeploy; use PostgreSQL for production |

## Performance Tips ✓

- Render free tier: Auto-sleeps after 15 min of inactivity (plan upgrade for always-on)
- Long-running requests: Increase timeout in `render.yaml` if needed
- Add caching for frequent API calls
- Keep `log level` as `INFO` in production

## Support Links

- Render: https://render.com/docs
- API Docs at deployment: `https://your-service-url/docs`
- Check logs: Dashboard → Service → Logs tab
