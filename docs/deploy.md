# Deploying to Streamlit Community Cloud

The app runs on Streamlit Community Cloud from the `main` branch; every push redeploys it. Hosted mode was also tested locally with the same secrets.

1. Push the repository to GitHub.
2. On share.streamlit.io choose **Create app**, pick the repository and branch, and set the main file to `app.py`. Choose Python 3.12.
3. Under **Advanced settings, Secrets**, paste:

   ```toml
   HOSTED = "true"
   OPENAI_API_KEY = "..."
   ANTHROPIC_API_KEY = "..."
   ```

4. Deploy.

| Secret | What it does |
|---|---|
| `HOSTED` | Turns on hosted mode: the access code, the limits below, and no writes to disk. |
| `ACCESS_CODE` | Optional. Leave it out and the Your data tab is open to everyone. Set it and uploading your own files asks for the code (compared in constant time); "Try the second sample" and the other tabs stay open, since they make no API calls. |
| `DAILY_RUN_LIMIT` | Optional, default 40. Runs on uploaded files per day across all visitors, to bound API spend when there is no access code. |
| `OPENAI_API_KEY` | Optional. Embeddings for uploaded files. The local embedding model is not installed on the host. |
| `ANTHROPIC_API_KEY` | Optional. Column mapping and matching for uploaded files. |

**Bring your own keys.** If either key is left out of the secrets, the Your data tab shows a password field for it, with the line "To run your own files, add your own API keys. They are used for this session only and are not stored." A visitor's keys are kept in their own session's memory, handed to the API client for the length of a call, and never written to disk, logged, put in the URL or shown in a message. A rejected key gives a message that names the provider ("The OpenAI API key was rejected…") and not the key. Both built-in samples work with no keys from either source, and the hosted limits below apply whichever keys are used.

The second sample needs neither key: its Claude answers (`cache/llm_sample/`) and its embeddings (`cache/sample_embeddings.npz`) are committed, so "Try the second sample" makes no API calls.

Keys are read from Streamlit secrets first, then from the environment or `.env.local`, and are passed straight to the API clients; they are never printed or logged. `.streamlit/secrets.toml` is gitignored.

Hosted limits, each shown on the page when it is hit: 5 MB per file, 2,000 purchase lines per run, 150 Claude calls per run (the largest lines go first; the rest wait in Review), 3 runs per browser session, and 40 runs on uploaded files per day across all visitors.

What is kept where on the host: each visitor gets their own session. Uploaded files, the matched results, review decisions and slider settings live in that session's memory on the server and are visible only to that visitor; a reload starts fresh. Every visitor starts from the same built-in sample, which no one can change. Nothing derived from uploaded text is written to disk, and there is no shared decisions file. Uploaded text is sent to OpenAI and Anthropic for matching, as the tab says.

Dependencies: Community Cloud uses `uv.lock` when a repository has one, ahead of `requirements.txt`. The default install in both is the same small set, without torch or sentence-transformers. Those, with scikit-learn, are in the optional `local` extra, needed only to rebuild the results: `uv sync --extra local`.
