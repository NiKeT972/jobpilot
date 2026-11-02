"""Demo inputs. The JD is illustrative (written for this demo) and the candidate is fictional."""

JD_AI = """AI Engineer (GenAI) — Bengaluru
About the role: You'll build LLM-powered agents that automate merchant support and onboarding.

What you'll do:
- Design and ship AI agents using LangChain and LangGraph
- Build RAG pipelines over internal docs with a vector database
- Expose models through FastAPI services and deploy them with Docker on AWS
- Set up LLM evaluation for accuracy and hallucination
- Work with product and ops teams to define success metrics

Requirements:
- Strong Python and SQL
- Hands-on experience with LLMs, prompt engineering and RAG
- Experience with LangChain or LangGraph and building AI agents
- REST APIs with FastAPI, Docker, Git
- 0-2 years of experience

Nice to have:
- AWS, Kubernetes, MLOps
- Fine-tuning with Hugging Face Transformers
- Power BI or other dashboards
"""

JD_DATA_ENG = """Senior Data Engineer
Requirements:
- 5+ years building ETL pipelines with Spark and Airflow
- Kubernetes, Scala, Kafka, data warehousing on Snowflake
- Strong Java
Nice to have: dbt, Terraform
"""

RESUME = """Aarav Shah
B.Tech Computer Science, 2026 · Mumbai

Experience
AI/ML Intern, Fintech startup (May–Jul 2026)
- Built a customer-support chatbot on Azure OpenAI GPT-4 with a retrieval pipeline over 1,200 policy PDFs stored in FAISS
- Wrote FastAPI endpoints for the chatbot and containerised the service with Docker
- Built an MCP server exposing internal tools so the assistant could look up order status and raise tickets
- Cut average response time from 9s to 3s by caching embeddings and tuning chunk sizes

Projects
- Stock volatility dashboard: GARCH and EWMA models in Python with pandas, visualised in Power BI
- Hand-gesture controller using OpenCV and MediaPipe, published as an IEEE conference paper
- Kidney disease classifier with scikit-learn and XGBoost reaching 94% accuracy on 400 records
- Personal productivity app in React and Supabase with row-level security

Skills
- Python, SQL (PostgreSQL), JavaScript, Git, GitHub, prompting, pandas, Power BI
"""

SAMPLES = {
    "ai": {"label": "AI Engineer (good match)", "company": "Razorpay", "jd": JD_AI, "resume": RESUME},
    "de": {"label": "Senior Data Engineer (poor match)", "company": "Flipkart", "jd": JD_DATA_ENG, "resume": RESUME},
}
