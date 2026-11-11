"""Skill lexicon: canonical skill -> (synonyms, family). Used for matching, ATS scoring and honesty checks."""
import re

SKILLS = {
    # languages
    "Python": (["python", "py"], "language"), "SQL": (["sql", "mysql", "postgresql", "postgres", "sqlite", "t-sql"], "database"),
    "Java": (["java"], "language"), "JavaScript": (["javascript", "js", "node.js", "nodejs"], "language"),
    "TypeScript": (["typescript", "ts"], "language"), "C++": (["c++", "cpp"], "language"), "R": (["r programming", "rstudio"], "language"),
    # ML / DL
    "Machine Learning": (["machine learning", "ml", "predictive model", "predictive modelling", "classification", "regression model"], "ml"),
    "Deep Learning": (["deep learning", "neural network", "neural networks", "cnn", "rnn", "lstm"], "dl"),
    "PyTorch": (["pytorch", "torch"], "dl_framework"), "TensorFlow": (["tensorflow", "keras"], "dl_framework"),
    "scikit-learn": (["scikit-learn", "sklearn", "scikit learn"], "ml_lib"), "XGBoost": (["xgboost", "lightgbm", "gradient boosting", "catboost"], "ml_lib"),
    "NLP": (["nlp", "natural language processing", "text classification", "named entity", "sentiment analysis"], "nlp"),
    "Computer Vision": (["computer vision", "opencv", "image classification", "object detection", "yolo", "mediapipe"], "cv"),
    "Transformers": (["transformers", "hugging face", "huggingface", "bert", "fine-tuning", "fine-tuned", "fine tuning"], "llm"),
    # GenAI
    "LLMs": (["llm", "llms", "large language model", "large language models", "gpt-4", "gpt-4o", "gpt", "llama", "openai", "azure openai", "gemini", "claude", "groq"], "llm"),
    "RAG": (["rag", "retrieval-augmented", "retrieval augmented", "retrieval pipeline"], "llm"),
    "LangChain": (["langchain"], "agent_framework"), "LangGraph": (["langgraph"], "agent_framework"),
    "AI Agents": (["ai agent", "ai agents", "agentic", "multi-agent", "tool calling", "function calling", "mcp server", "mcp"], "agent"),
    "Prompt Engineering": (["prompt engineering", "prompt design", "prompting", "prompts"], "llm"),
    "Vector Databases": (["vector database", "vector db", "faiss", "pinecone", "chroma", "chromadb", "pgvector", "weaviate", "qdrant", "milvus"], "vector_db"),
    "Embeddings": (["embedding", "embeddings", "sentence-transformers"], "llm"),
    "LLM Evaluation": (["llm evaluation", "evals", "ragas", "hallucination"], "llm"),
    # backend / mlops
    "FastAPI": (["fastapi"], "backend"), "Flask": (["flask"], "backend"), "Django": (["django"], "backend"),
    "REST APIs": (["rest api", "rest apis", "restful", "api endpoints", "endpoints", "apis"], "backend"),
    "Docker": (["docker", "containerised", "containerized", "dockerfile"], "devops"), "Kubernetes": (["kubernetes", "k8s"], "devops"),
    "CI/CD": (["ci/cd", "github actions", "jenkins", "continuous integration"], "devops"), "Git": (["git", "github", "gitlab"], "devops"),
    "MLOps": (["mlops", "mlflow", "model deployment", "model monitoring", "kubeflow"], "mlops"),
    "AWS": (["aws", "amazon web services", "sagemaker", "lambda", "s3", "ec2"], "cloud"), "Azure": (["azure", "azure blob"], "cloud"),
    "GCP": (["gcp", "google cloud", "vertex ai", "bigquery"], "cloud"),
    # data
    "Pandas": (["pandas", "numpy"], "data"), "Spark": (["spark", "pyspark", "databricks"], "data_eng"),
    "Airflow": (["airflow", "dagster", "prefect"], "data_eng"), "Kafka": (["kafka", "kinesis"], "data_eng"),
    "Snowflake": (["snowflake", "redshift", "data warehouse", "data warehousing"], "data_eng"), "dbt": (["dbt"], "data_eng"),
    "Scala": (["scala"], "language"), "Go": (["golang"], "language"), "Terraform": (["terraform"], "devops"), "ETL": (["etl", "data pipeline", "data pipelines", "elt"], "data_eng"),
    "Power BI": (["power bi", "powerbi", "dax"], "bi"), "Tableau": (["tableau"], "bi"), "Excel": (["excel", "spreadsheets"], "bi"),
    "Statistics": (["statistics", "statistical", "hypothesis testing", "a/b test", "a/b testing", "regression analysis"], "stats"),
    "Data Visualization": (["data visualization", "data visualisation", "dashboards", "dashboard", "matplotlib", "seaborn", "plotly"], "bi"),
    # web
    "React": (["react", "react.js", "reactjs", "next.js", "nextjs"], "frontend"), "Supabase": (["supabase"], "database"),
    "MongoDB": (["mongodb", "mongo"], "database"), "Redis": (["redis"], "database"),
    # soft / product
    "Communication": (["communication", "stakeholder", "stakeholders", "presented", "presentation", "presenting"], "soft"),
    "Product Thinking": (["product thinking", "product sense", "user research", "product metrics"], "soft"),
}

CANON = {k.lower(): k for k in SKILLS}


def _pattern(term: str) -> re.Pattern:
    return re.compile(r"(?<![a-z0-9])" + re.escape(term.lower()) + r"(?![a-z0-9])")


_PATS = {k: [(_pattern(s), s) for s in sorted(set(syns + ([k.lower()] if len(k) > 1 else [])), key=len, reverse=True)] for k, (syns, _) in SKILLS.items()}


def find_skills(text: str) -> dict[str, str]:
    """Return {canonical skill: the surface form found} for every lexicon skill present in text.

    A match that sits inside a longer match of another skill doesn't count: "GitHub Actions" is CI/CD, not also
    Git, and "text classification" is NLP, not also Machine Learning."""
    t = text.lower()
    hits = []                                   # (start, end, skill, surface)
    for k, pats in _PATS.items():
        for p, s in pats:
            hits += [(m.start(), m.end(), k, s) for m in p.finditer(t)]
    out = {}
    for a, b, k, s in sorted(hits, key=lambda h: h[0] - h[1]):          # longest first
        inside_longer = any(a2 <= a and b <= b2 and (b2 - a2) > (b - a) and k2 != k for a2, b2, k2, _ in hits)
        if not inside_longer and k not in out:
            out[k] = s
    return out


def has_exact(text: str, term: str) -> bool:
    return bool(_pattern(term).search(text.lower()))


# Families where knowing one skill is real partial credit for another (PyTorch -> TensorFlow, Power BI ->
# Tableau, Azure -> AWS). Broad families like "language" or "devops" are left out on purpose: Python doesn't make
# you half a Go developer, and Docker doesn't make you half a Kubernetes or Terraform engineer.
TRANSFERABLE = {"dl_framework", "ml_lib", "bi", "cloud", "vector_db", "agent_framework", "data_eng", "backend", "llm", "ml"}


def family(skill: str) -> str:
    return SKILLS.get(skill, ([], skill.lower()))[1]


def canonical(term: str) -> str:
    """Map any term (e.g. 'GPT-4', 'langchain') to a canonical skill name if known."""
    if term.lower() in CANON:
        return CANON[term.lower()]
    t = term.lower()
    best, blen = None, 0
    for k, pats in _PATS.items():          # longest matching surface form wins
        for p, surf in pats:
            if p.search(t) and len(surf) > blen:
                best, blen = k, len(surf)
    return best or term.strip()


# An umbrella JD term is honestly evidenced by a more specific skill: a bullet that uses scikit-learn or XGBoost
# shows machine learning, even though it never says "machine learning". Used only by verify_claims, and only
# for the JD term a rewrite is targeting, so it can't sneak in unrelated skills.
IMPLIED_BY = {
    "Machine Learning": {"scikit-learn", "XGBoost", "Deep Learning", "PyTorch", "TensorFlow", "NLP", "Computer Vision"},
    "Deep Learning": {"PyTorch", "TensorFlow"},
    "LLMs": {"RAG", "Prompt Engineering", "Transformers"},
    "Data Visualization": {"Power BI", "Tableau"},
}
