TIER1_RETRIEVAL = {
    "faiss", "pinecone", "qdrant", "milvus", "weaviate", "chroma", "pgvector",
    "sentence-transformers", "sentence_transformers", "vector search",
    "dense retrieval", "semantic search", "embedding", "ann",
    "approximate nearest neighbor", "colbert", "bi-encoder", "cross-encoder",
    "vector database", "vector db", "neural search", "hybrid search"
}

TIER2_NLP_IR = {
    "nlp", "bm25", "elasticsearch", "information retrieval", "text ranking",
    "tfidf", "tf-idf", "inverted index", "lucene", "solr", "text search",
    "ranking model", "learning to rank", "ltr", "ndcg", "mrr", "recall",
    "query understanding", "passage retrieval"
}

TIER2_RECSYS = {
    "recommendation system", "collaborative filtering", "matrix factorization",
    "two-tower", "retrieval ranking", "a/b testing", "experimentation",
    "personalization", "candidate generation", "ranking pipeline"
}

TIER3_LLM = {
    "llm", "large language model", "rag", "retrieval augmented", "fine-tuning",
    "lora", "qlora", "bert", "transformers", "huggingface", "prompt engineering",
    "langchain", "openai api", "gpt", "instruction tuning"
}

TIER3_MLOPS = {
    "mlflow", "weights and biases", "wandb", "bentoml", "triton", "mlops",
    "model serving", "feature store", "kubeflow", "model monitoring",
    "experiment tracking", "data versioning"
}

NEGATIVE_CV_SPEECH = {
    "yolo", "object detection", "image classification", "opencv",
    "resnet", "efficientnet", "speech recognition", "asr", "tts",
    "text-to-speech", "whisper asr", "wav2vec", "pose estimation",
    "face detection", "image segmentation", "action recognition"
}

WITCH_COMPANIES = {
    "tcs", "wipro", "infosys", "hcl", "cognizant", "accenture",
    "tech mahindra", "mphasis", "hexaware", "l&t infotech", "ltimindtree"
}

PRODUCT_STARTUPS_INDIA = {
    "razorpay", "zomato", "swiggy", "paytm", "cred", "flipkart",
    "meesho", "ola", "phonepe", "zepto", "blinkit", "nykaa",
    "policybazaar", "freshworks", "zoho", "dream11", "unacademy",
    "byju's", "upgrad", "groww", "slice", "sarvam ai", "krutrim",
    "rephrase.ai", "yellow.ai", "haptik", "verloop.io", "aganitha",
    "mad street den", "glance", "inmobi", "locobuzz", "niramai",
    "pharmeasy", "vedantu", "genpact ai", "linkedin india",
    "amazon india", "google india", "meta india", "microsoft india",
    "netflix india", "apple india"
}

PREFERRED_CITIES = {
    "hyderabad", "pune", "mumbai", "delhi", "noida", "gurgaon", "gurugram",
    "bengaluru", "bangalore", "delhi ncr", "new delhi"
}

NOTICE_MULTIPLIERS = {
    30: 1.0,
    60: 0.85,
    90: 0.70,
    120: 0.55,
    999: 0.40
}

ALL_JD_SKILLS = TIER1_RETRIEVAL.union(TIER2_NLP_IR).union(TIER2_RECSYS).union(TIER3_LLM)
