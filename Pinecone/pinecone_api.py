import pinecone
from langchain.vectorstores import Pinecone
pinecone.init(
    api_key="228d3258-50c7-4214-90b9-58d58854bd01",
    environment="northamerica-northeast1-gcp"
)
index_name = "image-text"
my_index = pinecone.Index(index_name = index_name)
