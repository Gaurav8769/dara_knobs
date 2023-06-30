from pinecone_api import my_index
from function import *


#-----------------------TEXT TO IMAGE--------------------------------------------------------------------------------------------------
text_query = "beaches with sunset"

# Get the caption embedding
query_embedding =  get_single_text_embedding(text_query).tolist()

# Run the query
aa=my_index.query(query_embedding, top_k=1, include_metadata=True)
for match in aa['matches']:
    image_url = match['metadata']['image']
    response = requests.get(image_url, stream=True)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    image.show()
#------------------------------------IMAGE TO TEXT--------------------------------------------------------------------
aa=get_image_from_url('https://9to5google.com/wp-content/uploads/sites/4/2018/03/screenshot-2018-03-06-at-3-08-51-am_polarr.jpg?quality=82&strip=all')
query_embedding = get_single_image_embedding(aa).tolist()
aa=my_index.query(query_embedding, top_k=4, include_metadata=True)
for match in aa['matches']:
    image_url = match['metadata']['image']
    response = requests.get(image_url, stream=True)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    image.show()
