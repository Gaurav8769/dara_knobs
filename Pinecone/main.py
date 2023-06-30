from function import *
import pinecone
from langchain.vectorstores import Pinecone
from pinecone_api import my_index
# in this dataset is prepare and upsert in document
def main():
    df=pd.read_excel('/content/photos-all.xlsx')
    df=df.dropna(axis=0)
    df['Path']=df['Path'].apply(convert_gs_url_to_https)
    df['Path']=df['Path'].apply(change_url)
    df['image']=df['Path'].apply(get_image_from_url)
    df=df.dropna(axis=0)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Define the model ID
    model_ID = "openai/clip-vit-base-patch32"
# Get model, processor & tokenizer
    model, processor, tokenizer = get_model_info(model_ID, device)
    
    df['image_embeddings'] = df['image'].apply(get_single_image_embedding)
    df['Joined'] = df['Metadata1'].astype(str) +","+ df['Metadata2'].astype(str)+"," + df['Metadata3'].astype(str)+","+df['Unnamed: 4'].astype(str)+","+df['Unnamed: 4'].astype(str)
    df = get_all_text_embeddings(df, "Joined")
    df["vector_id"] = df.index
    df["vector_id"] = df["vector_id"].apply(str)
    final_metadata = []
    for index in range(len(df)):
        final_metadata.append({
            'ID':  index,
            'caption': df.iloc[index].Joined,
            'image': df.iloc[index].Path })
   
    image_IDs = df.vector_id.tolist()
    image_embeddings = [arr.tolist() for arr in df.image_embeddings.tolist()]
    data_to_upsert = list(zip(image_IDs, image_embeddings, final_metadata))
    my_index.upsert(data_to_upsert)


main()
#--------------------------------------TESTING--------------------------------------------------------------
