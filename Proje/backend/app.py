"""from fastapi import FastAPI,UploadFile,File
import logging 
import os
import shutil
from fastapi.responses import JSONResponse
logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)
app = FastAPI()
UPLOAD_DIRECTORY = "uploaded_pdfs"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    if not file.filename.endswith('.pdf'):
        return JSONResponse(
            status_code=400,
            content={"message": "Sadece PDF dosyaları kabul edilmektedir."}
        )
    logger.debug(f"File name: {file.filename}")
    logger.debug(f"File content type: {file.content_type}")
    file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return file_path
    

"""
from fastapi import FastAPI, UploadFile, File
import logging
import os
import shutil
from fastapi.responses import JSONResponse
import uuid
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


load_dotenv()
vector_store = None 
embeddings=None
pinecone_apikey = os.getenv("PINECONE_API_KEY")
pinecone_env=os.getenv("PINECONE_ENVIRONMENT","gcp-starter")
groq_aPi_key=os.getenv("GROQ_API_KEY")
# Loglama ayarları
logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)

app = FastAPI()
# PDF dosyalarını kaydedeceğimiz klasör
UPLOAD_DIRECTORY = "uploaded_pdfs"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    global vector_store
    # Dosya türü kontrolü
    if not file.filename.endswith('.pdf'):
        logger.warning(f"Geçersiz dosya türü: {file.content_type}")
        return JSONResponse(
            status_code=400,
            content={"message": "Sadece PDF dosyaları kabul edilmektedir."}
        )
    
    # Log bilgileri
    logger.debug(f"File name: {file.filename}")
    logger.debug(f"File content type: {file.content_type}")
    
    # Dosya adını güvenli hale getir (aynı isimli dosyaların üzerine yazılmasını önlemek için)
    # Orijinal dosya adını korumak için, UUID ön eki ekler
    safe_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIRECTORY, safe_filename)
    
    # Dosyayı kaydet
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Dosya başarıyla kaydedildi: {file_path}")
        
        # Mutlak yol döndür (tam dosya yolu)
        absolute_path = os.path.abspath(file_path)   
    except Exception as e:
        logger.error(f"Dosya yükleme hatası: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": f"Dosya yükleme hatası: {str(e)}"}
        )
     
    pdf_yol=absolute_path
    print(f"PDF dosyası yolu: {pdf_yol}")
    # PDF belgesini yükle
    print("PDF yükleniyor...")
    try:
        loader = PyPDFLoader(pdf_yol)
        documents = loader.load()
        print(f"PDF yüklendi. {len(documents)} sayfa bulundu.")
    except Exception as e:
        print(f"PDF yüklenirken hata: {e}")
        raise
    # PDF'yi küçük parçalara ayır
    text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)
    # Pinecone vektör veritabanına belgeleri yükle
    embeddings = PineconeEmbeddings(model="multilingual-e5-large",api_key=pinecone_apikey)
    print("Vektör veritabanı oluşturuluyor...")

    try:
        vector_store=PineconeVectorStore.from_documents(
            embedding=embeddings,
            documents=docs,
            index_name="ragproje"
        )
        
        print("Belgeler vektör veritabanına başarıyla yüklendi.")
        
        
        
    except Exception as e:
        print(f"Vektör veritabanı oluşturulurken hata: {e}")
        raise
    
   
    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # app.py
    

# Bu satırı vector_store tanımlamadan önce ekleyin
embeddings = PineconeEmbeddings(model="multilingual-e5-large", api_key=pinecone_apikey,)
vector_store = PineconeVectorStore(index_name="ragproje", embedding=embeddings)
retriever = vector_store.as_retriever(search_type="similarity_score_threshold",
                                      search_kwargs={'score_threshold': 0.5})
llm = ChatGroq(model="llama3-70b-8192"
               ,api_key=groq_aPi_key)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Bir sohbet geçmişi ve sohbet geçmişindeki bağlama atıfta bulunabilecek en son kullanıcı sorusu verildiğinde, "
               "sohbet geçmişi olmadan anlaşılabilecek bağımsız bir soru oluşturun. Ardından, bu soruyu İngilizce'ye çevirin. "
               "Çıktı olarak sadece İngilizce çeviriyi verin."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Soru cevaplama görevleri için bir asistansınız. "
               "Soruyu cevaplamak için aşağıdaki alınan bağlam parçalarını kullanın. "
               "Kullanıcıya verdiğiniz cevabı Türkçe olarak yazın. "
               "Eğer cevabı bilmiyorsanız, bilmediğinizi söyleyin. "
               "Cevapları kısa tutun (maksimum 3 cümle). \n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Gelen istek modeli
class Message(BaseModel):
    role: str
    content: str

class AskRequest(BaseModel):
    input: str
    chat_history: List[Message]

@app.post("/ask")
async def ask_question(request: AskRequest):
    # Chat history'i LangChain formatına dönüştür
    formatted_history = []
    for msg in request.chat_history:
        if msg.role == "user":
            formatted_history.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            formatted_history.append(AIMessage(content=msg.content))

    result = rag_chain.invoke({
        "input": request.input,
        "chat_history": formatted_history
    })

    return {"answer": result["answer"]}
