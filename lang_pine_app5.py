import streamlit as st
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LangChainPinecone
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# アプリのタイトルと説明
st.set_page_config(page_title="港湾の施設の技術上の基準 ChatBot", page_icon="💬")
st.title(" 📘港湾の施設の技術上の基準📖 ChatBot💬")
st.markdown("""
このアプリは、会話履歴を考慮したRetrieval-Augmented Generation (RAG) チャットボットです。
Pinecone のベクトルストアと OpenAI の GPT を組み合わせ、過去の履歴を踏まえて対話を行います。
""")

# サイドバーの設定
st.sidebar.title("🛠 パラメータ調整")
temperature = st.sidebar.slider("Temperature (生成の多様性)", 0.0, 1.0, 0.7, step=0.1)
top_k = st.sidebar.slider("Top-k Documents (検索時の上位文書数)", 1, 10, 3)

# 環境変数の読み込み
load_dotenv()

# 環境変数からAPIキーを取得
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
index_name = os.getenv("PINECONE_INDEX_NAME")

# Pinecone の初期化
if pinecone_api_key:
    try:
        pc = Pinecone(api_key=pinecone_api_key)
        if index_name not in [index.name for index in pc.list_indexes()]:
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=pinecone_env)
            )
            st.success(f"新しいインデックス '{index_name}' が作成されました。")
        else:
            st.success(f"インデックス '{index_name}' が正常にロードされました。")
    except Exception as e:
        st.error(f"Pinecone 初期化エラー: {e}")
else:
    st.error("Pinecone API Key が指定されていません。")

# セッションステートの初期化
if "history" not in st.session_state:
    st.session_state["history"] = []  # 初期化

# ユーザーの質問入力ボックス
st.subheader("質問を入力してください：")
user_input = st.text_input("質問", placeholder="例: 最近のAI技術のトレンドは？")

if user_input and openai_api_key and pinecone_api_key:
    # OpenAI Embeddingsの初期化
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Pinecone のインデックスを読み込む
    vector_store = LangChainPinecone.from_existing_index(
        index_name=index_name, embedding=embeddings
    )

    # OpenAI LLM の初期化
    llm = OpenAI(openai_api_key=openai_api_key, temperature=temperature, max_tokens=1500)

    # これまでの会話履歴をまとめたプロンプトを作成
    conversation_history = "\n".join(
        [f"ユーザー: {user}\nアシスタント: {assistant}" for user, assistant in st.session_state["history"]]
    )

    # 現在の質問を履歴に追加してプロンプトを生成
    if conversation_history:
        prompt = f"{conversation_history}\nユーザー: {user_input}\nアシスタント:"
    else:
        prompt = f"ユーザー: {user_input}\nアシスタント:"

    # RetrievalQA の作成
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, chain_type="stuff", return_source_documents=True  # ソース文書も返すように設定
    )

    # 質問に対する回答と参照された文書を取得
    result = qa_chain({"query": prompt})

    response = result['result']  # 回答部分
    source_documents = result['source_documents']  # 参照された文書

    # チャット履歴に質問と回答を追加
    st.session_state["history"].append((user_input, response))

    # チャット履歴の表示
    with st.expander("💬 チャット履歴", expanded=True):
        for i, (user, assistant) in enumerate(st.session_state["history"]):
            st.markdown(f"**ユーザー ({i+1})**: {user}")
            st.markdown(f"**アシスタント**: {assistant}")

    # 参照された文書の表示
    st.subheader("🔍 参照された文書の一部:")
    for doc in source_documents:
        # メタデータから文書名とページ番号を取得
        source_name = doc.metadata.get("source", "文書名不明")
        page_number = doc.metadata.get("page", "ページ番号不明")
        
        st.markdown(f"**文書名**: {source_name}")
        st.markdown(f"**ページ番号**: {page_number}")
        st.markdown(f"**文書内容**: {doc.page_content}")  # 参照された文書の内容を表示
