import streamlit as st
from streamlit_chat import message
import numpy as np
import logging
import sys
import os

from llama_index import (
    NotionPageReader, 
    GPTVectorStoreIndex,
    StorageContext,
    ServiceContext,
)
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.vector_stores import SimpleVectorStore

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.faiss import FaissReader
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt

import faiss #Facebookが開発したベクター検索ライブラリ。意味が近い文書を検索できます。


st.markdown('### chatgpt Q&A from Notion')

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

integration_token = st.secrets['NOTION_INTEGRATION_TOKEN']
database_id = st.secrets['NOTION_DATABASE_ID']

# #ログの設定
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

##########################　notionからテキストデータの取得

page_ids = [
    '24dd540daff647299bad85343be9e8ca', #家具市場
    '3feb25502da64fc2a73d7c3392195675', #住宅市場
    'cf1a9bcdfc1e4eff8338947ead54d49c', #クレーム対応
    '1e4cf80a32ba4d4ca1e41aaafd684393', #修理
    '53b4c698eaa84e25944a760ae420533a', #住宅知識
    'c318641702b04a1092757f44907da883', #知識_その他
    'ef935bbaa8134bb38e37cae451242f9f', #知識_家具
    '053b995ddb8f4593aff78a9940a3ffbb', #知識_自社商品
    '0705701977f949649cb6a2358c915b66', #知識_他社商品
    '5bbbf477da6d445284b2c64fd4f54c7b', #商品開発時の考え方、アイデア
    '68f042118592479aba4bf7435288ad71', #人間
    '75f05d9d37d6456ebfddaa0568c5b1b1', #知識_木
    '558cc86e2e274c13b5af5e5c648e57d6' #通達
]

documents = NotionPageReader(integration_token=integration_token).load_data(page_ids=page_ids)

def make_index():
##########################google driveからテキストファイルの取得

    ###########################テキストデータの読み込み・index化

    # Indexの作成
    index = GPTVectorStoreIndex.from_documents(documents)
    # persistでstorage_contextを保存
    index.storage_context.persist(persist_dir="./storage_context")

    ###########################persistでstorage_contextを保存
    st.write(documents)
    st.info('index化完了')

def qa_calc():
    
    #質問の入力
    question = st.text_input('質問を入力してください', key='question')

    if not question:
        st.info('質問を入力してください')
        st.stop()
    
    clear_chat = st.button("履歴消去", key='clear_chat')

    # チャット履歴を保存
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if clear_chat:
        st.session_state["chat_history"] = []
        st.stop()
    
    #ストレージからindexデータの読み込み
    storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore.from_persist_dir(persist_dir="./storage_context"),
        vector_store=SimpleVectorStore.from_persist_dir(persist_dir="./storage_context"),
        index_store=SimpleIndexStore.from_persist_dir(persist_dir="./storage_context"),
    )

    # 実装時点でデフォルトはtext-ada-embedding-002
    #文書テキストを埋め込みベクトルに変換するためのモデル（事前学習済みのニューラルネットワークモデル）
    embed_model = OpenAIEmbedding()

    #埋め込みモデルによるテキスト埋め込みを生成
    #埋め込みベクトルを保持するためのリスト
    docs = []
    #文書のIDとテキストの対応を保持するための辞書
    id_to_text_map = {}
    #文書データを格納しているストレージコンテキストから文書の一覧を取得
    for i, (_, node) in enumerate(storage_context.docstore.docs.items()):
        #文書ノード（node）からテキストを取得
        text = node.get_text()
        #テキストの埋め込みを生成します
        docs.append(embed_model.get_text_embedding(text))
        id_to_text_map[i] = text
    docs = np.array(docs)

    #text-ada-embedding-002から出力されるベクトル長を指定
    d = 1536
    index = faiss.IndexFlatL2(d)
    #Faissにベクトルを登録
    index.add(docs)

    # クエリとFaissから取り出すノード数の設定
    query_text = question
    k = 2

    # questionのベクトル化
    query = embed_model.get_text_embedding(query_text)
    query=np.array([query])

    # Faissからのquestionに近いノードの取り出し2個
    reader = FaissReader(index)
    documents = reader.load_data(query=query, id_to_text_map=id_to_text_map, k=k)

    st.write(f'count_node: {len(documents)}')

    
    # デバッグ用
    llama_debug_handler = LlamaDebugHandler()
    callback_manager = CallbackManager([llama_debug_handler])
    service_context = ServiceContext.from_defaults(callback_manager=callback_manager)

    #Faissで確認した類似したノードを使って、GPTListIndexを作成。
    # index = GPTListIndex.from_documents(documents, service_context=service_context)
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

    QA_PROMPT_TMPL = (
    "私たちは以下の情報をコンテキスト情報として与えます。 \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "あなたはAIとして、この情報をもとに質問を日本語で答えます。前回と同じ回答の場合は同じ回答を行います。: {query_str}\n"
    )
    QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

    query_engine = index.as_query_engine(text_qa_template=QA_PROMPT)

    # テンプレを送る
    response = query_engine.query(question)

    #responseからtextとsourseの取り出し
    response_text = response.response.replace("\n", "")

    def display_chat(chat_history):
        for i, chat in enumerate(reversed(chat_history)):
            if "user" in chat:
                message(chat["user"], is_user=True, key=str(i)) 
            else:
                message(chat["bot"])
                
    # 質問と応答をチャット履歴に追加
    st.session_state["chat_history"].append({"user": question})
    st.session_state["chat_history"].append({"bot": response_text})

    display_chat(st.session_state["chat_history"])

    st.write('sources')
    st.write(response.get_formatted_sources(length=1000))

def main():
    # アプリケーション名と対応する関数のマッピング
    apps = {
        'Q&A': qa_calc,
        'txtのindex化': make_index,
    }
    selected_app_name = st.selectbox(label='項目の選択',
                                                options=list(apps.keys()))


    # 選択されたアプリケーションを処理する関数を呼び出す
    render_func = apps[selected_app_name]
    render_func()

if __name__ == '__main__':
    main()

