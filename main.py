import streamlit as st
from streamlit_chat import message
import numpy as np
import logging
import sys
import os
import requests
import json
# from notion_client import Client
import datetime
from PIL import Image

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
database_id_nondomain = st.secrets['NOTION_DATABASE_ID_NONDOMAIN']

# #ログの設定
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

##########################　notionからテキストデータの取得 関数内で使用
def make_doc():
    
    ############　page idの自動取得
    url = f"https://api.notion.com/v1/search"

    headers = {
        "accept": "application/json",
        "Notion-Version": "2022-06-28",
        "Authorization": f"Bearer {integration_token}"
    }

    json_data = {
        # タイトルを検索できる
        #"query": "ブログ",
        # 絞り込み(データベースだけに絞るなど)
        #"filter": {
        #    "value": "database",
        #    "property": "object"
        #},
        # ソート順
        "sort": {
            "direction": "ascending",
            "timestamp": "last_edited_time"
        }
    }

    response = requests.post(url, json=json_data, headers=headers)
    j_response = response.json()
    j_response1 = j_response['object']
    j_response2 = j_response['results']

    #page_idの取得
    page_ids = []
    for page in j_response2:
        page_id = page["id"]
        page_ids.append(page_id)

    documents = NotionPageReader(integration_token=integration_token).load_data(page_ids=page_ids)

    return documents

def get_nearlynode():
    st.markdown('### ノード収集')

    with st.form('入力'):
        #質問の入力
        theme = st.text_input('情報を集めるテーマを入力', key='question')
        question = theme + 'に関連する情報を集めてください。'
        num_node = st.number_input('抽出ノード数を指定してください', value=8, key='num_node')

        submitted = st.form_submit_button('submitted')
    
    if submitted:

        #ストレージからindexデータの読み込み
        @st.cache_data(ttl=datetime.timedelta(hours=1))
        def read_storage():
            storage_context = StorageContext.from_defaults(
                docstore=SimpleDocumentStore.from_persist_dir(persist_dir="./storage_context"),
                vector_store=SimpleVectorStore.from_persist_dir(persist_dir="./storage_context"),
                index_store=SimpleIndexStore.from_persist_dir(persist_dir="./storage_context"),
            )
            return storage_context
        
        storage_context = read_storage()

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

        #全ノード数表示
        st.write(f'count_allnode: {len(docs)}')

        #text-ada-embedding-002から出力されるベクトル長を指定
        d = 1536
        index = faiss.IndexFlatL2(d)
        #Faissにベクトルを登録
        index.add(docs)

        # クエリとFaissから取り出すノード数の設定
        k = num_node

        # questionのベクトル化
        query = embed_model.get_text_embedding(question)
        query=np.array([query])

        # Faissからのquestionに近いノードの取り出し2個
        reader = FaissReader(index)
        documents = reader.load_data(query=query, id_to_text_map=id_to_text_map, k=k)

        #抽出ノード数の表示
        st.write(f'count_node: {len(documents)}')

        #ドキュメントオブジェクトからテキストデータを抽出
        texts = []
        for document in documents:
            text = document.text
            texts.append(text)

        #リスト内のテキストを結合
        j_text = ''.join(texts)

        #質問用のQAプロンプトを生成
        QA_PROMPT_TMPL = \
            f'私たちは以下の情報をコンテキスト情報として与えます。\
            ---------------------\
            ### コンテキスト情報 ###\
            {j_text}\
            --------------------\
            \
            あなたは家族の時間、絆を重視する優秀なマーケターです。\
            飛騨の木の家具を家具の販売店で販売するための戦略、戦術を立案しようとしています。\
            \
            以下の質問に対し日本語で答えます。前回と同じ回答の場合は同じ回答を行います。\
            \
            ###質問###\
            1. 上記のコンテキスト情報をもとに\
                {theme}に関する情報を抽出して、テーマ毎にまとめてください。'
            

        #コピー画像
        image = Image.open('書類複製.jpeg')
        st.image(image, width=30)

        st.code(QA_PROMPT_TMPL, language='None')



def make_index():
    ##############notionからテキストデータの取得
    documents = make_doc()

    ###########################テキストデータの読み込み・index化

    # Indexの作成
    index = GPTVectorStoreIndex.from_documents(documents)
    # persistでstorage_contextを保存
    index.storage_context.persist(persist_dir="./storage_context")

    st.info('index化完了')

def check_doc():
    documents = make_doc()
    st.write(documents)

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
    @st.cache_data(ttl=datetime.timedelta(hours=1))
    def read_storage():
        storage_context = StorageContext.from_defaults(
            docstore=SimpleDocumentStore.from_persist_dir(persist_dir="./storage_context"),
            vector_store=SimpleVectorStore.from_persist_dir(persist_dir="./storage_context"),
            index_store=SimpleIndexStore.from_persist_dir(persist_dir="./storage_context"),
        )
        return storage_context
    
    storage_context = read_storage()

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

    #質問用のQAプロンプトを生成
    QA_PROMPT_TMPL = (
        "私たちは以下の情報をコンテキスト情報として与えます。 \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "あなたはAIとして、この情報をもとに質問を日本語で答えます。前回と同じ回答の場合は同じ回答を行います。: {query_str}\n"
    )
    qa_prompt = QuestionAnswerPrompt(QA_PROMPT_TMPL)

    ###############################################################tets
    #回答要求用のプロンプトを生成
    REFINE_PROMPT = ("元の質問は次のとおりです: {query_str} \n"
        "既存の回答を提供しました: {existing_answer} \n"
        "既存の答えを洗練する必要があります \n"
        "(必要な場合のみ)以下にコンテキストを追加します。 \n"
        "------------\n"
        "{context_msg}\n"
        "------------\n"
        "新しいコンテキストを考慮して、元の答えをより良く洗練して質問に答えてください。\n"
        "コンテキストが役に立たない場合は、元の回答と同じものを返します。")

    refine_prompt = RefinePrompt(REFINE_PROMPT)

    ################################################################
    query_engine = index.as_query_engine(text_qa_template=qa_prompt, refine_template=refine_prompt)

    # query_engine = index.as_query_engine(text_qa_template=QA_PROMPT)

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
        '質問に近いノードの抽出': get_nearlynode,
        'txtのindex化': make_index,
        'documentsの確認': check_doc,
        'Q&A ※基本的に使用しない': qa_calc
    }
    selected_app_name = st.selectbox(label='項目の選択',
                                                options=list(apps.keys()))


    # 選択されたアプリケーションを処理する関数を呼び出す
    render_func = apps[selected_app_name]
    render_func()

if __name__ == '__main__':
    main()

