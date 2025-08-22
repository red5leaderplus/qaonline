import streamlit as st
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# default page configuration on main block
st.set_page_config(page_title="中文客服檢索回覆", page_icon="💬", layout="wide")
st.title('中文客服檢索服務')

q = st.text_input("請輸入問題(中文)",placeholder="例如:如何退貨?")
query = st.button("送出")
result_placeholder = st.empty()
st.markdown('---')
if "status_label" not in st.session_state:
    st.session_state.status_label = "狀態：初始化Q&A資料庫"
if "status_state" not in st.session_state:
    st.session_state.status_state = "running"
status_box = st.status(st.session_state.status_label, state=st.session_state.status_state)

DEFAULT_FAQ = pd.DataFrame(
    [
        {"question":"你們的營業時間是？","answer":"我們的客服時間為週一至週五 09:00–18:00（國定假日除外）。"},
        {"question":"如何申請退貨？","answer":"請於到貨 7 天內透過訂單頁面點選『申請退貨』，系統將引導您完成流程。"},
        {"question":"運費如何計算？","answer":"單筆訂單滿 NT$ 1000 免運，未滿則酌收 NT$ 80。"},
        {"question":"可以開立發票嗎？","answer":"我們提供電子發票，請於結帳時填寫統一編號與抬頭。"},
    ]
)

if "faq_df" not in st.session_state:
    st.session_state.faq_df = DEFAULT_FAQ.copy()
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "tfidf" not in st.session_state:
    st.session_state.tfidf = None

with st.expander('檢視現有Q&A資料', expanded=False):
    st.dataframe(st.session_state.faq_df, use_container_width=True)

# Sidebar for knowledge base operations
with st.sidebar:
    st.header("知識庫操作")
    uploader = st.file_uploader('限上傳CSV檔', type=['csv'])
    if uploader is not None:
        df = pd.read_csv(uploader)
        st.session_state.faq_df = df.dropna().reset_index(drop=True)
        st.success(f'已成功載入{len(df)}筆資料。')
    do_index = st.button("更新Q&A索引")
    st.markdown("---")
    st.header("參數設置")
    top_k = st.slider("取得前k筆回答", 1, 3, value=2, key="top_k_slider")
    c = st.slider("信心門檻", 0.0, 1.0, value=0.3, key="confidence_slider")

# Function to tokenize text using jieba
def jieba_tokenize(text:str):
    return list(jieba.cut(text))#將句子分詞

if st.session_state.vectorizer is None:
    st.session_state.status_label = "狀態：尚未建立索引，會自動建立"
    st.session_state.status_state = "running"
    status_box.update(label=st.session_state.status_label, state=st.session_state.status_state)
    corpus = (st.session_state.faq_df["question"].astype(str)+
              " "+
              st.session_state.faq_df["answer"].astype(str)).tolist()
    v = TfidfVectorizer(tokenizer=jieba_tokenize)
    tfidf = v.fit_transform(corpus)
    st.session_state.vectorizer = v
    st.session_state.tfidf = tfidf
    st.session_state.status_label = "狀態：已成功載入預設的Q&A資料"
    st.session_state.status_state = "complete"
    status_box.update(label=st.session_state.status_label, state=st.session_state.status_state)
elif do_index:
    corpus = (st.session_state.faq_df["question"].astype(str)+
              " "+
              st.session_state.faq_df["answer"].astype(str)).tolist()
    v = TfidfVectorizer(tokenizer=jieba_tokenize)
    tfidf = v.fit_transform(corpus)
    st.session_state.vectorizer = v
    st.session_state.tfidf = tfidf
    st.session_state.status_label = "狀態：已成功載入新上傳的Q&A資料"
    st.session_state.status_state = "complete"
    status_box.update(label=st.session_state.status_label, state=st.session_state.status_state)

# query processing
if query and q.strip():
    st.session_state.status_label = "查詢中..."
    st.session_state.status_state = "running"
    status_box.update(label=st.session_state.status_label, state=st.session_state.status_state)
    corpus = (st.session_state.faq_df["question"].astype(str)+
              " "+
              st.session_state.faq_df["answer"].astype(str)).tolist()
    v = TfidfVectorizer(tokenizer=jieba_tokenize)
    tfidf = v.fit_transform(corpus)
    st.session_state.vectorizer = v
    st.session_state.tfidf = tfidf

    vec = st.session_state.vectorizer.transform([q])
    sims = linear_kernel(vec, st.session_state.tfidf).flatten()
    idxc =sims.argsort()[::-1][:top_k]
    rows = st.session_state.faq_df.iloc[idxc].copy()
    rows['score']=sims[idxc]

    best_ans = None
    best_score = float(rows['score'].iloc[0]) if len(rows) else 0.0
    with result_placeholder.container():
        if best_score >= c:
            best_ans = rows['answer'].iloc[0]
            if best_ans:
                st.markdown(
                """
                <div style="display: flex; justify-content: space-between;">
                    <span>最佳查詢結果：</span>
                    <span>按正下方圖示可複製回答</span>
                </div>
                """,
                unsafe_allow_html=True
            )
                st.code(best_ans, language='text')
                st.session_state.status_label = "狀態：查詢完成"
                st.session_state.status_state = "complete"
                status_box.update(label=st.session_state.status_label, state=st.session_state.status_state)
        else:
            st.info('找不到適合的回應。')
            st.session_state.status_label = "狀態：查詢完成，但無合適回應"
            st.session_state.status_state = "complete"
            status_box.update(label=st.session_state.status_label, state=st.session_state.status_state)
        st.write("候選查詢結果：")
        st.dataframe(rows[['question', 'answer', 'score']], use_container_width=True)
