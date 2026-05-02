# import streamlit as st
# import matplotlib.pyplot as plt
# from sklearn .feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import PyPDF2
# import re
# from collections import Counter
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk import pos_tag


# # Download NLTK resources

# nltk.download("punkt_tab")
# nltk.download("stopwords")
# nltk.download("averaged_perceptron_tagger_eng")


#                     # Page Setup

# st.set_page_config(page_title="Resume Job Match Scorer",page_icon="📄",layout="wide")

# st.markdown("""
# Upload your resume (PDF) and paste a job description to see how well they match!  
# This tool uses **TF-IDF + Cosine Similarity** to analyze your resume against job requirements.
# """)

# with st.sidebar:
#     st.header("About")
#     st.info("""
#     This tool helps you:
#     - Measures how your resume matches a job description
    
#     """)
#     st.header("How It works")
#     st.write("""
#     1. Upload your resume (PDF)
#     2. Paste the job description
#     3. Click **Analyze Match**
#     """)



#                 # helper function


# def extract_text_from_pdf(uploaded_file):
#     try:
#         pdf_reder=PyPDF2.PdfReader(uploaded_file)
#         text=""
#         for page in pdf_reder.pages:
#             text=text+page.extract_text()
#         return text
#     except Exception as e:
#         st.error(f"Error reading PDF:{e}")
#         return ""
    
    
# def clean_text(text):
#     text=text.lower()
#     text=re.sub(r'[^a-zA-Z\s]','',text)
#     text=re.sub(r'\s+',' ',text).strip()
#     return text


# def  remove_stopwords(text):
#     stop_words=set(stopwords.words('english'))
#     words=word_tokenize(text)
#     return " ".join([word for word in words if word not in stop_words])


# def calculate_similarity(resume_text,job_description):
#     resume_processed=remove_stopwords(clean_text(resume_text))
#     job_processed=remove_stopwords(clean_text(job_description))
#     vectorizer=TfidfVectorizer()
#     tfidf_matrix=vectorizer.fit_transform([resume_processed,job_processed])
#     score=cosine_similarity(tfidf_matrix[0:1],tfidf_matrix[1:2])[0][0]*100
#     return round(score,2),resume_processed,job_processed

# # def extract_keywords(text,num_keywords=10):
# #     words=word_tokenize(text)
# #     words=[w for w in words if len(w)>2]
# #     tagged_words=pos_tag(words)
# #     nouns=[w for w,pos in tagged_words if pos.startswith('NN') or pos.startswith('JJ')]
# #     word_freq=Counter(nouns)
# #     return word_freq.most_common(num_keywords)


#                     # Main aap

# def main():
#     uploaded_file=st.file_uploader("Upload your resuem (PDF)",type=['pdf'])
#     job_description=st.text_area("Paste the job description",height=200)

#     if st.button("Analyze Match"):
#         if not uploaded_file:
#             st.warning("Please upload your resume")
#             return
#         if not job_description:
#             st.warning("Please paste the job description")
#             return
        
        
#         with st.spinner("Analyzing your resume...."):
#             resume_text=extract_text_from_pdf(uploaded_file)
#             if not resume_text:
#                 st.error("could not extract text from pdf. please try another pdf")
#                 return 
            
#             # calculate similarity
#             similarity_score,resume_processed,job_processed=calculate_similarity(resume_text,job_description)

#             # Result
#             st.subheader("Results")
#             st.metric("Match Score",f"{similarity_score:.2f}%")

#             # gauge chart

#             fig,ax=plt.subplots(figsize=(6,0.5))
#             colors=['#ff4b4b','#ffa726','#0f9d58']
#             color_index=min(int(similarity_score//33),2)
#             ax.barh([0],[similarity_score],color=colors[color_index])
#             ax.set_xlim(0,100)
#             ax.set_xlabel("Match percentage")
#             ax.set_yticks([])
#             ax.set_title("Resume Job Match")
#             st.pyplot(fig)

#             if similarity_score<40:
#                 st.warning("Low Match, consider tailoring your resume more closely.")
#             elif similarity_score<70:
#                 st.info("Good Match. Your resume aligin fairly well")
#             else:
#                 st.success("Excellent Match ! Your resume strongly aligns.")

# if __name__=="__main__":
#     main()


import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Download NLTK resources
@st.cache_resource
def download_data():
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("punkt_tab")

download_data()

# Page Setup
st.set_page_config(page_title="Resume Job Match Scorer", page_icon="📄", layout="wide")

st.markdown("""
# Resume Job Match Scorer 
Upload your resume (PDF) and paste a job description. 
""")

with st.sidebar:
    st.header("About")
    st.info("Uses NLP (TF-IDF + POS Tagging) to match skills accurately.")

# --- Helper Functions ---

def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def clean_text(text):
    text = text.lower()
    # Remove special characters but keep numbers (important for versions like Python 3)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_technical_keywords(text):
    """Filters out common words to focus on real technical skills."""
    stop_words = set(stopwords.words('english'))
    # Filler words jo accuracy kharab karte hain
    ignore = {'requirements', 'experience', 'years', 'skills', 'role', 'work', 'candidate', 'ability'}
    
    words = word_tokenize(text.lower())
    filtered = [w for w in words if w.isalnum() and w not in stop_words and w not in ignore]
    
    # POS Tagging: Sirf Nouns (skills) ko priority dena
    tagged = pos_tag(filtered)
    keywords = {w for w, pos in tagged if pos.startswith('NN')}
    return keywords

def calculate_similarity(resume_text, job_description):
    res_clean = clean_text(resume_text)
    jd_clean = clean_text(job_description)
    
    # 1. Contextual Match (TF-IDF with Bi-grams)
    # n-gram range (1,2) helps recognize phrases like "Web Developer"
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([res_clean, jd_clean])
    context_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    # 2. Hard Skill Match (Keyword Overlap)
    res_keys = get_technical_keywords(res_clean)
    jd_keys = get_technical_keywords(jd_clean)
    
    if not jd_keys:
        return 0
        
    matched = res_keys.intersection(jd_keys)
    keyword_score = len(matched) / len(jd_keys)
    
    # Final Weightage: 70% Skills Match + 30% Contextual Match
    final_score = (keyword_score * 0.7) + (context_score * 0.3)
    return round(final_score * 100, 2)

# --- Main App ---

def main():
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Upload your resume (PDF)", type=['pdf'])
    with col2:
        job_description = st.text_area("Paste the job description", height=200)

    if st.button("Analyze Match"):
        if not uploaded_file or not job_description:
            st.warning("Please upload both resume and job description.")
            return
        
        with st.spinner("Analyzing..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            if not resume_text:
                return 

            similarity_score = calculate_similarity(resume_text, job_description)

            # Result Display
            st.subheader("Analysis Result")
            st.metric("Final Accuracy Match", f"{similarity_score}%")

            # Gauge Chart
            fig, ax = plt.subplots(figsize=(6, 0.5))
            color = '#ff4b4b' if similarity_score < 40 else '#ffa726' if similarity_score < 70 else '#0f9d58'
            
            ax.barh([0], [similarity_score], color=color)
            ax.set_xlim(0, 100)
            ax.set_yticks([])
            ax.set_xlabel("Match Percentage")
            st.pyplot(fig)

            if similarity_score < 40:
                st.error("Low Match: Skill gap identified. Try adding more keywords from the JD.")
            elif similarity_score < 70:
                st.info("Good Match: Your profile aligns well with major requirements.")
            else:
                st.success("Excellent Match: Strong alignment with the job profile!")

if __name__ == "__main__":
    main()