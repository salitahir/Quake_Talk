# 🌍 Quake-Talk 4.0  
**Tremors to Tweets: A GPT-4o Powered Application to Improving Disaster Response Analytical Frameworks**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-orange?logo=streamlit)](https://quake-talk.streamlit.app/)     [![LinkedIn](https://img.shields.io/badge/Author-LinkedIn-blue?logo=linkedin)](https://www.linkedin.com/in/salitahir/)     [![Email](https://img.shields.io/badge/Email-s.ali.tahir%40outlook.com-blue?logo=microsoft-outlook)](mailto:s.ali.tahir@outlook.com)


---

## 📌 Project Overview

Quake-Talk 4.0 is an interactive disaster analytics tool designed as a prototype to extract **real-time, actionable insights** from user-generated social media content during humanitarian crises. This pipeline combines **semantic search**, **keyword filtering**, and **GPT-4o-powered Q&A** to mine information from over **478 000 tweets** related to the **2023 Türkiye–Syria Earthquake**.  

📎 [Slide Deck: Overview & Business Applications](https://www.canva.com/design/DAGlyTsyFjo/MHYjTbNINhCSEOmxVzpXAA/view?utm_content=DAGlyTsyFjo&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=hd2aea126e2)

---

## 🚀 Key Features

| Component               | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| **Preprocessing**       | Tokenization, regex-based cleaning, lemmatization, stopword removal         |
| **Semantic Search**     | all-MiniLM-L6-v2 embeddings + FAISS retrieval                                |
| **Keyword Filtering**   | Transparent rule-based matching with lemmatization support                   |
| **GPT-4o Integration**  | Contextualized answers & summaries with temperature control                 |
| **User Interface**      | Streamlit UI with filters, context viewers & expandable summaries           |
| **Live Querying**       | Ask specific questions like “What supplies are needed in Malatya?”           |

---

## 🧠 Architecture

```mermaid
flowchart TD
    A[User Question] --> B{Filter Method}
    B -->|Keyword| C[Regex + Lemmatizer]
    B -->|Semantic| D[Cosine Similarity Search]
    C & D --> E[Context Builder]
    E --> F[Token Truncation]
    F --> G[GPT-4o QA Engine]
    G --> H[Answer + Summary Output]
````
---

## 🗃️ Dataset

| File                     | Description                           |
|--------------------------|---------------------------------------|
| `tweets.csv`             | Raw social media posts (~478 k tweets) |
| `tweets_english.csv`     | English-language subset               |
| `tweets_cleaned.parquet` | Cleaned & lemmatized data for pipeline |
| `tweet_embeddings.npy`   | Precomputed 384-dim semantic vectors   |
| `tweets.index`           | FAISS index for fast similarity search |

_Sources: Tweets via `snscrape` (X/Twitter)_

---

## 🎯 Use Cases

- **Disaster Response & Relief**  
  Map help requests, resource needs & sentiment shifts in near-real time  
- **Sentiment & Trend Analysis**  
  Track how public mood evolves over phases of the crisis  
- **NGO / CSR Insights**  
  Gauge volunteer & donor sentiment to optimize engagement  
- **Academic & Policy Research**  
  Support crisis informatics studies and evidence-based decision making  

---

## 📈 Example Queries

> 🔍 “What medical supplies are most requested in Gaziantep?”  
> 🔍 “How did people react to the second aftershock?”  
> 🔍 “Which areas express frustration with aid delivery delays?”  

Each query returns:  
1. 💬 **GPT-4o Answer** rooted in mined tweets  
2. 📝 **Optional Summary** of key themes  
3. 🔍 **Raw Context** sentences  

---

## 💻 Tech Stack

- **Python 3.11**  
- **Streamlit** – interactive dashboard  
- **Polars** – high-performance DataFrames  
- **spaCy & NLTK** – text preprocessing & lemmatization  
- **Sentence-Transformers** & **faiss-cpu** – embeddings & vector search  
- **tiktoken** – precise token counting  
- **OpenAI GPT-4o** – Q&A and summarization  

---

## 🚀 Local Setup

```bash
git clone https://github.com/salitahir/Quake_Talk.git
cd Quake_Talk
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Copy example env and fill in your API keys:
cp .env.example .env

streamlit run app.py
````
---

## 📚 Citation & Credit

**Syed Ali Tahir** >
✉️ [tahirsy@tcd.ie](mailto:tahirsy@tcd.ie) >
✉️ [s.ali.tahir@outlook.com](mailto:s.ali.tahir@outlook.com) >
🔗 [LinkedIn: syed-ali-tahir](https://www.linkedin.com/in/salitahir/)

© 2025 Syed Ali Tahir. All rights reserved. No redistribution without permission.  
