import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from wordcloud import WordCloud
import re
import string
import plotly.express as px
import hashlib

# Download necessary NLTK data
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('maxent_ne_chunker_tab')

# Company description
company_description = """
AgriGrow Solutions is a leading provider of innovative agricultural products and technologies.
We specialize in precision farming, organic fertilizers, and AI-driven crop monitoring systems.
Our products include high-yield seeds, automated irrigation systems, and drone-based soil analysis.
With a commitment to sustainability, we help farmers maximize productivity while conserving resources.
"""


products = [
    {
        "name": "High-yield Hybrid Seeds",
        "price": "$25 per kg",
        "units_sold": "50,000+",
        "rating": "4.7/5",
        "description": "These seeds offer 30% higher yield and are resistant to common pests."
    },
    {
        "name": "Organic Fertilizers",
        "price": "$18 per bag (50kg)",
        "units_sold": "75,000+",
        "rating": "4.8/5",
        "description": "Eco-friendly fertilizers that enhance soil fertility and improve crop health."
    },
    {
        "name": "AI-powered Crop Monitoring System",
        "price": "$999 per unit",
        "units_sold": "10,000+",
        "rating": "4.9/5",
        "description": "Uses AI to analyze crop health, detect diseases, and suggest best farming practices."
    },
    {
        "name": "Automated Irrigation System",
        "price": "$1,500 per system",
        "units_sold": "8,500+",
        "rating": "4.6/5",
        "description": "Smart irrigation that optimizes water usage based on real-time weather conditions."
    },
    {
        "name": "Drone-based Soil Analysis",
        "price": "$3,000 per drone",
        "units_sold": "5,000+",
        "rating": "4.9/5",
        "description": "Advanced drones that scan soil health and provide detailed fertility reports."
    },
    {
        "name": "Smart Weather Prediction Tools",
        "price": "$199 per subscription",
        "units_sold": "20,000+",
        "rating": "4.5/5",
        "description": "Accurate weather forecasts for better planning and risk management."
    },
    {
        "name": "Pest Control Biopesticides",
        "price": "$40 per bottle (1L)",
        "units_sold": "30,000+",
        "rating": "4.7/5",
        "description": "Natural and chemical-free pesticides that effectively control pests without harming crops."
    }
]

# Create a single string containing all product descriptions
all_product_descriptions = company_description + " " + " ".join([p["name"] + " " + p["description"] for p in products])

# Text preprocessing functions
def preprocess_text(text):
    """Clean and preprocess text for NLP analysis"""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(f'[{string.punctuation}]', ' ', text)
    # Remove numbers
    text = re.sub(r'\d+', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def lemmatize_text(text):
    """Lemmatize text using NLTK"""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return " ".join([lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 1])

def get_keywords(text, top_n=10):
    """Extract top keywords from text using TF-IDF"""
    vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    
    # Get scores for each word
    scores = tfidf_matrix.toarray()[0]
    
    # Get top N keywords
    top_indices = np.argsort(scores)[-top_n:][::-1]
    top_keywords = [(feature_names[i], scores[i]) for i in top_indices]
    
    return top_keywords

def get_named_entities(text):
    """Extract named entities from text using NLTK"""
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    named_entities = ne_chunk(tagged)
    
    entities = []
    # Process named entities
    for chunk in named_entities:
        if hasattr(chunk, 'label'):
            entity_text = ' '.join([word for word, tag in chunk])
            entity_type = chunk.label()
            entities.append((entity_text, entity_type))
    
    return entities

def create_word_vectors(text_list, vocabulary=None):
    """Create TF-IDF vectors for text comparison"""
    # If no vocabulary provided, create it from the text list
    if vocabulary is None:
        vectorizer = TfidfVectorizer(stop_words='english')
        vectorizer.fit(text_list)
        vocabulary = vectorizer.vocabulary_
    
    # Create vectorizer with fixed vocabulary
    vectorizer = TfidfVectorizer(stop_words='english', vocabulary=vocabulary)
    vectors = vectorizer.fit_transform(text_list)
    
    return vectors, vocabulary

def calculate_similarity(query, company_text):
    """Calculate similarity between query and company text using TF-IDF and cosine similarity"""
    # Preprocess texts
    processed_query = preprocess_text(query)
    processed_company = preprocess_text(company_text)
    
    # Create vectors using TF-IDF
    vectors, _ = create_word_vectors([processed_query, processed_company])
    query_vector = vectors[0]
    company_vector = vectors[1]
    
    # Calculate cosine similarity
    similarity_score = cosine_similarity(query_vector, company_vector)[0][0]
    return similarity_score+0.50

def recommend_products(query, products_list):
    """Recommend products based on similarity to query"""
    # Preprocess query
    processed_query = preprocess_text(query)
    
    # Create list of product texts
    product_texts = [preprocess_text(p["name"] + " " + p["description"]) for p in products_list]
    all_texts = [processed_query] + product_texts
    
    # Create vectors
    vectors, _ = create_word_vectors(all_texts)
    query_vector = vectors[0]
    product_vectors = vectors[1:]
    
    # Calculate similarities
    similarities = []
    for i, product_vector in enumerate(product_vectors):
        similarity = cosine_similarity(query_vector, product_vector)[0][0]
        similarities.append((products_list[i], similarity))
    
    # Sort by similarity score
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities

def main():
    st.title("AgriGrow Solutions - NLP Query Analyzer")
    
    # Sidebar
    st.sidebar.header("About AgriGrow Solutions")
    st.sidebar.write(company_description)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Query Analysis", "Company NLP Analysis", "Product Explorer", "NLP Visualization"])
    
    # Tab 1: Query Analysis
    with tab1:
        st.header("Query Relevance Analysis")
        st.write("Enter a query to check its relevance to AgriGrow Solutions")
        
        query = st.text_area("Your Query:", height=100, 
                             value="I am looking for AI solutions for farming and smart irrigation systems.")
        
        if st.button("Analyze Query"):
            with st.spinner("Analyzing query..."):
                # Calculate similarity
                similarity_score = calculate_similarity(query, all_product_descriptions)
                
                # Display results
                st.subheader("Results:")
                st.metric("Relevance Score", f"{similarity_score:.2f}")
                
                # Interpretation
                if similarity_score > 0.8:
                    st.success("Highly relevant to our company! We have products that match your needs.")
                elif similarity_score > 0.6:
                    st.info("Moderately relevant to our company. We may have products that interest you.")
                elif similarity_score > 0.4:
                    st.warning("Somewhat relevant to our company. Some aspects may align with our offerings.")
                else:
                    st.error("Not very relevant to our company focus areas.")
                
                # Extract keywords from query
                processed_query = preprocess_text(query)
                keywords = get_keywords(processed_query, top_n=5)
                st.subheader("Key terms in your query:")
                for keyword, score in keywords:
                    st.write(f"- {keyword}: {score:.4f}")
                
                # Recommend products
                st.subheader("Recommended Products:")
                recommended = recommend_products(query, products)
                
                for i, (product, score) in enumerate(recommended[:3]):
                    with st.container():
                        st.write(f"**{i+1}. {product['name']}** (Relevance: {score:.2f})")
                        st.write(f"Price: {product['price']}")
                        st.write(f"Description: {product['description']}")
                        st.write("---")
    
    # Tab 2: Company NLP Analysis
    with tab2:
        st.header("AgriGrow Solutions NLP Analysis")
        
        # Preprocess company text
        processed_company_text = preprocess_text(all_product_descriptions)
        lemmatized_text = lemmatize_text(processed_company_text)
        
        # Extract and display keywords
        st.subheader("Top Keywords")
        keywords = get_keywords(processed_company_text, top_n=10)
        keywords_df = pd.DataFrame(keywords, columns=["Keyword", "Score"])
        
        # Plot keywords
        fig = px.bar(keywords_df, x="Keyword", y="Score", title="Key Terms in Company Description")
        st.plotly_chart(fig)
        
        # Extract named entities
        st.subheader("Named Entities")
        entities = get_named_entities(company_description)
        
        # Group entities by type
        entity_types = {}
        for entity, label in entities:
            if label not in entity_types:
                entity_types[label] = []
            entity_types[label].append(entity)
        
        # Display entities
        for label, entities_list in entity_types.items():
            st.write(f"**{label}**: {', '.join(entities_list)}")
    
    # Tab 3: Product Explorer
    with tab3:
        st.header("AgriGrow Products")
        
        # Convert products to DataFrame for easier display
        products_df = pd.DataFrame(products)
        
        # Display products in a table
        st.dataframe(products_df[["name", "price", "rating", "units_sold"]])
        
        # Allow filtering by product category
        product_categories = ["All"] + list(set([p["name"].split()[0] for p in products]))
        selected_category = st.selectbox("Filter by Category:", product_categories)
        
        if selected_category != "All":
            filtered_products = [p for p in products if p["name"].startswith(selected_category)]
        else:
            filtered_products = products
        
        # Display filtered products
        for product in filtered_products:
            with st.expander(f"{product['name']} - {product['price']} (Rating: {product['rating']})"):
                st.write(f"**Description:** {product['description']}")
                st.write(f"**Units Sold:** {product['units_sold']}")
    
    # Tab 4: NLP Visualization
    with tab4:
        st.header("NLP Visualizations")
        
        # Generate word cloud
        st.subheader("Word Cloud of Company and Product Descriptions")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_product_descriptions)
        
        # Display the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
        
        # N-gram analysis
        st.subheader("Common Word Combinations (Bigrams)")
        
        # Get bigrams
        tokens = word_tokenize(processed_company_text)
        bigrams = list(nltk.bigrams(tokens))
        
        # Count frequencies
        bigram_freq = {}
        for b in bigrams:
            if b[0] not in stopwords.words('english') and b[1] not in stopwords.words('english'):
                bigram = ' '.join(b)
                if bigram in bigram_freq:
                    bigram_freq[bigram] += 1
                else:
                    bigram_freq[bigram] = 1
        
        # Convert to DataFrame and display top bigrams
        bigram_df = pd.DataFrame(list(bigram_freq.items()), columns=['Bigram', 'Frequency'])
        bigram_df = bigram_df.sort_values('Frequency', ascending=False).head(10)
        
        # Plot bigrams
        fig = px.bar(bigram_df, x='Bigram', y='Frequency', title='Top 10 Bigrams')
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
