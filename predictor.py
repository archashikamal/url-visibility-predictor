"""
============================================================================
URL-BASED VISIBILITY PREDICTOR - STREAMLIT APP
Extracts features from URLs and compares with visibility standards
============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import time

# Page config
st.set_page_config(
    page_title="URL Visibility Predictor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .feature-gap {
        background-color: #fff3cd;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #ffc107;
        border-radius: 0.3rem;
    }
    .feature-good {
        background-color: #d4edda;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
        border-radius: 0.3rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_models():
    """Load trained models, standards, and metadata"""
    try:
        model_dir = Path("model_outputs")
        
        with open(model_dir / "stage1_classifier.pkl", "rb") as f:
            clf = pickle.load(f)
        
        with open(model_dir / "stage2_regressor.pkl", "rb") as f:
            reg = pickle.load(f)
        
        with open(model_dir / "stage1_features.pkl", "rb") as f:
            features_s1 = pickle.load(f)
        
        with open(model_dir / "stage2_features.pkl", "rb") as f:
            features_s2 = pickle.load(f)
        
        with open(model_dir / "model_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        
        with open(model_dir / "feature_standards.pkl", "rb") as f:
            standards = pickle.load(f)
        
        with open(model_dir / "feature_thresholds.pkl", "rb") as f:
            thresholds = pickle.load(f)
        
        with open(model_dir / "core_extractable_features.pkl", "rb") as f:
            core_features = pickle.load(f)
        
        return clf, reg, features_s1, features_s2, metadata, standards, thresholds, core_features
    
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.info("Please ensure model_outputs/ directory contains all required files")
        st.stop()

# ============================================================================
# FEATURE EXTRACTION FROM URL
# ============================================================================

def extract_basic_features_from_url(url, query):
    """
    Extract basic features from URL (for demo/testing)
    This is a simplified version - you can enhance with real scraping
    """
    try:
        # Parse URL
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Try to fetch the page
        try:
            response = requests.get(url, timeout=5, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Word count
            word_count = len(text.split())
            
            # Simple relevance calculation (keyword matching)
            query_words = query.lower().split()
            text_lower = text.lower()
            relevance = sum(word in text_lower for word in query_words) / len(query_words) if query_words else 0
            relevance = min(relevance, 1.0)
            
        except Exception as e:
            st.warning(f"Could not fetch URL content: {e}. Using default values.")
            word_count = 500
            relevance = 0.6
        
        # Domain-based influence estimation
        influence = estimate_domain_influence(domain)
        
        # Calculate other features
        features = {
            'Relevance': relevance,
            'Influence': influence,
            'Uniqueness': 0.7,  # Default - would need content analysis
            'Click_Probability': min(0.5 + (relevance * 0.3) + (influence * 0.2), 1.0),
            'Diversity': 0.6,  # Default
            'WC': word_count,
            'Subjective_Position': 1,  # Assuming position 1 for single URL analysis
            'Subjective_Count': 10,  # Default
            'WC_rel': 1.0,  # Default
            'query_length': len(query),
            'query_type_list': 0,
            'query_type_opinion': 0,
            'query_type_other': 1,
            'num_sources': 10,
            'is_suggested_source': 0,
            'domain_freq': 1,
            'avg_PAWC_source': 50.0,
            'Influence_x_Position': influence * 1,
            'Relevance_x_Uniqueness': relevance * 0.7,
        }
        
        # Add rank features (all 1.0 for single source)
        features.update({
            'Influence_rank': 1.0,
            'Relevance_rank': 1.0,
            'Uniqueness_rank': 1.0,
            'Click_Prob_rank': 1.0,
            'Diversity_rank': 1.0
        })
        
        # Add composite features
        features['Quality_Score'] = (
            features['Relevance'] * 0.4 +
            features['Influence'] * 0.3 +
            features['Uniqueness'] * 0.3
        )
        features['Position_weighted_Influence'] = features['Influence'] / (features['Subjective_Position'] + 1)
        features['Click_Prob_rel'] = 1.0
        features['Source_Density'] = 1 / (features['num_sources'] + 1)
        features['Domain_Popularity'] = features['domain_freq'] * features['avg_PAWC_source']
        features['PAWC_rank'] = 1.0
        features['PAWC_pct'] = 1.0
        features['WC_x_Relevance'] = features['WC'] * features['Relevance']
        
        return features, None
        
    except Exception as e:
        return None, str(e)

def estimate_domain_influence(domain):
    """
    Estimate domain influence based on known patterns
    In production, use a domain authority API or database
    """
    # High authority domains
    high_authority = ['wikipedia.org', 'edu', 'gov', 'nih.gov', 'nature.com', 'science.org']
    medium_authority = ['medium.com', 'forbes.com', 'techcrunch.com', 'nytimes.com']
    
    domain_lower = domain.lower()
    
    if any(ha in domain_lower for ha in high_authority):
        return np.random.uniform(0.85, 0.95)
    elif any(ma in domain_lower for ma in medium_authority):
        return np.random.uniform(0.65, 0.80)
    else:
        return np.random.uniform(0.45, 0.65)

# ============================================================================
# PREDICTION & COMPARISON FUNCTIONS
# ============================================================================

def predict_visibility(features_dict, clf, reg, features_s1, features_s2):
    """Two-stage prediction"""
    # Stage 1
    X1 = np.array([features_dict[f] for f in features_s1]).reshape(1, -1)
    is_visible = bool(clf.predict(X1)[0])
    visibility_prob = float(clf.predict_proba(X1)[0][1])
    
    # Stage 2
    pawc_score = None
    if is_visible:
        X2 = np.array([features_dict[f] for f in features_s2]).reshape(1, -1)
        log_pawc = reg.predict(X2)[0]
        pawc_score = float(np.expm1(log_pawc))
    
    return is_visible, visibility_prob, pawc_score

def compare_with_standards(features_dict, standards, thresholds, core_features):
    """Compare extracted features with visibility standards"""
    comparison = {}
    gaps = []
    
    for feature in core_features:
        if feature in features_dict and feature in standards:
            actual_val = features_dict[feature]
            
            # Get target value
            if isinstance(standards[feature], dict):
                target_val = standards[feature].get('75th_percentile') or standards[feature].get('mean', 0)
            else:
                target_val = standards[feature]
            
            gap_pct = ((target_val - actual_val) / target_val) * 100 if target_val > 0 else 0
            
            comparison[feature] = {
                'actual': actual_val,
                'target': target_val,
                'gap_percentage': gap_pct,
                'meets_threshold': actual_val >= thresholds.get(feature, 0),
                'status': '✅' if gap_pct <= 10 else '⚠️' if gap_pct <= 30 else '❌'
            }
            
            if gap_pct > 10:
                gaps.append({
                    'feature': feature,
                    'actual': actual_val,
                    'target': target_val,
                    'gap': gap_pct
                })
    
    gaps.sort(key=lambda x: x['gap'], reverse=True)
    
    return comparison, gaps

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown('<p class="main-header">🔍 URL Visibility Predictor</p>', unsafe_allow_html=True)
    st.markdown("### Analyze your website's visibility potential based on learned standards")
    
    # Load models
    with st.spinner("Loading models and standards..."):
        clf, reg, features_s1, features_s2, metadata, standards, thresholds, core_features = load_models()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        st.metric("Visibility Threshold", f"{metadata['visibility_threshold']:.2f}")
        st.metric("Stage 1 ROC AUC", f"{metadata['stage1_roc_auc']:.3f}")
        st.metric("Stage 2 R²", f"{metadata['stage2_r2']:.3f}")
        
        st.divider()
        
        st.subheader("🎯 Visibility Standards")
        st.info(f"""
        **Core Features Analyzed:**
        {len(core_features)} extractable features
        
        **Standards Based On:**
        Top 25% of visible sources
        """)
        
        with st.expander("View Standards"):
            for feature in core_features:
                if feature in standards:
                    if isinstance(standards[feature], dict):
                        target = standards[feature].get('75th_percentile', 0)
                    else:
                        target = standards[feature]
                    st.write(f"**{feature}:** {target:.3f}")
    
    # Main content
    st.divider()
    
    # Input section
    st.header("🌐 Enter URL and Query")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        url = st.text_input(
            "Website URL",
            placeholder="https://example.com/article",
            help="Enter the full URL of the webpage to analyze"
        )
    
    with col2:
        query = st.text_input(
            "Search Query",
            placeholder="machine learning tutorial",
            help="The search query this page would appear for"
        )
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        use_mock = st.checkbox(
            "Use mock data (for testing without internet)",
            value=False,
            help="Generate realistic mock features without fetching the URL"
        )
    
    # Analyze button
    if st.button("🔍 Analyze Visibility", type="primary", use_container_width=True):
        if not url or not query:
            st.error("Please enter both URL and query")
            return
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Extract features
        status_text.text("🔍 Step 1/4: Extracting features from URL...")
        progress_bar.progress(25)
        time.sleep(0.5)
        
        if use_mock:
            # Mock features for testing
            features = {
                'Relevance': 0.75,
                'Influence': 0.70,
                'Uniqueness': 0.65,
                'Click_Probability': 0.60,
                'Diversity': 0.55,
                'WC': 450,
                'Subjective_Position': 1
            }
            # Add all required features
            features.update({
                'Subjective_Count': 10, 'WC_rel': 1.0, 'query_length': len(query),
                'query_type_list': 0, 'query_type_opinion': 0, 'query_type_other': 1,
                'num_sources': 10, 'is_suggested_source': 0, 'domain_freq': 1,
                'avg_PAWC_source': 50.0, 'Influence_x_Position': 0.70,
                'Relevance_x_Uniqueness': 0.49, 'Influence_rank': 1.0,
                'Relevance_rank': 1.0, 'Uniqueness_rank': 1.0, 'Click_Prob_rank': 1.0,
                'Diversity_rank': 1.0, 'Quality_Score': 0.70,
                'Position_weighted_Influence': 0.35, 'Click_Prob_rel': 1.0,
                'Source_Density': 0.09, 'Domain_Popularity': 50.0,
                'PAWC_rank': 1.0, 'PAWC_pct': 1.0, 'WC_x_Relevance': 337.5
            })
            error = None
        else:
            features, error = extract_basic_features_from_url(url, query)
        
        if error:
            st.error(f"❌ Error extracting features: {error}")
            return
        
        # Step 2: Compare with standards
        status_text.text("📊 Step 2/4: Comparing with visibility standards...")
        progress_bar.progress(50)
        time.sleep(0.5)
        
        comparison, gaps = compare_with_standards(features, standards, thresholds, core_features)
        
        # Step 3: Predict visibility
        status_text.text("🎯 Step 3/4: Predicting visibility...")
        progress_bar.progress(75)
        time.sleep(0.5)
        
        is_visible, visibility_prob, pawc_score = predict_visibility(
            features, clf, reg, features_s1, features_s2
        )
        
        # Step 4: Complete
        status_text.text("✅ Step 4/4: Analysis complete!")
        progress_bar.progress(100)
        time.sleep(0.3)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # ====================================================================
        # RESULTS DISPLAY
        # ====================================================================
        
        st.divider()
        st.header("📊 Analysis Results")
        
        # Main prediction
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if is_visible:
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown("### ✅ VISIBLE")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="danger-box">', unsafe_allow_html=True)
                st.markdown("### ❌ NOT VISIBLE")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.metric("Visibility Probability", f"{visibility_prob:.1%}")
        
        with col3:
            if pawc_score:
                st.metric("Predicted PAWC Score", f"{pawc_score:.1f}")
            else:
                st.metric("Predicted PAWC Score", "N/A")
        
        # Feature comparison
        st.divider()
        st.subheader("🎯 Feature Analysis")
        
        # Summary metrics
        total_features = len(comparison)
        features_meeting = sum(1 for c in comparison.values() if c['gap_percentage'] <= 10)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Features Analyzed", total_features)
        with col2:
            st.metric("Meeting Target", f"{features_meeting}/{total_features}")
        with col3:
            pct_meeting = (features_meeting / total_features * 100) if total_features > 0 else 0
            st.metric("Success Rate", f"{pct_meeting:.0f}%")
        
        # Detailed comparison table
        st.subheader("📋 Detailed Feature Comparison")
        
        comparison_data = []
        for feature, values in comparison.items():
            comparison_data.append({
                'Feature': feature,
                'Your Value': f"{values['actual']:.3f}",
                'Target (75th %ile)': f"{values['target']:.3f}",
                'Gap': f"{values['gap_percentage']:.1f}%",
                'Status': values['status']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, height=300)
        
        # Gaps and recommendations
        if gaps:
            st.divider()
            st.subheader("🔴 Features Below Target (Improvement Needed)")
            
            for i, gap in enumerate(gaps[:5], 1):  # Show top 5 gaps
                gap_pct = gap['gap']
                
                if gap_pct > 30:
                    box_class = "danger-box"
                    icon = "🔴"
                elif gap_pct > 10:
                    box_class = "warning-box"
                    icon = "⚠️"
                else:
                    continue
                
                st.markdown(f'<div class="{box_class}">', unsafe_allow_html=True)
                st.markdown(f"""
                **{icon} {i}. {gap['feature']}**
                - Your value: `{gap['actual']:.3f}`
                - Target value: `{gap['target']:.3f}`
                - Gap: `{gap_pct:.1f}%` below target
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Recommendations
            st.divider()
            st.subheader("💡 Recommendations")
            
            recommendations = generate_recommendations(gaps, is_visible)
            for rec in recommendations:
                st.info(rec)
        
        else:
            st.success("✅ All features meet or exceed visibility standards! Great job!")
        
        # Extracted features (expandable)
        with st.expander("🔍 View All Extracted Features"):
            extracted_df = pd.DataFrame([
                {'Feature': k, 'Value': f"{v:.3f}" if isinstance(v, float) else v}
                for k, v in features.items()
                if k in core_features
            ])
            st.dataframe(extracted_df, use_container_width=True)

def generate_recommendations(gaps, is_visible):
    """Generate actionable recommendations based on gaps"""
    recommendations = []
    
    if not is_visible:
        recommendations.append("🎯 **Primary Goal:** Your page is predicted as NOT VISIBLE. Focus on the gaps below to improve visibility.")
    else:
        recommendations.append("✅ **Good News:** Your page is predicted as VISIBLE, but you can still improve these areas:")
    
    for gap in gaps[:3]:
        feature = gap['feature']
        
        if feature == 'Relevance':
            recommendations.append(f"📝 **Improve Relevance ({gap['gap']:.0f}% gap):** Add more keywords related to your target query. Ensure content directly answers the search intent.")
        elif feature == 'Influence':
            recommendations.append(f"🏆 **Build Influence ({gap['gap']:.0f}% gap):** Get backlinks from authoritative sites. Improve domain authority through quality content and citations.")
        elif feature == 'Uniqueness':
            recommendations.append(f"✨ **Increase Uniqueness ({gap['gap']:.0f}% gap):** Add original insights, data, or perspectives. Avoid duplicate content.")
        elif feature == 'WC':
            recommendations.append(f"📄 **Expand Content ({gap['gap']:.0f}% gap):** Target word count should be around {gap['target']:.0f} words. Add more detailed explanations and examples.")
        elif feature == 'Click_Probability':
            recommendations.append(f"👆 **Improve Clickability ({gap['gap']:.0f}% gap):** Optimize title and meta description. Make them compelling and relevant.")
        elif feature == 'Diversity':
            recommendations.append(f"🎨 **Add Diversity ({gap['gap']:.0f}% gap):** Include various content types (text, images, videos). Cover multiple sub-topics.")
    
    return recommendations

if __name__ == "__main__":
    main()
