import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ----------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------
st.set_page_config(
    page_title="Sales Dynamics Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------------------------------
# PREMIUM STYLING (CSS)
# ----------------------------------------------------
st.markdown("""
<style>
    /* Import modern Google font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Background gradient for the app */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Sleek Typography */
    .main-header {
        font-weight: 800;
        font-size: 4rem;
        background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    
    .sub-header {
        font-weight: 400;
        font-size: 1.3rem;
        color: #94a3b8;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 40px;
    }
    
    /* Style slider tracks */
    .stSlider > div > div > div > div {
        background-color: #38bdf8 !important;
    }
    
    /* Override metric value colors */
    div[data-testid="stMetricValue"] {
        font-weight: 800;
        font-size: 4.5rem !important;
        background: -webkit-linear-gradient(45deg, #10b981, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------
# HEADER
# ----------------------------------------------------
st.markdown("<h1 class='main-header'>Sales Dynamics Pro</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>AI-Driven Polynomial Regression for Advertising ROI</p>", unsafe_allow_html=True)


# ----------------------------------------------------
# MODEL TRAINING (CACHED)
# ----------------------------------------------------
@st.cache_resource
def train_model():
    """Load data and train the degree=3 polynomial regression model."""
    try:
        data = pd.read_csv("advertising.csv.csv")
    except Exception as e:
        # Fallback if there's any path issue
        st.error(f"Failed to load data: {e}")
        return None, None, None

    # Features and target
    X = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']
    
    # Polynomial transformation
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)
    
    # Linear Regression fit
    model = LinearRegression()
    model.fit(X_poly, y)
    
    return model, poly, data

model, poly, data = train_model()


# ----------------------------------------------------
# MAIN LAYOUT
# ----------------------------------------------------
if model is not None:
    # Use columns to separate inputs and output
    col1, col_space, col2 = st.columns([1, 0.1, 1.2])
    
    # ===== LEFT COLUMN: INPUTS =====
    with col1:
        st.subheader("📊 Advertising Budget ($K)")
        st.markdown("<span style='color:#cbd5e1;'>Adjust the strategic budget allocation across channels:</span>", unsafe_allow_html=True)
        st.write("")
        
        tv_budget = st.slider("Television (TV)", 
                               min_value=float(data['TV'].min()), 
                               max_value=350.0, 
                               value=float(data['TV'].mean()), 
                               step=1.0)
        
        radio_budget = st.slider("Radio", 
                                  min_value=float(data['Radio'].min()), 
                                  max_value=70.0, 
                                  value=float(data['Radio'].mean()), 
                                  step=0.5)
        
        news_budget = st.slider("Newspaper", 
                                 min_value=float(data['Newspaper'].min()), 
                                 max_value=120.0, 
                                 value=float(data['Newspaper'].mean()), 
                                 step=1.0)
                                 
    # ===== RIGHT COLUMN: PREDICTIONS =====
    with col2:
        st.subheader("📈 Predicted Sales Return")
        st.markdown("<span style='color:#cbd5e1;'>Real-time AI estimations based on your budget:</span>", unsafe_allow_html=True)
        
        # Inference
        example = np.array([[tv_budget, radio_budget, news_budget]])
        example_poly = poly.transform(example)
        prediction = model.predict(example_poly)[0]
        
        # Display large metric
        st.metric(label="Estimated Sales (in thousands)", value=f"{prediction:.2f}")
        
        st.markdown("---")
        st.markdown("""
        ### 💡 Strategic Insight
        The underlying algorithm utilizes a **3rd-Degree Polynomial Regression Model**. 
        Unlike basic linear models, this architecture captures **diminishing returns**; notice how exponential investments in a single channel (like TV) eventually plateau in generating relative sales growth.
        """)

    # ----------------------------------------------------
    # VISUALIZATION (Bottom Section)
    # ----------------------------------------------------
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.subheader("Diminishing Returns Analysis: TV Spend vs. Sales")
    
    # Prepare plotting data point by keeping Radio & News at their average
    tv_range = np.linspace(data['TV'].min(), data['TV'].max() + 50, 100)
    radio_mean = data['Radio'].mean()
    news_mean = data['Newspaper'].mean()

    X_plot = np.array([[tv, radio_mean, news_mean] for tv in tv_range])
    X_plot_poly = poly.transform(X_plot)
    y_plot = model.predict(X_plot_poly)

    # Styling the Matplotlib Chart for a Dark App context
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_alpha(0.0)      # Transparent figure background
    ax.set_facecolor('none')      # Transparent axes background
    
    # Style spines and ticks
    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_alpha(0.3)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')

    # Draw data
    ax.scatter(data['TV'], data['Sales'], color='#cbd5e1', alpha=0.4, s=30, label='Historical Data', zorder=1)
    ax.plot(tv_range, y_plot, color='#38bdf8', linewidth=3, label='Model Trend (Degree=3)', zorder=2)
    
    # Mark user selection
    ax.axvline(tv_budget, color='#10b981', linestyle=':', linewidth=3, label='Current TV Budget', zorder=3)
    ax.scatter([tv_budget], [prediction], color='#facc15', s=100, label='Predicted Coordinate', zorder=4, edgecolor='black')

    ax.set_xlabel("TV Advertising Budget ($K)")
    ax.set_ylabel("Sales (Thousands)")
    
    # Style legend
    leg = ax.legend(loc="lower right", facecolor='#1e293b', edgecolor='none')
    for text in leg.get_texts():
        text.set_color("white")
        
    st.pyplot(fig)

